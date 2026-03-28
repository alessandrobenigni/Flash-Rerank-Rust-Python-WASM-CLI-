//! Parallel sub-batch scorer for CPU inference.
//!
//! Splits large batches across N ORT sessions, each pinned to a subset of CPU cores.
//! This bypasses ORT's single-session thread contention and delivers near-linear
//! scaling on multi-core CPUs.

use std::path::Path;
use std::thread;

use crate::Result;
use crate::calibrate::{Calibrator, SigmoidCalibrator};
use crate::engine::Scorer;
use crate::tokenize::Tokenizer;
use crate::types::{ModelConfig, RerankResult};

use ort::execution_providers::CPU;
use ort::session::Session;
use ort::value::Tensor;

/// A scorer that runs inference in parallel across multiple ORT sessions.
///
/// Each session is owned by a dedicated OS thread and fed sub-batches via channels.
/// Best for CPU inference where a single ORT session can't saturate all cores.
pub struct ParallelScorer {
    config: ModelConfig,
    workers: Vec<WorkerHandle>,
    tokenizer: std::sync::Mutex<Tokenizer>,
    num_workers: usize,
}

struct WorkerHandle {
    tx: std::sync::mpsc::Sender<WorkerRequest>,
}

struct WorkerRequest {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
    batch_size: usize,
    seq_len: usize,
    reply: std::sync::mpsc::Sender<WorkerResult>,
}

type WorkerResult = std::result::Result<Vec<f32>, String>;

impl ParallelScorer {
    /// Create a new parallel scorer with `num_workers` ORT sessions.
    ///
    /// Each worker gets `total_threads / num_workers` intra-op threads.
    /// Default: 4 workers with `available_parallelism / 4` threads each.
    pub fn new(config: ModelConfig, model_dir: &Path, num_workers: Option<usize>) -> Result<Self> {
        let total_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let n_workers = num_workers.unwrap_or_else(|| (total_threads / 2).max(2).min(8));
        let threads_per_worker = (total_threads / n_workers).max(1);

        let model_path = super::ort_backend::OrtScorer::resolve_model_path(model_dir, &config);
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)?;

        let mut workers = Vec::with_capacity(n_workers);

        for worker_id in 0..n_workers {
            let (tx, rx) = std::sync::mpsc::channel::<WorkerRequest>();
            let model_path = model_path.clone();
            let tpw = threads_per_worker;

            thread::Builder::new()
                .name(format!("flash-rerank-worker-{worker_id}"))
                .spawn(move || {
                    worker_loop(rx, &model_path, tpw);
                })
                .map_err(|e| {
                    crate::Error::Inference(format!("Failed to spawn worker {worker_id}: {e}"))
                })?;

            workers.push(WorkerHandle { tx });
        }

        tracing::info!(
            num_workers = n_workers,
            threads_per_worker,
            model = %model_path.display(),
            "ParallelScorer initialized"
        );

        Ok(Self {
            config,
            workers,
            tokenizer: std::sync::Mutex::new(tokenizer),
            num_workers: n_workers,
        })
    }
}

fn worker_loop(rx: std::sync::mpsc::Receiver<WorkerRequest>, model_path: &Path, threads: usize) {
    // Create a session with limited threads for this worker
    let build_result = (|| -> std::result::Result<Session, ort::Error> {
        let mut builder = Session::builder()?;
        builder = builder.with_intra_threads(threads)?;
        builder = builder.with_execution_providers([CPU::default().build()])?;
        builder.commit_from_file(model_path)
    })();

    let mut session = match build_result {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("Worker failed to create session: {e}");
            while let Ok(req) = rx.recv() {
                let _ = req.reply.send(Err(format!("Session init failed: {e}")));
            }
            return;
        }
    };

    while let Ok(req) = rx.recv() {
        let result = run_inference(
            &mut session,
            &req.input_ids,
            &req.attention_mask,
            &req.token_type_ids,
            req.batch_size,
            req.seq_len,
        );
        let _ = req.reply.send(result);
    }
}

fn run_inference(
    session: &mut Session,
    input_ids: &[i64],
    attention_mask: &[i64],
    token_type_ids: &[i64],
    batch_size: usize,
    seq_len: usize,
) -> WorkerResult {
    let shape = [batch_size as i64, seq_len as i64];

    let ids =
        Tensor::from_array((shape, input_ids.to_vec())).map_err(|e| format!("tensor: {e}"))?;
    let mask =
        Tensor::from_array((shape, attention_mask.to_vec())).map_err(|e| format!("tensor: {e}"))?;
    let tids =
        Tensor::from_array((shape, token_type_ids.to_vec())).map_err(|e| format!("tensor: {e}"))?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => ids,
            "attention_mask" => mask,
            "token_type_ids" => tids,
        ])
        .map_err(|e| format!("inference: {e}"))?;

    let (_shape, logits) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("extract: {e}"))?;

    Ok(logits.iter().copied().collect())
}

impl Scorer for ParallelScorer {
    fn score(&self, query: &str, documents: &[String]) -> Result<Vec<RerankResult>> {
        if query.is_empty() {
            return Err(crate::Error::Inference("Empty query".to_string()));
        }
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // 1. Tokenize all pairs at once
        let encodings = {
            let mut tokenizer = self
                .tokenizer
                .lock()
                .map_err(|e| crate::Error::Tokenizer(format!("lock: {e}")))?;
            tokenizer.tokenize_pairs(query, documents, self.config.max_length)?
        };

        let total = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        // 2. Split into sub-batches and dispatch to workers
        let chunk_size = (total + self.num_workers - 1) / self.num_workers;
        let mut receivers = Vec::new();

        for (worker_idx, chunk) in encodings.chunks(chunk_size).enumerate() {
            let batch_size = chunk.len();
            let input_ids: Vec<i64> = chunk
                .iter()
                .flat_map(|e| e.get_ids().iter().map(|&id| id as i64))
                .collect();
            let attention_mask: Vec<i64> = chunk
                .iter()
                .flat_map(|e| e.get_attention_mask().iter().map(|&m| m as i64))
                .collect();
            let token_type_ids: Vec<i64> = chunk
                .iter()
                .flat_map(|e| e.get_type_ids().iter().map(|&t| t as i64))
                .collect();

            let (reply_tx, reply_rx) = std::sync::mpsc::channel();

            let worker = &self.workers[worker_idx % self.num_workers];
            worker
                .tx
                .send(WorkerRequest {
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    batch_size,
                    seq_len,
                    reply: reply_tx,
                })
                .map_err(|e| crate::Error::Inference(format!("Worker send failed: {e}")))?;

            receivers.push(reply_rx);
        }

        // 3. Collect results from all workers
        let calibrator = SigmoidCalibrator;
        let mut all_results = Vec::with_capacity(total);
        let mut global_idx = 0;

        for rx in receivers {
            let logits = rx
                .recv()
                .map_err(|e| crate::Error::Inference(format!("Worker recv failed: {e}")))?
                .map_err(|e| crate::Error::Inference(e))?;

            for logit in logits {
                all_results.push(RerankResult {
                    index: global_idx,
                    score: calibrator.calibrate(logit),
                    document: None,
                });
                global_idx += 1;
            }
        }

        // 4. Sort by score descending
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(all_results)
    }
}
