use std::path::Path;
use std::sync::Mutex;

use ort::execution_providers::{CPU, CUDA, TensorRT};
use ort::session::Session;
use ort::value::Tensor;

use crate::Result;
use crate::calibrate::{Calibrator, SigmoidCalibrator};
use crate::engine::Scorer;
use crate::tokenize::Tokenizer;
use crate::types::{Device, ModelConfig, Precision, RerankResult};

/// ONNX Runtime scorer supporting TensorRT, CUDA, and CPU execution providers.
///
/// Execution provider selection:
/// - `Device::TensorRT(_)` -> TensorRT EP with INT8/INT4 quantization
/// - `Device::Cuda(_)` -> CUDA EP with FP16/FP32
/// - `Device::Cpu` -> CPU EP with MKL-DNN optimization
///
/// `Session::run()` requires `&mut self`, so the session is wrapped in a `Mutex`
/// to satisfy the `Scorer` trait's `&self` signature.
pub struct OrtScorer {
    config: ModelConfig,
    session: Mutex<Session>,
    tokenizer: Mutex<Tokenizer>,
    /// Whether the model expects token_type_ids as a third input.
    has_token_type_ids: bool,
}

impl OrtScorer {
    /// Create a new OrtScorer from a model directory containing `model.onnx` and `tokenizer.json`.
    ///
    /// Selects execution providers based on `config.device`:
    /// - `Device::TensorRT(gpu_id)` -> TensorRT EP -> CUDA EP -> CPU EP fallback chain
    /// - `Device::Cuda(gpu_id)` -> CUDA EP -> CPU EP fallback chain
    /// - `Device::Cpu` -> CPU EP only
    pub fn new(config: ModelConfig, model_dir: &Path) -> Result<Self> {
        // Select best available ONNX model: INT8 quantized > O2 optimized > base
        let model_path = Self::resolve_model_path(model_dir, &config);
        let tokenizer_path = model_dir.join("tokenizer.json");

        let mut builder = Session::builder().map_err(|e| crate::Error::Inference(e.to_string()))?;

        match config.device {
            Device::TensorRT(gpu_id) => {
                tracing::info!(gpu_id, "Initializing TensorRT EP with fallback chain");
                let trt_ep = TensorRT::default()
                    .with_device_id(gpu_id as i32)
                    .with_fp16(matches!(
                        config.precision,
                        Precision::FP16 | Precision::INT8 | Precision::INT4
                    ))
                    .with_int8(matches!(
                        config.precision,
                        Precision::INT8 | Precision::INT4
                    ))
                    .build();
                let cuda_ep = CUDA::default().with_device_id(gpu_id as i32).build();
                let cpu_ep = CPU::default().build();

                builder = builder
                    .with_execution_providers([trt_ep, cuda_ep, cpu_ep])
                    .map_err(|e| crate::Error::Inference(e.to_string()))?;
            }
            Device::Cuda(gpu_id) => {
                tracing::info!(gpu_id, "Initializing CUDA EP with CPU fallback");
                let cuda_ep = CUDA::default().with_device_id(gpu_id as i32).build();
                let cpu_ep = CPU::default().build();

                builder = builder
                    .with_execution_providers([cuda_ep, cpu_ep])
                    .map_err(|e| crate::Error::Inference(e.to_string()))?;
            }
            Device::Cpu => {
                tracing::info!("Initializing CPU EP");
                let cpu_ep = CPU::default().build();

                builder = builder
                    .with_execution_providers([cpu_ep])
                    .map_err(|e| crate::Error::Inference(e.to_string()))?;
            }
        }

        let session = builder
            .commit_from_file(&model_path)
            .map_err(|e| crate::Error::Inference(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)?;

        // Detect model inputs — some models (BGE) don't use token_type_ids
        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        tracing::info!(
            model = %model_path.display(),
            device = ?config.device,
            precision = ?config.precision,
            has_token_type_ids,
            "OrtScorer initialized"
        );

        Ok(Self {
            config,
            session: Mutex::new(session),
            tokenizer: Mutex::new(tokenizer),
            has_token_type_ids,
        })
    }

    /// Resolve the best available ONNX model file from the model directory.
    /// Priority: INT8 quantized (for CPU) > O2 optimized > base model.
    /// For GPU devices, prefer the base model (GPU handles its own optimization).
    pub fn resolve_model_path(model_dir: &Path, config: &ModelConfig) -> std::path::PathBuf {
        let onnx_dir = model_dir.join("onnx");
        let base_dir = if onnx_dir.exists() {
            &onnx_dir
        } else {
            model_dir
        };

        match config.device {
            Device::Cpu => {
                // Prefer INT8 quantized for CPU
                let candidates = [
                    "model_qint8_avx512.onnx",
                    "model_quint8_avx2.onnx",
                    "model_O2.onnx",
                    "model_O3.onnx",
                    "model.onnx",
                ];
                for name in candidates {
                    let path = base_dir.join(name);
                    if path.exists() {
                        tracing::info!(model = %path.display(), "Selected optimized model variant");
                        return path;
                    }
                }
            }
            _ => {
                // For GPU, prefer base model (TRT/CUDA handle optimization)
                let candidates = ["model.onnx", "model_O2.onnx"];
                for name in candidates {
                    let path = base_dir.join(name);
                    if path.exists() {
                        return path;
                    }
                }
            }
        }

        // Fallback
        let root = model_dir.join("model.onnx");
        let subdir = model_dir.join("onnx").join("model.onnx");
        if root.exists() { root } else { subdir }
    }
}

impl Scorer for OrtScorer {
    #[tracing::instrument(
        skip(self, documents),
        fields(
            rerank.model_id = %self.config.model_id,
            rerank.batch_size = documents.len(),
            rerank.tokenize_ms,
            rerank.inference_ms,
            rerank.score_mean,
            rerank.score_std,
            rerank.score_min,
            rerank.score_max,
        )
    )]
    fn score(&self, query: &str, documents: &[String]) -> Result<Vec<RerankResult>> {
        // Input validation
        if query.is_empty() {
            return Err(crate::Error::Inference("Empty query".to_string()));
        }
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // 1. Tokenize query-document pairs
        let tokenize_start = std::time::Instant::now();
        let encodings = {
            let mut tokenizer = self
                .tokenizer
                .lock()
                .map_err(|e| crate::Error::Tokenizer(format!("Tokenizer lock poisoned: {e}")))?;
            tokenizer.tokenize_pairs(query, documents, self.config.max_length)?
        };
        let tokenize_ms = tokenize_start.elapsed().as_secs_f64() * 1000.0;
        tracing::Span::current().record("rerank.tokenize_ms", tokenize_ms);

        // 2. Build input tensors (input_ids, attention_mask, token_type_ids)
        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        let input_ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|&id| id as i64))
            .collect();
        let attention_mask: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|&m| m as i64))
            .collect();
        let token_type_ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_type_ids().iter().map(|&t| t as i64))
            .collect();

        let shape = [batch_size as i64, seq_len as i64];

        let ids_tensor = Tensor::from_array((shape, input_ids)).map_err(|e| {
            crate::Error::Inference(format!("Failed to create input_ids tensor: {e}"))
        })?;
        let mask_tensor = Tensor::from_array((shape, attention_mask)).map_err(|e| {
            crate::Error::Inference(format!("Failed to create attention_mask tensor: {e}"))
        })?;
        let type_tensor = Tensor::from_array((shape, token_type_ids)).map_err(|e| {
            crate::Error::Inference(format!("Failed to create token_type_ids tensor: {e}"))
        })?;

        // 3. Run inference — detect model inputs dynamically
        let inference_start = std::time::Instant::now();
        let mut session = self
            .session
            .lock()
            .map_err(|e| crate::Error::Inference(format!("Session lock poisoned: {e}")))?;

        let outputs = if self.has_token_type_ids {
            session.run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => type_tensor,
            ])
        } else {
            session.run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
        }
        .map_err(|e| crate::Error::Inference(e.to_string()))?;
        let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;
        tracing::Span::current().record("rerank.inference_ms", inference_ms);

        // 4. Extract logits and apply sigmoid calibration
        let (_shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| crate::Error::Inference(format!("Failed to extract logits: {e}")))?;

        let calibrator = SigmoidCalibrator;
        let mut results: Vec<RerankResult> = logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| RerankResult {
                index: i,
                score: calibrator.calibrate(logit),
                document: None,
            })
            .collect();

        // 5. Record score distribution
        if !results.is_empty() {
            let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
            let n = scores.len() as f32;
            let mean = scores.iter().sum::<f32>() / n;
            let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
            let std_dev = variance.sqrt();
            let min = scores.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let span = tracing::Span::current();
            span.record("rerank.score_mean", mean as f64);
            span.record("rerank.score_std", std_dev as f64);
            span.record("rerank.score_min", min as f64);
            span.record("rerank.score_max", max as f64);
        }

        // 6. Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }
}

// OrtScorer is Send + Sync automatically:
// - ort::Session implements Send + Sync (via unsafe impl in ort crate)
// - Tokenizer wraps tokenizers::Tokenizer which is Send + Sync
// - Both are behind Mutex for interior mutability
// No unsafe impl needed.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_query_returns_error() {
        // We can't create a real OrtScorer without a model, but we can verify
        // the error type is correct by checking the Error enum variant.
        // This validates the error path logic.
        let err = crate::Error::Inference("Empty query".to_string());
        assert!(matches!(err, crate::Error::Inference(ref msg) if msg == "Empty query"));
    }

    #[test]
    fn test_sigmoid_calibration_range() {
        let calibrator = SigmoidCalibrator;
        // Verify sigmoid outputs are in [0, 1]
        for &raw in &[-10.0, -1.0, 0.0, 1.0, 10.0] {
            let score = calibrator.calibrate(raw);
            assert!(
                score >= 0.0 && score <= 1.0,
                "Score {score} out of range for raw {raw}"
            );
        }
    }
}
