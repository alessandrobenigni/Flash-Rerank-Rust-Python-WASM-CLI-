//! Python bindings for Flash-Rerank via PyO3.
//!
//! Provides `load()` to create a `Reranker`, which wraps an `OrtScorer`
//! behind a `Mutex` for thread-safe inference with GIL release.

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyTuple};

use ::flash_rerank::engine::Scorer;
use ::flash_rerank::engine::ort_backend::OrtScorer;
use ::flash_rerank::models::ModelRegistry;
use ::flash_rerank::types::{Device, ModelConfig, Precision};

// ---------------------------------------------------------------------------
// load() — top-level function
// ---------------------------------------------------------------------------

/// Load a reranking model, returning a `Reranker` instance.
///
/// Arguments:
///   model_id: HuggingFace model identifier (e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2")
///   device: "cpu", "cuda", or "cuda:<N>" (default "cpu")
///   precision: "fp32", "fp16", "int8", or "int4" (default "fp32")
#[pyfunction]
#[pyo3(signature = (model_id, device="cpu", precision="fp32"))]
fn load(py: Python, model_id: &str, device: &str, precision: &str) -> PyResult<Reranker> {
    // Capture owned copies before entering allow_threads (no Python refs allowed).
    let model_id_owned = model_id.to_string();
    let device_owned = device.to_string();
    let precision_owned = precision.to_string();

    py.allow_threads(move || {
        let device = parse_device(&device_owned)?;
        let precision = parse_precision(&precision_owned)?;

        let config = ModelConfig {
            model_id: model_id_owned.clone(),
            precision,
            device,
            ..Default::default()
        };

        // Resolve model from HuggingFace Hub cache
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
            .join("huggingface/hub");
        let registry = ModelRegistry::new(cache_dir);
        let model_dir = registry
            .load(&model_id_owned)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let scorer = OrtScorer::new(config, &model_dir)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Reranker {
            scorer: Some(Arc::new(Mutex::new(scorer))),
        })
    })
}

// ---------------------------------------------------------------------------
// Reranker class
// ---------------------------------------------------------------------------

/// Python-facing reranker class.
///
/// Holds an `OrtScorer` behind `Arc<Mutex<_>>` so it can be shared with
/// async tasks while remaining thread-safe.  All inference calls release
/// the GIL via `py.allow_threads()`.
#[pyclass]
struct Reranker {
    scorer: Option<Arc<Mutex<OrtScorer>>>,
}

#[pymethods]
impl Reranker {
    /// Rerank documents for a query, returning a list of (index, score) tuples
    /// sorted by descending score.
    ///
    /// Arguments:
    ///   query: The search query.
    ///   documents: List of document strings to rerank.
    ///   top_k: If set, return only the top-k results.
    #[pyo3(signature = (query, documents, top_k=None))]
    fn rerank(
        &self,
        py: Python,
        query: &str,
        documents: Vec<String>,
        top_k: Option<usize>,
    ) -> PyResult<Vec<(usize, f32)>> {
        let scorer_arc = self
            .scorer
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Reranker has been unloaded"))?
            .clone();

        // Own the query before releasing the GIL.
        let query_owned = query.to_string();

        py.allow_threads(move || {
            let scorer = scorer_arc
                .lock()
                .map_err(|e| PyRuntimeError::new_err(format!("Scorer lock poisoned: {e}")))?;

            let mut results = scorer
                .score(&query_owned, &documents)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            if let Some(k) = top_k {
                results.truncate(k);
            }

            Ok(results.into_iter().map(|r| (r.index, r.score)).collect())
        })
    }

    /// Hybrid reranking: combine BM25 scores with neural reranking via RRF.
    ///
    /// Builds a BM25 ranked list from the provided scores, obtains neural
    /// scores via cross-encoder inference, then fuses both lists using
    /// Reciprocal Rank Fusion (RRF).
    ///
    /// Arguments:
    ///   query: The search query.
    ///   documents: List of document strings to rerank.
    ///   bm25_scores: BM25 scores corresponding to each document.
    ///   alpha: Weight for the neural ranker (BM25 weight = 1 - alpha). Default 0.5.
    #[pyo3(signature = (query, documents, bm25_scores, alpha=None))]
    fn rerank_hybrid(
        &self,
        py: Python,
        query: &str,
        documents: Vec<String>,
        bm25_scores: Vec<f32>,
        alpha: Option<f32>,
    ) -> PyResult<Vec<(usize, f32)>> {
        let scorer_arc = self
            .scorer
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Reranker has been unloaded"))?
            .clone();

        let query_owned = query.to_string();
        let alpha = alpha.unwrap_or(0.5);

        py.allow_threads(move || {
            let scorer = scorer_arc
                .lock()
                .map_err(|e| PyRuntimeError::new_err(format!("Scorer lock poisoned: {e}")))?;

            // Build BM25 ranked list sorted by descending score
            let mut bm25_ranked: Vec<(usize, f32)> = bm25_scores
                .iter()
                .enumerate()
                .map(|(i, &s)| (i, s))
                .collect();
            bm25_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Get neural scores via cross-encoder
            let neural_results = scorer
                .score(&query_owned, &documents)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let neural_ranked: Vec<(usize, f32)> = neural_results
                .into_iter()
                .map(|r| (r.index, r.score))
                .collect();

            // Fuse via RRF with alpha weighting
            let config = ::flash_rerank::fusion::FusionConfig {
                k: 60,
                weights: vec![1.0 - alpha, alpha],
            };
            let fused = ::flash_rerank::fusion::rrf_fusion(&[bm25_ranked, neural_ranked], &config);

            Ok(fused)
        })
    }

    /// Async rerank for use with Python asyncio.
    ///
    /// Returns a coroutine that runs the blocking rerank in a thread pool
    /// executor via `asyncio.get_event_loop().run_in_executor(None, fn)`.
    /// This releases the GIL during inference so the asyncio event loop
    /// can process other tasks concurrently.
    #[pyo3(signature = (query, documents, top_k=None))]
    fn arerank<'py>(
        &self,
        py: Python<'py>,
        query: String,
        documents: Vec<String>,
        top_k: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let scorer_arc = self
            .scorer
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Reranker has been unloaded"))?
            .clone();

        // Build a Python callable that performs the blocking rerank.
        // When called from run_in_executor, it will run on a thread-pool
        // thread, and allow_threads releases the GIL during inference.
        let rerank_fn = PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| {
                let py = args.py();
                let scorer_ref = scorer_arc.clone();
                let q = query.clone();
                let docs = documents.clone();
                let k = top_k;

                py.allow_threads(move || {
                    let scorer = scorer_ref.lock().map_err(|e| {
                        PyRuntimeError::new_err(format!("Scorer lock poisoned: {e}"))
                    })?;

                    let mut results = scorer
                        .score(&q, &docs)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                    if let Some(k) = k {
                        results.truncate(k);
                    }

                    Ok::<Vec<(usize, f32)>, PyErr>(
                        results.into_iter().map(|r| (r.index, r.score)).collect(),
                    )
                })
            },
        )?;

        // Return: asyncio.get_running_loop().run_in_executor(None, rerank_fn)
        // Try get_running_loop first (Python 3.10+), fall back to get_event_loop
        let asyncio = py.import("asyncio")?;
        let event_loop = match asyncio.call_method0("get_running_loop") {
            Ok(loop_) => loop_.to_owned(),
            Err(_) => asyncio.call_method0("get_event_loop")?.to_owned(),
        };
        let future = event_loop.call_method1("run_in_executor", (py.None(), rerank_fn))?;
        Ok(future)
    }

    /// Unload the model, releasing memory.
    ///
    /// Subsequent calls to `rerank()` or `arerank()` will raise `RuntimeError`.
    fn unload(&mut self) -> PyResult<()> {
        self.scorer = None;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pipeline class — two-stage BM25 + neural reranking
// ---------------------------------------------------------------------------

/// Two-stage retrieval pipeline: BM25 first-stage retrieval followed by
/// neural reranking.
///
/// The `bm25_index` must be any Python object with a `.search(query, k)`
/// method that returns a list of document strings.
#[pyclass]
struct Pipeline {
    reranker: Py<Reranker>,
    bm25_index: PyObject,
}

#[pymethods]
impl Pipeline {
    /// Create a new Pipeline.
    ///
    /// Arguments:
    ///   bm25_index: Python object with a `.search(query, k)` method.
    ///   reranker: A loaded `Reranker` instance.
    #[new]
    #[pyo3(signature = (bm25_index, reranker))]
    fn new(bm25_index: PyObject, reranker: Py<Reranker>) -> Self {
        Self {
            reranker,
            bm25_index,
        }
    }

    /// Run the two-stage search pipeline.
    ///
    /// 1. Retrieve `retrieve` candidate documents via BM25.
    /// 2. Neural-rerank the candidates and return the top `top_k`.
    ///
    /// Arguments:
    ///   query: The search query.
    ///   top_k: Number of results to return (default 10).
    ///   retrieve: Number of BM25 candidates to fetch (default 100).
    #[pyo3(signature = (query, top_k=10, retrieve=100))]
    fn search(
        &self,
        py: Python,
        query: &str,
        top_k: Option<usize>,
        retrieve: Option<usize>,
    ) -> PyResult<Vec<(usize, f32)>> {
        let retrieve_k = retrieve.unwrap_or(100);

        // Step 1: BM25 retrieval — call into Python
        let bm25_results: Vec<String> = self
            .bm25_index
            .call_method1(py, "search", (query, retrieve_k))?
            .extract(py)?;

        // Step 2: Neural reranking
        let reranker = self.reranker.borrow(py);
        reranker.rerank(py, query, bm25_results, top_k)
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Flash-Rerank Python module.
#[pymodule]
fn flash_rerank(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_class::<Reranker>()?;
    m.add_class::<Pipeline>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a device string into a `Device` enum variant.
fn parse_device(s: &str) -> PyResult<Device> {
    match s {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda(0)),
        d if d.starts_with("cuda:") => {
            let idx = d[5..]
                .parse::<usize>()
                .map_err(|_| PyValueError::new_err(format!("Invalid CUDA device index: {d}")))?;
            Ok(Device::Cuda(idx))
        }
        d if d.starts_with("tensorrt") => {
            let idx = if d.contains(':') {
                d.split(':')
                    .nth(1)
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0)
            } else {
                0
            };
            Ok(Device::TensorRT(idx))
        }
        _ => Err(PyValueError::new_err(format!("Unknown device: {s}"))),
    }
}

/// Parse a precision string into a `Precision` enum variant.
fn parse_precision(s: &str) -> PyResult<Precision> {
    match s {
        "fp32" => Ok(Precision::FP32),
        "fp16" => Ok(Precision::FP16),
        "int8" => Ok(Precision::INT8),
        "int4" => Ok(Precision::INT4),
        _ => Err(PyValueError::new_err(format!("Unknown precision: {s}"))),
    }
}
