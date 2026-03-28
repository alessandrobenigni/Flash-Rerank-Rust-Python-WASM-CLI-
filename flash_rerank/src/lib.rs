//! Flash-Rerank -- Blazing-fast neural reranking engine.
//!
//! Provides cross-encoder and ColBERT inference via ONNX Runtime with
//! TensorRT, CUDA, and CPU execution providers.

pub mod batch;
pub mod calibrate;
pub mod cascade;
pub mod engine;
pub mod fusion;
pub mod models;
pub mod multi_gpu;
pub mod tokenize;
pub mod types;

pub use types::{
    CacheMetadata, Device, ModelConfig, ModelFile, ModelManifest, Precision, RerankConfig,
    RerankRequest, RerankResult, ScorerType,
};

use thiserror::Error;

/// Top-level error type for flash_rerank.
#[derive(Debug, Error)]
pub enum Error {
    #[error("model error: {0}")]
    Model(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("download error: {0}")]
    Download(String),

    #[error("cache error: {0}")]
    Cache(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("calibration error: {0}")]
    Calibration(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Result alias for flash_rerank operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Convenience function: create an OrtScorer from a model directory and score documents.
///
/// This is the simplest way to use flash-rerank. It creates an `OrtScorer`,
/// scores the documents, applies `top_k` filtering, and optionally attaches
/// document text to results.
///
/// # Arguments
/// * `model_dir` - Path to directory containing `model.onnx` and `tokenizer.json`
/// * `config` - Model configuration (device, precision, max_length)
/// * `request` - Reranking request (query, documents, top_k, return_documents)
///
/// # Example
/// ```no_run
/// use flash_rerank::{rerank, ModelConfig, RerankRequest};
/// use std::path::Path;
///
/// let config = ModelConfig::default();
/// let request = RerankRequest {
///     query: "what is machine learning?".to_string(),
///     documents: vec!["ML is a subset of AI".to_string()],
///     top_k: None,
///     return_documents: false,
/// };
/// let results = rerank(Path::new("models/my-model"), &config, &request).unwrap();
/// ```
pub fn rerank(
    model_dir: &std::path::Path,
    config: &ModelConfig,
    request: &RerankRequest,
) -> Result<Vec<RerankResult>> {
    use engine::Scorer;
    use engine::ort_backend::OrtScorer;

    let scorer = OrtScorer::new(config.clone(), model_dir)?;
    let mut results = scorer.score(&request.query, &request.documents)?;

    // Apply top_k filter
    if let Some(top_k) = request.top_k {
        results.truncate(top_k);
    }

    // Attach document text if requested
    if request.return_documents {
        for result in &mut results {
            if result.index < request.documents.len() {
                result.document = Some(request.documents[result.index].clone());
            }
        }
    }

    Ok(results)
}
