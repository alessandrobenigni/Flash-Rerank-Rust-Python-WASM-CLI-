//! Inference backends for cross-encoder and ColBERT models.
//!
//! This module provides the [`Scorer`] trait and concrete implementations:
//! - [`ort_backend::OrtScorer`] -- ONNX Runtime with CPU, CUDA, and TensorRT providers
//! - [`parallel::ParallelScorer`] -- CPU parallel sub-batch scoring
//! - [`colbert::ColBertScorer`] -- ColBERT late interaction scoring
//! - [`tensorrt::TrtScorer`] -- TensorRT-optimized inference

pub mod colbert;
pub mod ort_backend;
pub mod parallel;
pub mod tensorrt;

use crate::{RerankResult, Result};

/// Trait for scoring query-document pairs.
pub trait Scorer: Send + Sync {
    /// Score a query against a batch of documents, returning ranked results.
    fn score(&self, query: &str, documents: &[String]) -> Result<Vec<RerankResult>>;
}
