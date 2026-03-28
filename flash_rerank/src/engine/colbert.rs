use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Mutex;

use lru::LruCache;
use ort::session::Session;
use ort::value::Tensor;

use crate::Result;
use crate::engine::Scorer;
use crate::tokenize::Tokenizer;
use crate::types::{ModelConfig, RerankResult};

/// ColBERT late-interaction scorer using MaxSim.
///
/// Pre-computes per-document token embeddings, then scores via MaxSim
/// (maximum similarity) between query and document token embeddings.
/// Document embeddings are cached in an LRU cache to avoid re-encoding.
pub struct ColBERTScorer {
    session: Mutex<Session>,
    tokenizer: Mutex<Tokenizer>,
    max_query_length: usize,
    max_doc_length: usize,
    embedding_cache: Mutex<LruCache<String, Vec<Vec<f32>>>>,
}

impl ColBERTScorer {
    /// Create a new ColBERT scorer from a model directory.
    ///
    /// The model directory should contain `model.onnx` (ColBERT ONNX model)
    /// and `tokenizer.json` (HuggingFace tokenizer config).
    ///
    /// # Arguments
    /// * `model_dir` - Path to directory containing model files
    /// * `config` - Model configuration (device, precision, max_length)
    /// * `cache_capacity` - Maximum number of document embeddings to cache
    pub fn new(model_dir: &Path, config: &ModelConfig, cache_capacity: usize) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let session = Session::builder()
            .map_err(|e| crate::Error::Inference(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| crate::Error::Inference(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)?;

        let cap = NonZeroUsize::new(cache_capacity.max(1)).expect("cache_capacity is at least 1");

        tracing::info!(
            model = %model_path.display(),
            max_length = config.max_length,
            cache_capacity,
            "ColBERTScorer initialized"
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: Mutex::new(tokenizer),
            max_query_length: config.max_length,
            max_doc_length: config.max_length,
            embedding_cache: Mutex::new(LruCache::new(cap)),
        })
    }

    /// Encode a text string into per-token embeddings via the ColBERT model.
    ///
    /// Tokenizes the text, runs the model, and extracts the per-token embedding
    /// matrix from the model output.
    fn encode(&self, text: &str, max_length: usize) -> Result<Vec<Vec<f32>>> {
        let encoding = {
            let mut tokenizer = self
                .tokenizer
                .lock()
                .map_err(|e| crate::Error::Tokenizer(format!("Tokenizer lock poisoned: {e}")))?;

            // Encode as single sequence (not a pair) for ColBERT
            // Use tokenize_pairs with a dummy empty second sequence to get proper encoding,
            // or encode directly. For ColBERT we encode query and doc separately.
            let truncation = tokenizers::TruncationParams {
                max_length,
                ..Default::default()
            };
            tokenizer
                .inner_mut()
                .with_truncation(Some(truncation))
                .map_err(|e| crate::Error::Tokenizer(e.to_string()))?;

            tokenizer
                .inner_ref()
                .encode(text, true)
                .map_err(|e| crate::Error::Tokenizer(e.to_string()))?
        };

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let seq_len = input_ids.len();

        let shape = [1i64, seq_len as i64];

        let ids_tensor = Tensor::from_array((shape, input_ids)).map_err(|e| {
            crate::Error::Inference(format!("Failed to create input_ids tensor: {e}"))
        })?;
        let mask_tensor = Tensor::from_array((shape, attention_mask)).map_err(|e| {
            crate::Error::Inference(format!("Failed to create attention_mask tensor: {e}"))
        })?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| crate::Error::Inference(format!("Session lock poisoned: {e}")))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(|e| crate::Error::Inference(e.to_string()))?;

        // Output shape: [1, seq_len, embedding_dim]
        let (output_shape, raw_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| crate::Error::Inference(format!("Failed to extract embeddings: {e}")))?;

        let output_seq_len = output_shape[1] as usize;
        let embedding_dim = output_shape[2] as usize;

        // Convert flat tensor to Vec<Vec<f32>> (one vector per token)
        let embeddings: Vec<Vec<f32>> = (0..output_seq_len)
            .map(|t| {
                let start = t * embedding_dim;
                let end = start + embedding_dim;
                raw_data[start..end].to_vec()
            })
            .collect();

        Ok(embeddings)
    }

    /// Encode a query into per-token embeddings.
    fn encode_query(&self, query: &str) -> Result<Vec<Vec<f32>>> {
        self.encode(query, self.max_query_length)
    }

    /// Encode a document into per-token embeddings, with LRU caching.
    ///
    /// On cache hit, returns the cached embeddings without re-running the model.
    /// On cache miss, encodes the document and stores the result in the cache.
    fn encode_document(&self, doc: &str) -> Result<Vec<Vec<f32>>> {
        // Check cache first
        {
            let mut cache = self
                .embedding_cache
                .lock()
                .map_err(|e| crate::Error::Inference(format!("Cache lock poisoned: {e}")))?;

            if let Some(cached) = cache.get(doc) {
                return Ok(cached.clone());
            }
        }

        // Cache miss -- encode the document
        let embeddings = self.encode(doc, self.max_doc_length)?;

        // Store in cache
        {
            let mut cache = self
                .embedding_cache
                .lock()
                .map_err(|e| crate::Error::Inference(format!("Cache lock poisoned: {e}")))?;

            cache.put(doc.to_string(), embeddings.clone());
        }

        Ok(embeddings)
    }

    /// Compute MaxSim between query and document token embeddings.
    ///
    /// For each query token, find the maximum cosine similarity to any document
    /// token, then sum across all query tokens. This is the ColBERT late
    /// interaction scoring function.
    fn maxsim(query_embeddings: &[Vec<f32>], doc_embeddings: &[Vec<f32>]) -> f32 {
        let mut total = 0.0f32;
        for q_emb in query_embeddings {
            let max_sim = doc_embeddings
                .iter()
                .map(|d_emb| cosine_similarity(q_emb, d_emb))
                .fold(f32::NEG_INFINITY, f32::max);
            // If doc_embeddings is empty, max_sim stays NEG_INFINITY; clamp to 0
            if max_sim.is_finite() {
                total += max_sim;
            }
        }
        total
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero norm (avoids division by zero).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

impl Scorer for ColBERTScorer {
    fn score(&self, query: &str, documents: &[String]) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Encode query once
        let query_embeddings = self.encode_query(query)?;

        // Encode each document (cache-aware) and compute MaxSim
        let mut results: Vec<RerankResult> = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let doc_embeddings = self.encode_document(doc)?;
                let raw_score = Self::maxsim(&query_embeddings, &doc_embeddings);
                // Apply sigmoid calibration to produce scores in [0.0, 1.0]
                let calibrated = 1.0 / (1.0 + (-raw_score).exp());
                Ok(RerankResult {
                    index: i,
                    score: calibrated,
                    document: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }
}

// ColBERTScorer is Send + Sync automatically:
// - ort::Session implements Send + Sync (via unsafe impl in ort crate)
// - Tokenizer wraps tokenizers::Tokenizer which is Send + Sync
// - LruCache<String, Vec<Vec<f32>>> is Send + Sync
// - All three are behind Mutex for interior mutability
// No unsafe impl needed.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical vectors should have similarity 1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "Orthogonal vectors should have similarity 0.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - (-1.0)).abs() < 1e-6,
            "Opposite vectors should have similarity -1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Zero vector should produce similarity 0.0");
    }

    #[test]
    fn test_maxsim_known_embeddings() {
        // Query has 2 tokens, doc has 3 tokens, embedding dim = 3
        let query_embs = vec![
            vec![1.0, 0.0, 0.0], // q_token_0: unit x
            vec![0.0, 1.0, 0.0], // q_token_1: unit y
        ];
        let doc_embs = vec![
            vec![1.0, 0.0, 0.0], // d_token_0: unit x (identical to q0)
            vec![0.0, 0.0, 1.0], // d_token_1: unit z (orthogonal to both)
            vec![0.0, 0.5, 0.5], // d_token_2: mix of y and z
        ];

        let score = ColBERTScorer::maxsim(&query_embs, &doc_embs);

        // q_token_0: max sim to d_token_0 = 1.0 (identical)
        // q_token_1: max sim to d_token_2 = cos(y, [0, 0.5, 0.5])
        //   = (0*0 + 1*0.5 + 0*0.5) / (1.0 * sqrt(0.5)) = 0.5 / 0.7071 ~ 0.7071
        // Total ~ 1.0 + 0.7071 ~ 1.7071
        // cos([0,1,0], [0,0.5,0.5]) = 0.5 / (1.0 * sqrt(0.5)) = 0.5 / 0.7071 ~ 0.7071
        let expected_q1_max = 0.5 / (0.5f32).sqrt();
        let expected = 1.0 + expected_q1_max;
        assert!(
            (score - expected).abs() < 1e-4,
            "MaxSim expected ~{expected}, got {score}"
        );
    }

    #[test]
    fn test_maxsim_empty_doc_embeddings() {
        let query_embs = vec![vec![1.0, 0.0, 0.0]];
        let doc_embs: Vec<Vec<f32>> = vec![];

        let score = ColBERTScorer::maxsim(&query_embs, &doc_embs);
        assert_eq!(score, 0.0, "Empty doc embeddings should produce score 0.0");
    }

    #[test]
    fn test_maxsim_perfect_match() {
        // When query and doc embeddings are identical, MaxSim should equal num_tokens
        let embs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let score = ColBERTScorer::maxsim(&embs, &embs);
        assert!(
            (score - 2.0).abs() < 1e-6,
            "Perfect match with 2 tokens should produce score 2.0, got {score}"
        );
    }

    #[test]
    fn test_sigmoid_calibration_range() {
        // Verify that sigmoid output is in [0, 1] for various raw MaxSim values
        for raw in [-10.0f32, -1.0, 0.0, 1.0, 5.0, 10.0] {
            let calibrated = 1.0 / (1.0 + (-raw).exp());
            assert!(
                (0.0..=1.0).contains(&calibrated),
                "Sigmoid({raw}) = {calibrated} should be in [0, 1]"
            );
        }
    }

    // --- Additional cosine similarity and MaxSim tests ---

    #[test]
    fn test_cosine_similarity_both_zero_vectors() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Both zero vectors should produce 0.0");
    }

    #[test]
    fn test_cosine_similarity_unit_vectors() {
        // Unit vectors along different axes
        let x = vec![1.0, 0.0, 0.0];
        let y = vec![0.0, 1.0, 0.0];
        let z = vec![0.0, 0.0, 1.0];
        assert!(cosine_similarity(&x, &x).abs() - 1.0 < 1e-6);
        assert!(cosine_similarity(&x, &y).abs() < 1e-6);
        assert!(cosine_similarity(&y, &z).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_negative_vectors() {
        let a = vec![-1.0, -2.0, -3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical negative vectors should have similarity 1.0"
        );
    }

    #[test]
    fn test_cosine_similarity_single_dimension() {
        let a = vec![5.0];
        let b = vec![3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Parallel 1D vectors should have similarity 1.0"
        );

        let c = vec![-3.0];
        let sim2 = cosine_similarity(&a, &c);
        assert!(
            (sim2 - (-1.0)).abs() < 1e-6,
            "Anti-parallel 1D vectors should have similarity -1.0"
        );
    }

    #[test]
    fn test_cosine_similarity_high_dimensional() {
        let dim = 768;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim >= -1.0 && sim <= 1.0,
            "Cosine sim should be in [-1, 1], got {sim}"
        );
    }

    #[test]
    fn test_maxsim_single_query_token() {
        let query_embs = vec![vec![1.0, 0.0]];
        let doc_embs = vec![vec![0.5, 0.5], vec![1.0, 0.0]];
        let score = ColBERTScorer::maxsim(&query_embs, &doc_embs);
        // Max sim for the single query token should be cos([1,0], [1,0]) = 1.0
        assert!(
            (score - 1.0).abs() < 1e-6,
            "Single query token maxsim should find exact match"
        );
    }

    #[test]
    fn test_maxsim_single_doc_token() {
        let query_embs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc_embs = vec![vec![0.707, 0.707]]; // ~45 degree vector
        let score = ColBERTScorer::maxsim(&query_embs, &doc_embs);
        // Each query token's max sim is the only doc token
        let expected_per_token = cosine_similarity(&[1.0, 0.0], &[0.707, 0.707]);
        // Both query tokens have same cos to the single doc token
        assert!((score - 2.0 * expected_per_token).abs() < 1e-3);
    }

    #[test]
    fn test_maxsim_all_zero_query() {
        let query_embs = vec![vec![0.0, 0.0, 0.0]];
        let doc_embs = vec![vec![1.0, 0.0, 0.0]];
        let score = ColBERTScorer::maxsim(&query_embs, &doc_embs);
        // cosine_similarity returns 0.0 for zero vectors, which is finite, so it accumulates
        assert_eq!(
            score, 0.0,
            "All-zero query embeddings should produce score 0.0"
        );
    }

    #[test]
    fn test_cosine_similarity_symmetry() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        let sim_ab = cosine_similarity(&a, &b);
        let sim_ba = cosine_similarity(&b, &a);
        assert!(
            (sim_ab - sim_ba).abs() < 1e-6,
            "Cosine similarity should be symmetric"
        );
    }
}
