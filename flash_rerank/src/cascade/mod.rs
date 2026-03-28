use crate::engine::Scorer;
use crate::types::RerankResult;

/// Configuration for cascade confidence thresholds.
#[derive(Debug, Clone)]
pub struct CascadeConfig {
    /// Score threshold above which documents are accepted without rescoring.
    pub high_confidence: f32,
    /// Score threshold below which documents are rejected entirely.
    pub low_confidence: f32,
}

/// Two-stage cascade pipeline: fast model filters, big model rescores uncertain zone.
///
/// Three confidence zones:
/// - **High confidence** (score >= high_threshold): passed through without rescoring
/// - **Uncertain** (low_threshold <= score < high_threshold): rescored by the big model
/// - **Low confidence** (score < low_threshold): rejected entirely
///
/// The cascade implements `Scorer` so it composes with other scorers.
pub struct CascadePipeline {
    fast_model: Box<dyn Scorer>,
    big_model: Box<dyn Scorer>,
    cascade_top_k: usize,
    config: CascadeConfig,
}

impl std::fmt::Debug for CascadePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CascadePipeline")
            .field("cascade_top_k", &self.cascade_top_k)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl CascadePipeline {
    /// Create a new cascade pipeline.
    ///
    /// # Arguments
    /// * `fast_model` - Lightweight scorer for initial filtering (e.g., MiniLM)
    /// * `big_model` - Accurate scorer for rescoring uncertain documents (e.g., large cross-encoder)
    /// * `cascade_top_k` - Maximum number of results to return
    /// * `high_confidence` - Threshold above which documents pass without rescoring
    /// * `low_confidence` - Threshold below which documents are rejected
    ///
    /// # Errors
    /// Returns `Config` error if `high_confidence <= low_confidence`.
    pub fn new(
        fast_model: Box<dyn Scorer>,
        big_model: Box<dyn Scorer>,
        cascade_top_k: usize,
        high_confidence: f32,
        low_confidence: f32,
    ) -> crate::Result<Self> {
        if high_confidence <= low_confidence {
            return Err(crate::Error::Config(format!(
                "high_confidence ({high_confidence}) must be greater than low_confidence ({low_confidence})"
            )));
        }
        Ok(Self {
            fast_model,
            big_model,
            cascade_top_k,
            config: CascadeConfig {
                high_confidence,
                low_confidence,
            },
        })
    }

    /// Run the cascade: score all docs with fast model, rescore uncertain zone with big model.
    ///
    /// Stage 1: Fast model scores ALL documents, partitions into three zones.
    /// Stage 2: Big model rescores only the uncertain zone.
    /// Merge: High-confidence + rescored results, sorted descending, truncated to top_k.
    ///
    /// Graceful degradation: if big model fails, returns high-confidence results only.
    pub fn rerank(&self, query: &str, documents: &[String]) -> crate::Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Stage 1: Fast model scores ALL documents
        let fast_results = self.fast_model.score(query, documents)?;

        let mut high_confidence_results = Vec::new();
        let mut uncertain_indices = Vec::new();

        for result in &fast_results {
            if result.score >= self.config.high_confidence {
                high_confidence_results.push(result.clone());
            } else if result.score >= self.config.low_confidence {
                uncertain_indices.push(result.index);
            }
            // Below low_confidence: rejected
        }

        // If no uncertain documents, skip big model entirely
        if uncertain_indices.is_empty() {
            tracing::info!(
                "All candidates above high-confidence threshold; skipping accurate model"
            );
            high_confidence_results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            high_confidence_results.truncate(self.cascade_top_k);
            return Ok(high_confidence_results);
        }

        // Stage 2: Big model rescores uncertain zone
        let uncertain_docs: Vec<String> = uncertain_indices
            .iter()
            .map(|&i| documents[i].clone())
            .collect();

        let big_results = match self.big_model.score(query, &uncertain_docs) {
            Ok(results) => results
                .into_iter()
                .map(|mut r| {
                    // Remap indices back to original document positions
                    r.index = uncertain_indices[r.index];
                    r
                })
                .collect::<Vec<_>>(),
            Err(e) => {
                tracing::warn!(
                    "Accurate model failed: {e}. Returning high-confidence results only."
                );
                high_confidence_results.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                high_confidence_results.truncate(self.cascade_top_k);
                return Ok(high_confidence_results);
            }
        };

        // Merge high-confidence + rescored, sort descending, truncate
        let mut all_results = high_confidence_results;
        all_results.extend(big_results);
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(self.cascade_top_k);

        Ok(all_results)
    }
}

/// CascadePipeline implements Scorer so it composes with other scorers.
impl Scorer for CascadePipeline {
    fn score(&self, query: &str, documents: &[String]) -> crate::Result<Vec<RerankResult>> {
        self.rerank(query, documents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock scorer that returns pre-determined scores for testing.
    struct MockScorer {
        scores: Vec<f32>,
    }

    impl Scorer for MockScorer {
        fn score(&self, _query: &str, documents: &[String]) -> crate::Result<Vec<RerankResult>> {
            let mut results: Vec<RerankResult> = documents
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let score = if i < self.scores.len() {
                        self.scores[i]
                    } else {
                        0.0
                    };
                    RerankResult {
                        index: i,
                        score,
                        document: None,
                    }
                })
                .collect();
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Ok(results)
        }
    }

    /// Mock scorer that always fails -- used to test graceful degradation.
    struct FailingScorer;

    impl Scorer for FailingScorer {
        fn score(&self, _query: &str, _documents: &[String]) -> crate::Result<Vec<RerankResult>> {
            Err(crate::Error::Inference("Big model unavailable".to_string()))
        }
    }

    #[test]
    fn test_cascade_config_validation() {
        let fast = Box::new(MockScorer { scores: vec![] });
        let big = Box::new(MockScorer { scores: vec![] });

        // high must be > low
        let result = CascadePipeline::new(fast, big, 10, 0.3, 0.7);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), crate::Error::Config(_)));
    }

    #[test]
    fn test_cascade_three_zone_routing() {
        // Fast model returns: 0.9 (high), 0.5 (uncertain), 0.1 (low)
        let fast = Box::new(MockScorer {
            scores: vec![0.9, 0.5, 0.1],
        });
        // Big model will rescore the uncertain zone (1 doc)
        let big = Box::new(MockScorer { scores: vec![0.75] });

        let cascade = CascadePipeline::new(fast, big, 10, 0.85, 0.15).expect("valid config");

        let docs = vec![
            "high confidence doc".to_string(),
            "uncertain doc".to_string(),
            "low confidence doc".to_string(),
        ];
        let results = cascade
            .rerank("test query", &docs)
            .expect("rerank succeeds");

        // Should have 2 results: high-confidence (index 0) + rescored uncertain (index 1)
        // Low-confidence (index 2, score 0.1) is below low_confidence=0.15, so rejected
        assert_eq!(results.len(), 2);

        // High-confidence doc should be present
        assert!(results.iter().any(|r| r.index == 0));
        // Uncertain doc should be rescored and present
        assert!(results.iter().any(|r| r.index == 1));
        // Low-confidence doc should be absent
        assert!(!results.iter().any(|r| r.index == 2));

        // Results should be sorted descending by score
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_cascade_all_high_confidence() {
        // All scores above high threshold -- big model should not be called
        let fast = Box::new(MockScorer {
            scores: vec![0.95, 0.90, 0.88],
        });
        let big = Box::new(FailingScorer); // would fail if called

        let cascade = CascadePipeline::new(fast, big, 10, 0.85, 0.15).expect("valid config");

        let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let results = cascade
            .rerank("query", &docs)
            .expect("should skip big model");

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_cascade_graceful_degradation() {
        // Fast model returns mixed scores
        let fast = Box::new(MockScorer {
            scores: vec![0.9, 0.5, 0.1],
        });
        // Big model fails
        let big = Box::new(FailingScorer);

        let cascade = CascadePipeline::new(fast, big, 10, 0.85, 0.15).expect("valid config");

        let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let results = cascade
            .rerank("query", &docs)
            .expect("graceful degradation");

        // Should return only high-confidence results
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0);
        assert!(results[0].score >= 0.85);
    }

    #[test]
    fn test_cascade_empty_documents() {
        let fast = Box::new(MockScorer { scores: vec![] });
        let big = Box::new(MockScorer { scores: vec![] });

        let cascade = CascadePipeline::new(fast, big, 10, 0.85, 0.15).expect("valid config");

        let results = cascade
            .rerank("query", &[])
            .expect("empty docs should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_cascade_implements_scorer_trait() {
        let fast = Box::new(MockScorer {
            scores: vec![0.9, 0.5],
        });
        let big = Box::new(MockScorer { scores: vec![0.7] });

        let cascade = CascadePipeline::new(fast, big, 10, 0.85, 0.15).expect("valid config");

        // Use through Scorer trait
        let scorer: &dyn Scorer = &cascade;
        let docs = vec!["a".to_string(), "b".to_string()];
        let results = scorer.score("query", &docs).expect("scorer trait works");
        assert!(!results.is_empty());
    }

    // --- Additional cascade tests ---

    #[test]
    fn test_cascade_all_low_confidence() {
        // All scores below low_confidence: everything rejected
        let fast = Box::new(MockScorer {
            scores: vec![0.05, 0.10, 0.02],
        });
        let big = Box::new(FailingScorer); // should never be called since nothing in uncertain zone

        let cascade = CascadePipeline::new(fast, big, 10, 0.85, 0.15).expect("valid config");

        let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let results = cascade
            .rerank("query", &docs)
            .expect("should handle all-low");
        assert!(
            results.is_empty(),
            "All below low_confidence should yield empty results"
        );
    }

    #[test]
    fn test_cascade_all_uncertain() {
        // All scores in the uncertain zone: big model rescores all
        let fast = Box::new(MockScorer {
            scores: vec![0.4, 0.5, 0.6],
        });
        let big = Box::new(MockScorer {
            scores: vec![0.9, 0.8, 0.7],
        });

        let cascade = CascadePipeline::new(fast, big, 10, 0.85, 0.15).expect("valid config");

        let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let results = cascade
            .rerank("query", &docs)
            .expect("all uncertain rescoring");
        assert_eq!(results.len(), 3, "All uncertain docs should be rescored");
    }

    #[test]
    fn test_cascade_equal_thresholds_error() {
        let fast = Box::new(MockScorer { scores: vec![] });
        let big = Box::new(MockScorer { scores: vec![] });

        let result = CascadePipeline::new(fast, big, 10, 0.5, 0.5);
        assert!(result.is_err(), "Equal thresholds should error");
    }

    #[test]
    fn test_cascade_top_k_truncation() {
        // More results than top_k
        let fast = Box::new(MockScorer {
            scores: vec![0.9, 0.92, 0.95, 0.88, 0.91],
        });
        let big = Box::new(FailingScorer); // not called: all high confidence

        let cascade = CascadePipeline::new(fast, big, 2, 0.85, 0.15).expect("valid config");

        let docs: Vec<String> = (0..5).map(|i| format!("doc{i}")).collect();
        let results = cascade.rerank("query", &docs).expect("top_k truncation");
        assert_eq!(
            results.len(),
            2,
            "Results should be truncated to cascade_top_k=2"
        );
    }

    #[test]
    fn test_cascade_index_remapping() {
        // Verify that rescored uncertain docs have correct original indices
        let fast = Box::new(MockScorer {
            scores: vec![0.9, 0.5, 0.1, 0.6],
        });
        // Big model rescores uncertain zone docs (indices 1 and 3 from original)
        let big = Box::new(MockScorer {
            scores: vec![0.75, 0.70],
        });

        let cascade = CascadePipeline::new(fast, big, 10, 0.85, 0.15).expect("valid config");

        let docs = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let results = cascade.rerank("query", &docs).expect("index remapping");

        // Results should contain doc 0 (high), doc 1 and 3 (uncertain, rescored), not doc 2 (low)
        let indices: Vec<usize> = results.iter().map(|r| r.index).collect();
        assert!(
            indices.contains(&0),
            "High-confidence doc 0 should be present"
        );
        assert!(
            indices.contains(&1),
            "Rescored uncertain doc 1 should be present"
        );
        assert!(
            indices.contains(&3),
            "Rescored uncertain doc 3 should be present"
        );
        assert!(
            !indices.contains(&2),
            "Low-confidence doc 2 should be rejected"
        );
    }

    #[test]
    fn test_cascade_big_model_preserves_high_confidence() {
        // When big model rescores, high-confidence results should retain their fast model scores
        let fast = Box::new(MockScorer {
            scores: vec![0.95, 0.5],
        });
        let big = Box::new(MockScorer {
            scores: vec![0.60], // rescore uncertain doc
        });

        let cascade = CascadePipeline::new(fast, big, 10, 0.85, 0.15).expect("valid config");

        let docs = vec!["high".to_string(), "uncertain".to_string()];
        let results = cascade
            .rerank("query", &docs)
            .expect("big model preserves high");

        let high_doc = results.iter().find(|r| r.index == 0).unwrap();
        assert!(
            (high_doc.score - 0.95).abs() < 1e-6,
            "High-confidence doc score should be preserved from fast model"
        );
    }

    #[test]
    fn test_cascade_scorer_trait_object() {
        // Verify CascadePipeline can be stored as Box<dyn Scorer>
        let fast = Box::new(MockScorer { scores: vec![0.9] });
        let big = Box::new(MockScorer { scores: vec![] });
        let cascade = CascadePipeline::new(fast, big, 10, 0.85, 0.15).unwrap();
        let _boxed: Box<dyn Scorer> = Box::new(cascade);
    }
}
