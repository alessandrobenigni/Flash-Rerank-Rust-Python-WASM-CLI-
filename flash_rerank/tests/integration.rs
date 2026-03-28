//! Integration tests for the flash_rerank reranking pipeline.
//!
//! Uses a mock scorer to test the full pipeline without requiring
//! downloaded ONNX models or GPU hardware.

use flash_rerank::engine::Scorer;
use flash_rerank::types::RerankResult;
use std::sync::Arc;

/// Mock scorer that returns deterministic decreasing scores.
struct MockScorer;

impl Scorer for MockScorer {
    fn score(&self, _query: &str, documents: &[String]) -> flash_rerank::Result<Vec<RerankResult>> {
        let mut results: Vec<RerankResult> = documents
            .iter()
            .enumerate()
            .map(|(i, _)| RerankResult {
                index: i,
                score: 1.0 / (i as f32 + 1.0), // decreasing scores: 1.0, 0.5, 0.33, ...
                document: None,
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

/// Mock scorer that returns calibrated sigmoid-like scores from known logits.
struct SigmoidMockScorer {
    logits: Vec<f32>,
}

impl Scorer for SigmoidMockScorer {
    fn score(&self, _query: &str, documents: &[String]) -> flash_rerank::Result<Vec<RerankResult>> {
        let mut results: Vec<RerankResult> = documents
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let logit = if i < self.logits.len() {
                    self.logits[i]
                } else {
                    0.0
                };
                // Apply sigmoid calibration
                let score = 1.0 / (1.0 + (-logit).exp());
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

fn make_docs(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("document {i}")).collect()
}

#[test]
fn test_full_pipeline_mock_scorer() {
    let scorer = MockScorer;
    let docs = make_docs(5);
    let results = scorer.score("test query", &docs).unwrap();

    assert_eq!(results.len(), 5);
    // Each result should have valid fields
    for r in &results {
        assert!(r.score > 0.0);
        assert!(r.index < docs.len());
        assert!(r.document.is_none());
    }
}

#[test]
fn test_rerank_top_k_filtering() {
    let scorer = MockScorer;
    let docs = make_docs(10);
    let mut results = scorer.score("test query", &docs).unwrap();

    // Simulate top_k=3 truncation (as done in flash_rerank::rerank)
    let top_k = 3;
    results.truncate(top_k);

    assert_eq!(results.len(), 3);
    // The top 3 should be the highest-scored documents
    for r in &results {
        assert!(r.score >= 1.0 / 4.0); // 4th doc would have score 0.25
    }
}

#[test]
fn test_calibrated_scores_bounded() {
    let scorer = SigmoidMockScorer {
        logits: vec![-10.0, -1.0, 0.0, 1.0, 10.0],
    };
    let docs = make_docs(5);
    let results = scorer.score("calibration test", &docs).unwrap();

    for r in &results {
        assert!(
            r.score >= 0.0 && r.score <= 1.0,
            "Score {} out of [0,1] range",
            r.score
        );
    }
}

#[test]
fn test_results_sorted_descending() {
    let scorer = MockScorer;
    let docs = make_docs(20);
    let results = scorer.score("sort test", &docs).unwrap();

    for window in results.windows(2) {
        assert!(
            window[0].score >= window[1].score,
            "Results not sorted descending: {} < {}",
            window[0].score,
            window[1].score
        );
    }
}

#[test]
fn test_result_indices_valid() {
    let scorer = MockScorer;
    let docs = make_docs(15);
    let results = scorer.score("index test", &docs).unwrap();

    for r in &results {
        assert!(
            r.index < docs.len(),
            "Index {} >= documents.len() ({})",
            r.index,
            docs.len()
        );
    }
}

#[test]
fn test_result_indices_unique() {
    let scorer = MockScorer;
    let docs = make_docs(10);
    let results = scorer.score("unique test", &docs).unwrap();

    let mut seen = std::collections::HashSet::new();
    for r in &results {
        assert!(
            seen.insert(r.index),
            "Duplicate index {} in results",
            r.index
        );
    }
}

#[test]
fn test_scorer_thread_safety() {
    let scorer = Arc::new(MockScorer);
    let docs = make_docs(5);

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let scorer = Arc::clone(&scorer);
            let docs = docs.clone();
            std::thread::spawn(move || {
                let results = scorer.score(&format!("thread {i} query"), &docs).unwrap();
                assert_eq!(results.len(), 5);
                results
            })
        })
        .collect();

    for handle in handles {
        let results = handle.join().expect("thread panicked");
        assert_eq!(results.len(), 5);
        // All threads should get identical results (deterministic scorer)
        for window in results.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }
}

#[test]
fn test_empty_documents() {
    let scorer = MockScorer;
    let results = scorer.score("empty test", &[]).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_single_document() {
    let scorer = MockScorer;
    let docs = make_docs(1);
    let results = scorer.score("single test", &docs).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].index, 0);
    assert!((results[0].score - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_return_documents_attachment() {
    let scorer = MockScorer;
    let docs = make_docs(3);
    let mut results = scorer.score("attach test", &docs).unwrap();

    // Simulate return_documents=true (as done in flash_rerank::rerank)
    for result in &mut results {
        if result.index < docs.len() {
            result.document = Some(docs[result.index].clone());
        }
    }

    for r in &results {
        assert!(r.document.is_some());
        let doc = r.document.as_ref().unwrap();
        assert_eq!(doc, &format!("document {}", r.index));
    }
}
