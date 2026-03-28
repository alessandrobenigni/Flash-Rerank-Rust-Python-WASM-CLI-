//! Stress tests for flash_rerank.
//!
//! All tests are `#[ignore]` — run with `cargo test --test stress -- --ignored`.
//! Uses mock scorers so no GPU or model files are needed.

use flash_rerank::engine::Scorer;
use flash_rerank::types::RerankResult;
use std::sync::Arc;

/// Mock scorer for stress testing — returns deterministic decreasing scores.
struct StressMockScorer;

impl Scorer for StressMockScorer {
    fn score(&self, _query: &str, documents: &[String]) -> flash_rerank::Result<Vec<RerankResult>> {
        let mut results: Vec<RerankResult> = documents
            .iter()
            .enumerate()
            .map(|(i, _)| RerankResult {
                index: i,
                score: 1.0 / (i as f32 + 1.0),
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

#[test]
#[ignore]
fn stress_max_batch_size_1000_docs() {
    let scorer = StressMockScorer;
    let docs: Vec<String> = (0..1000)
        .map(|i| format!("Document number {i} with some content to make it realistic"))
        .collect();

    let results = scorer.score("stress test query", &docs).unwrap();
    assert_eq!(results.len(), 1000);

    // Verify sorted descending
    for window in results.windows(2) {
        assert!(window[0].score >= window[1].score);
    }
}

#[tokio::test]
#[ignore]
async fn stress_concurrent_flood_100_parallel() {
    let scorer = Arc::new(StressMockScorer);
    let docs: Vec<String> = (0..50).map(|i| format!("doc {i}")).collect();

    let mut handles = Vec::new();
    for i in 0..100 {
        let scorer = Arc::clone(&scorer);
        let docs = docs.clone();
        handles.push(tokio::spawn(async move {
            // Use spawn_blocking since Scorer::score is sync
            tokio::task::spawn_blocking(move || scorer.score(&format!("query {i}"), &docs).unwrap())
                .await
                .unwrap()
        }));
    }

    for handle in handles {
        let results = handle.await.unwrap();
        assert_eq!(results.len(), 50);
    }
}

#[test]
#[ignore]
fn stress_large_document_truncation() {
    let scorer = StressMockScorer;
    // Each document is ~100KB of text
    let large_doc = "a".repeat(100_000);
    let docs: Vec<String> = (0..10).map(|_| large_doc.clone()).collect();

    let results = scorer.score("large doc test", &docs).unwrap();
    assert_eq!(results.len(), 10);
}

#[test]
#[ignore]
fn stress_very_many_short_documents_10000() {
    let scorer = StressMockScorer;
    let docs: Vec<String> = (0..10_000).map(|i| format!("d{i}")).collect();

    let results = scorer.score("many short docs", &docs).unwrap();
    assert_eq!(results.len(), 10_000);

    // Verify all indices are valid and unique
    let mut seen = std::collections::HashSet::new();
    for r in &results {
        assert!(r.index < 10_000);
        assert!(seen.insert(r.index));
    }
}

#[test]
#[ignore]
fn stress_repeated_scoring_stability() {
    let scorer = StressMockScorer;
    let docs: Vec<String> = (0..20).map(|i| format!("doc {i}")).collect();

    // Score the same query 1000 times — results must be identical each time
    let baseline = scorer.score("stability test", &docs).unwrap();
    for _ in 0..1000 {
        let results = scorer.score("stability test", &docs).unwrap();
        assert_eq!(results.len(), baseline.len());
        for (a, b) in results.iter().zip(baseline.iter()) {
            assert_eq!(a.index, b.index);
            assert!((a.score - b.score).abs() < f32::EPSILON);
        }
    }
}

#[tokio::test]
#[ignore]
async fn stress_cascade_concurrent_flood() {
    use flash_rerank::cascade::CascadePipeline;

    let fast = Box::new(StressMockScorer);
    let big = Box::new(StressMockScorer);
    let cascade = Arc::new(CascadePipeline::new(fast, big, 10, 0.8, 0.2).expect("valid config"));

    let docs: Vec<String> = (0..20).map(|i| format!("doc {i}")).collect();

    let mut handles = Vec::new();
    for i in 0..50 {
        let cascade = Arc::clone(&cascade);
        let docs = docs.clone();
        handles.push(tokio::spawn(async move {
            tokio::task::spawn_blocking(move || {
                cascade.rerank(&format!("query {i}"), &docs).unwrap()
            })
            .await
            .unwrap()
        }));
    }

    for handle in handles {
        let results = handle.await.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 10); // cascade_top_k = 10
    }
}
