//! Property-based tests for flash_rerank pure functions.
//!
//! Uses proptest to verify invariants across randomized inputs.
//! Tests only pure functions -- no model loading or GPU required.

use proptest::prelude::*;

use flash_rerank::calibrate::{Calibrator, PlattCalibrator, SigmoidCalibrator};
use flash_rerank::cascade::CascadePipeline;
use flash_rerank::engine::Scorer;
use flash_rerank::fusion::{FusionConfig, rrf_fusion};
use flash_rerank::types::RerankResult;

// ---------------------------------------------------------------------------
// Helper: standalone cosine similarity for property testing
// (the engine version is private)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Helper: standalone NDCG computation for property testing
// ---------------------------------------------------------------------------
fn dcg(scores: &[f32]) -> f64 {
    scores
        .iter()
        .enumerate()
        .map(|(i, &s)| s as f64 / (i as f64 + 2.0).log2())
        .sum()
}

fn ndcg(predicted: &[f32], ideal: &[f32]) -> f64 {
    let dcg_val = dcg(predicted);
    let idcg_val = dcg(ideal);
    if idcg_val == 0.0 {
        0.0
    } else {
        (dcg_val / idcg_val).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Helper: standalone MRR computation for property testing
// ---------------------------------------------------------------------------
fn mrr(rankings: &[Option<usize>]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;
    for rank in rankings {
        count += 1;
        if let Some(r) = rank {
            sum += 1.0 / (*r as f64 + 1.0);
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

// ---------------------------------------------------------------------------
// Mock scorer for cascade property tests
// ---------------------------------------------------------------------------
struct PropMockScorer {
    base_score: f32,
}

impl Scorer for PropMockScorer {
    fn score(&self, _query: &str, documents: &[String]) -> flash_rerank::Result<Vec<RerankResult>> {
        let mut results: Vec<RerankResult> = documents
            .iter()
            .enumerate()
            .map(|(i, _)| RerankResult {
                index: i,
                score: self.base_score - (i as f32 * 0.01),
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

// ===== Sigmoid property tests =====

proptest! {
    /// Sigmoid output is always in [0, 1] for any finite f32 input.
    #[test]
    fn prop_sigmoid_output_range(x in proptest::num::f32::NORMAL) {
        let cal = SigmoidCalibrator;
        let y = cal.calibrate(x);
        prop_assert!(y >= 0.0 && y <= 1.0, "sigmoid({x}) = {y} out of [0,1]");
    }

    /// Platt scaling output is always in [0, 1] for any finite f32 input.
    #[test]
    fn prop_platt_output_range(
        x in proptest::num::f32::NORMAL,
        a in -5.0f64..5.0,
        b in -5.0f64..5.0,
    ) {
        let cal = PlattCalibrator::new(a, b);
        let y = cal.calibrate(x);
        prop_assert!(y >= 0.0 && y <= 1.0, "platt({x}, a={a}, b={b}) = {y} out of [0,1]");
    }

    /// Sigmoid is monotonically non-decreasing.
    #[test]
    fn prop_sigmoid_monotonic(
        x in proptest::num::f32::NORMAL,
        delta in 0.0f32..100.0,
    ) {
        let cal = SigmoidCalibrator;
        let y1 = cal.calibrate(x);
        let y2 = cal.calibrate(x + delta);
        prop_assert!(y2 >= y1 - f32::EPSILON, "sigmoid({}) = {} > sigmoid({}) = {}", x, y1, x + delta, y2);
    }
}

// ===== RRF property tests =====

proptest! {
    /// RRF result count is at most the number of unique document IDs across all lists.
    #[test]
    fn prop_rrf_result_count_le_unique_docs(
        n_lists in 1usize..5,
        n_docs in 1usize..20,
    ) {
        let ranked_lists: Vec<Vec<(usize, f32)>> = (0..n_lists)
            .map(|l| {
                (0..n_docs)
                    .map(|d| (d + l * 3, 1.0 / (d as f32 + 1.0)))
                    .collect()
            })
            .collect();

        let config = FusionConfig {
            k: 60,
            weights: vec![1.0; n_lists],
        };

        let fused = rrf_fusion(&ranked_lists, &config);

        // Count unique doc IDs
        let unique: std::collections::HashSet<usize> = ranked_lists
            .iter()
            .flat_map(|l| l.iter().map(|(id, _)| *id))
            .collect();

        prop_assert!(
            fused.len() <= unique.len(),
            "fused.len()={} > unique docs={}",
            fused.len(),
            unique.len()
        );
    }

    /// RRF output is sorted in descending order by score.
    #[test]
    fn prop_rrf_output_sorted_descending(
        n_docs in 2usize..15,
    ) {
        let list1: Vec<(usize, f32)> = (0..n_docs)
            .map(|d| (d, 1.0 / (d as f32 + 1.0)))
            .collect();
        let list2: Vec<(usize, f32)> = (0..n_docs)
            .rev()
            .enumerate()
            .map(|(rank, d)| (d, 1.0 / (rank as f32 + 1.0)))
            .collect();

        let config = FusionConfig {
            k: 60,
            weights: vec![1.0, 1.0],
        };

        let fused = rrf_fusion(&[list1, list2], &config);

        for window in fused.windows(2) {
            prop_assert!(
                window[0].1 >= window[1].1,
                "Not sorted: {} < {}",
                window[0].1,
                window[1].1
            );
        }
    }
}

// ===== Cosine similarity property tests =====

proptest! {
    /// Cosine similarity is in [-1, 1] for any non-zero vectors.
    #[test]
    fn prop_cosine_similarity_range(
        a in proptest::collection::vec(proptest::num::f32::NORMAL, 1..32),
        b in proptest::collection::vec(proptest::num::f32::NORMAL, 1..32),
    ) {
        let min_len = a.len().min(b.len());
        let sim = cosine_similarity(&a[..min_len], &b[..min_len]);
        if sim.is_finite() {
            // Allow small floating-point overshoot due to f32 precision limits
            prop_assert!(
                sim >= -1.0 - 1e-2 && sim <= 1.0 + 1e-2,
                "cosine_similarity = {sim} out of [-1,1] (with tolerance)"
            );
        }
    }

    /// Cosine similarity is symmetric: cos(a, b) == cos(b, a).
    #[test]
    fn prop_cosine_similarity_symmetric(
        a in proptest::collection::vec(-10.0f32..10.0, 4..16),
        b in proptest::collection::vec(-10.0f32..10.0, 4..16),
    ) {
        let min_len = a.len().min(b.len());
        let sim_ab = cosine_similarity(&a[..min_len], &b[..min_len]);
        let sim_ba = cosine_similarity(&b[..min_len], &a[..min_len]);
        prop_assert!(
            (sim_ab - sim_ba).abs() < 1e-5,
            "cos(a,b)={sim_ab} != cos(b,a)={sim_ba}"
        );
    }
}

// ===== NDCG property tests =====

proptest! {
    /// NDCG is in [0, 1] for any non-negative relevance scores.
    #[test]
    fn prop_ndcg_range(
        scores in proptest::collection::vec(0.0f32..10.0, 1..20),
    ) {
        let mut ideal = scores.clone();
        ideal.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let val = ndcg(&scores, &ideal);
        prop_assert!(
            val >= 0.0 && val <= 1.0 + 1e-10,
            "ndcg = {val} out of [0,1]"
        );
    }
}

// ===== MRR property tests =====

proptest! {
    /// MRR is in [0, 1] for any valid rankings.
    #[test]
    fn prop_mrr_range(
        n_queries in 1usize..20,
    ) {
        let rankings: Vec<Option<usize>> = (0..n_queries)
            .map(|i| if i % 3 == 0 { None } else { Some(i) })
            .collect();
        let val = mrr(&rankings);
        prop_assert!(
            val >= 0.0 && val <= 1.0 + 1e-10,
            "mrr = {val} out of [0,1]"
        );
    }
}

// ===== Cascade property tests =====

proptest! {
    /// Cascade result count is at most the input document count.
    #[test]
    fn prop_cascade_result_count_le_input(
        n_docs in 1usize..30,
    ) {
        let fast = Box::new(PropMockScorer { base_score: 0.5 });
        let big = Box::new(PropMockScorer { base_score: 0.7 });
        let cascade = CascadePipeline::new(fast, big, n_docs, 0.8, 0.2)
            .expect("valid config");

        let docs: Vec<String> = (0..n_docs).map(|i| format!("doc {i}")).collect();
        let results = cascade.rerank("query", &docs).expect("rerank ok");

        prop_assert!(
            results.len() <= n_docs,
            "cascade returned {} results for {} docs",
            results.len(),
            n_docs
        );
    }
}
