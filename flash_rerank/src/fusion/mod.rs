use serde::{Deserialize, Serialize};

/// Configuration for Reciprocal Rank Fusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// RRF constant (default: 60).
    pub k: usize,
    /// Weight for each ranker's contribution.
    pub weights: Vec<f32>,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            k: 60,
            weights: vec![1.0],
        }
    }
}

/// Reciprocal Rank Fusion: combines multiple ranked lists into a single ranking.
///
/// For each document `d` across all ranked lists:
///   `RRF(d) = sum_r weight_r / (k + rank_r(d) + 1)`
///
/// where `rank_r(d)` is the 0-based rank of `d` in ranked list `r`.
///
/// # Arguments
/// * `ranked_lists` - Slice of ranked lists, each containing `(doc_id, score)` pairs
///   ordered by descending score.
/// * `config` - Fusion configuration with RRF constant `k` and per-list weights.
///
/// # Returns
/// A fused ranked list of `(doc_id, fused_score)` sorted by descending score.
pub fn rrf_fusion(ranked_lists: &[Vec<(usize, f32)>], config: &FusionConfig) -> Vec<(usize, f32)> {
    use std::collections::HashMap;

    if ranked_lists.is_empty() {
        return vec![];
    }
    if ranked_lists.len() == 1 {
        return ranked_lists[0].clone();
    }

    let mut scores: HashMap<usize, f32> = HashMap::new();

    for (list_idx, ranked_list) in ranked_lists.iter().enumerate() {
        let weight = config.weights.get(list_idx).copied().unwrap_or(1.0);
        for (rank, &(doc_idx, _score)) in ranked_list.iter().enumerate() {
            *scores.entry(doc_idx).or_insert(0.0) += weight / (config.k as f32 + rank as f32 + 1.0);
        }
    }

    let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_two_lists() {
        // Two overlapping ranked lists
        let list1 = vec![(0, 0.9), (1, 0.8), (2, 0.7)];
        let list2 = vec![(1, 0.95), (2, 0.85), (3, 0.75)];

        let config = FusionConfig {
            k: 60,
            weights: vec![1.0, 1.0],
        };

        let fused = rrf_fusion(&[list1, list2], &config);

        // doc 1 appears in both lists (rank 1 in list1, rank 0 in list2)
        // doc 1 RRF = 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61
        // doc 2 appears in both lists (rank 2 in list1, rank 1 in list2)
        // doc 2 RRF = 1/(60+2+1) + 1/(60+1+1) = 1/63 + 1/62
        // doc 0 only in list1 rank 0: 1/(60+0+1) = 1/61
        // doc 3 only in list2 rank 2: 1/(60+2+1) = 1/63

        assert!(!fused.is_empty());
        assert_eq!(fused.len(), 4); // 4 unique docs

        // doc 1 should be ranked first (highest combined score)
        assert_eq!(fused[0].0, 1);
        // doc 1 score ~= 1/62 + 1/61
        let expected_doc1 = 1.0 / 62.0 + 1.0 / 61.0;
        assert!((fused[0].1 - expected_doc1 as f32).abs() < 1e-6);

        // All scores should be in descending order
        for w in fused.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn test_rrf_three_lists() {
        let list1 = vec![(0, 0.9), (1, 0.8)];
        let list2 = vec![(1, 0.95), (2, 0.85)];
        let list3 = vec![(1, 0.99), (0, 0.5)];

        let config = FusionConfig {
            k: 60,
            weights: vec![1.0, 1.0, 1.0],
        };

        let fused = rrf_fusion(&[list1, list2, list3], &config);

        // doc 1 appears in all three lists, should be ranked first
        assert_eq!(fused[0].0, 1);
        assert_eq!(fused.len(), 3); // 3 unique docs
    }

    #[test]
    fn test_rrf_single_list_bypass() {
        let list = vec![(0, 0.9), (1, 0.8), (2, 0.7)];
        let config = FusionConfig::default();

        let fused = rrf_fusion(&[list.clone()], &config);

        // Single list should be returned as-is
        assert_eq!(fused, list);
    }

    #[test]
    fn test_rrf_empty_input() {
        let config = FusionConfig::default();
        let fused = rrf_fusion(&[], &config);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_weighted() {
        // Weight neural scores higher than BM25
        let bm25 = vec![(0, 10.0), (1, 8.0)];
        let neural = vec![(1, 0.95), (0, 0.3)];

        let config = FusionConfig {
            k: 60,
            weights: vec![0.3, 0.7], // 30% BM25, 70% neural
        };

        let fused = rrf_fusion(&[bm25, neural], &config);

        // doc 0: 0.3/(61) + 0.7/(62)  vs  doc 1: 0.3/(62) + 0.7/(61)
        // doc 1 should rank higher because neural weight is larger and it's rank 0 there
        assert_eq!(fused[0].0, 1);
    }

    // --- Additional fusion tests ---

    #[test]
    fn test_rrf_disjoint_lists() {
        let list1 = vec![(0, 0.9), (1, 0.8)];
        let list2 = vec![(2, 0.95), (3, 0.85)];
        let config = FusionConfig {
            k: 60,
            weights: vec![1.0, 1.0],
        };
        let fused = rrf_fusion(&[list1, list2], &config);
        assert_eq!(
            fused.len(),
            4,
            "Disjoint lists should produce union of all docs"
        );
    }

    #[test]
    fn test_rrf_single_doc_per_list() {
        let list1 = vec![(0, 1.0)];
        let list2 = vec![(1, 1.0)];
        let config = FusionConfig {
            k: 60,
            weights: vec![1.0, 1.0],
        };
        let fused = rrf_fusion(&[list1, list2], &config);
        assert_eq!(fused.len(), 2);
        // Both should have the same RRF score: 1/(60+0+1) = 1/61
        assert!(
            (fused[0].1 - fused[1].1).abs() < 1e-6,
            "Equal-ranked single docs should tie"
        );
    }

    #[test]
    fn test_rrf_large_k() {
        let list1 = vec![(0, 0.9), (1, 0.8)];
        let list2 = vec![(1, 0.95), (0, 0.3)];
        let config = FusionConfig {
            k: 10_000,
            weights: vec![1.0, 1.0],
        };
        let fused = rrf_fusion(&[list1, list2], &config);
        assert_eq!(fused.len(), 2);
        // With very large k, rank differences matter less
        let score_diff = (fused[0].1 - fused[1].1).abs();
        assert!(
            score_diff < 1e-6,
            "Large k should minimize rank difference effects"
        );
    }

    #[test]
    fn test_rrf_k_zero() {
        let list1 = vec![(0, 0.9), (1, 0.8)];
        let list2 = vec![(1, 0.95), (0, 0.3)];
        let config = FusionConfig {
            k: 0,
            weights: vec![1.0, 1.0],
        };
        let fused = rrf_fusion(&[list1, list2], &config);
        assert!(!fused.is_empty(), "k=0 should still produce results");
    }

    #[test]
    fn test_rrf_zero_weight() {
        let list1 = vec![(0, 0.9), (1, 0.8)];
        let list2 = vec![(2, 0.95), (3, 0.85)];
        let config = FusionConfig {
            k: 60,
            weights: vec![0.0, 1.0], // list1 has zero weight
        };
        let fused = rrf_fusion(&[list1, list2], &config);
        // list1 docs should have score 0, list2 docs should have positive scores
        let doc0_score = fused.iter().find(|d| d.0 == 0).unwrap().1;
        let doc2_score = fused.iter().find(|d| d.0 == 2).unwrap().1;
        assert_eq!(doc0_score, 0.0, "Zero-weight list should contribute 0");
        assert!(
            doc2_score > 0.0,
            "Non-zero weight list should contribute positive"
        );
    }

    #[test]
    fn test_rrf_missing_weight_defaults_to_one() {
        // Provide fewer weights than lists -- missing ones default to 1.0
        let list1 = vec![(0, 0.9)];
        let list2 = vec![(1, 0.95)];
        let config = FusionConfig {
            k: 60,
            weights: vec![1.0], // only one weight for two lists
        };
        let fused = rrf_fusion(&[list1, list2], &config);
        assert_eq!(fused.len(), 2);
        // doc 1 should have score 1/(61) with default weight 1.0
        let doc1_score = fused.iter().find(|d| d.0 == 1).unwrap().1;
        let expected = 1.0 / 61.0_f32;
        assert!(
            (doc1_score - expected).abs() < 1e-6,
            "Missing weight should default to 1.0"
        );
    }

    #[test]
    fn test_rrf_duplicate_doc_ids_in_same_list() {
        // If the same doc_id appears multiple times in one list, each occurrence contributes
        let list1 = vec![(0, 0.9), (0, 0.8)];
        let config = FusionConfig {
            k: 60,
            weights: vec![1.0, 1.0],
        };
        let list2 = vec![(1, 0.5)];
        let fused = rrf_fusion(&[list1, list2], &config);
        // doc 0 should accumulate score from both ranks in list1
        let doc0 = fused.iter().find(|d| d.0 == 0).unwrap();
        let expected = 1.0 / 61.0 + 1.0 / 62.0; // rank 0 and rank 1
        assert!((doc0.1 - expected).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_output_sorted_descending() {
        let list1 = vec![(0, 0.9), (1, 0.8), (2, 0.7)];
        let list2 = vec![(3, 0.95), (4, 0.85), (5, 0.75)];
        let config = FusionConfig {
            k: 60,
            weights: vec![1.0, 1.0],
        };
        let fused = rrf_fusion(&[list1, list2], &config);
        for w in fused.windows(2) {
            assert!(w[0].1 >= w[1].1, "Output should be sorted descending");
        }
    }
}
