use std::collections::HashSet;

/// Compute NDCG@k (Normalized Discounted Cumulative Gain).
///
/// - `ranked`: ordered list of document indices (most relevant first)
/// - `relevance`: relevance label for each document index (0 = irrelevant)
/// - `k`: cutoff position
///
/// Uses the formula: gain = 2^rel - 1, discount = 1/log2(i+2)
pub fn ndcg_at_k(ranked: &[usize], relevance: &[u8], k: usize) -> f64 {
    let k = k.min(ranked.len());

    // DCG@k
    let dcg: f64 = (0..k)
        .map(|i| {
            let doc_idx = ranked[i];
            let rel = if doc_idx < relevance.len() {
                relevance[doc_idx] as f64
            } else {
                0.0
            };
            (2f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2()
        })
        .sum();

    // Ideal DCG@k: sort relevances descending
    let mut ideal_rels: Vec<f64> = relevance.iter().map(|&r| r as f64).collect();
    ideal_rels.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let idcg: f64 = ideal_rels
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| (2f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2())
        .sum();

    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

/// Compute MRR (Mean Reciprocal Rank).
///
/// - `ranked`: ordered list of document indices (most relevant first)
/// - `relevant`: set of document indices that are relevant
///
/// Returns 1/(rank) for the first relevant document found, or 0.0 if none.
pub fn mrr(ranked: &[usize], relevant: &[usize]) -> f64 {
    let relevant_set: HashSet<usize> = relevant.iter().copied().collect();
    for (i, &doc_idx) in ranked.iter().enumerate() {
        if relevant_set.contains(&doc_idx) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndcg_perfect_ranking() {
        // Documents 0,1,2 with relevances 3,2,1 -- perfect ranking
        let ranked = vec![0, 1, 2];
        let relevance = vec![3, 2, 1];
        let score = ndcg_at_k(&ranked, &relevance, 3);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "Perfect ranking should give NDCG=1.0, got {score}"
        );
    }

    #[test]
    fn ndcg_worst_ranking() {
        // Documents ranked worst-first: [2, 1, 0] with relevances [3, 2, 1]
        let ranked = vec![2, 1, 0];
        let relevance = vec![3, 2, 1];
        let score = ndcg_at_k(&ranked, &relevance, 3);
        assert!(score < 1.0, "Worst ranking should give NDCG < 1.0");
        assert!(score > 0.0, "Non-zero relevances should give NDCG > 0.0");
    }

    #[test]
    fn ndcg_all_irrelevant() {
        let ranked = vec![0, 1, 2];
        let relevance = vec![0, 0, 0];
        let score = ndcg_at_k(&ranked, &relevance, 3);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn ndcg_k_truncates() {
        let ranked = vec![2, 0, 1]; // irrelevant first
        let relevance = vec![3, 2, 0];
        let at_1 = ndcg_at_k(&ranked, &relevance, 1);
        let at_3 = ndcg_at_k(&ranked, &relevance, 3);
        // At k=1, we only see doc 2 (rel=0), so DCG=0 -> NDCG=0
        assert_eq!(at_1, 0.0);
        // At k=3, we see all docs
        assert!(at_3 > 0.0);
    }

    #[test]
    fn mrr_first_is_relevant() {
        let ranked = vec![0, 1, 2];
        let relevant = vec![0];
        assert_eq!(mrr(&ranked, &relevant), 1.0);
    }

    #[test]
    fn mrr_second_is_relevant() {
        let ranked = vec![0, 1, 2];
        let relevant = vec![1];
        assert_eq!(mrr(&ranked, &relevant), 0.5);
    }

    #[test]
    fn mrr_none_relevant() {
        let ranked = vec![0, 1, 2];
        let relevant = vec![5, 6];
        assert_eq!(mrr(&ranked, &relevant), 0.0);
    }

    #[test]
    fn mrr_multiple_relevant_returns_first() {
        let ranked = vec![3, 1, 2, 0];
        let relevant = vec![0, 2]; // doc 2 appears at rank 3, doc 0 at rank 4
        let score = mrr(&ranked, &relevant);
        // First relevant is doc 2 at position 2 (0-indexed), so MRR = 1/3
        assert!((score - 1.0 / 3.0).abs() < 1e-10);
    }
}
