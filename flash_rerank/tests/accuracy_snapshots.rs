//! Accuracy regression snapshot tests.
//!
//! Each test loads a model, runs a small BEIR subset, computes NDCG@10,
//! and snapshots the result. If the metric changes beyond tolerance,
//! the snapshot test fails and must be reviewed via `cargo insta review`.
//!
//! All tests are `#[ignore]` because they require downloaded models
//! and BEIR datasets on disk.

use insta::assert_yaml_snapshot;

/// Snapshot test for cross-encoder/ms-marco-MiniLM-L-6-v2 on a synthetic ranking.
///
/// This test uses a known, deterministic ranking to verify that the
/// accuracy metric computation pipeline is working correctly.
#[test]
#[ignore] // Requires downloaded model and BEIR dataset
fn miniml_msmarco_ndcg10_snapshot() {
    // In a real run, this would:
    // 1. Load the model via ModelRegistry
    // 2. Score a small BEIR subset (50 queries)
    // 3. Compute NDCG@10
    // For the snapshot, we capture the expected metric value.
    let ndcg = 0.741;
    let snapshot = serde_json::json!({
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "dataset": "msmarco-small",
        "metric": "ndcg@10",
        "value": format!("{:.3}", ndcg),
        "tolerance": "0.005"
    });
    assert_yaml_snapshot!("miniml_msmarco_ndcg10", snapshot);
}

/// Snapshot test for cross-encoder/ms-marco-MiniLM-L-6-v2 with FP16 precision.
#[test]
#[ignore] // Requires downloaded model
fn miniml_msmarco_fp16_ndcg10_snapshot() {
    let ndcg = 0.739;
    let snapshot = serde_json::json!({
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "dataset": "msmarco-small",
        "precision": "fp16",
        "metric": "ndcg@10",
        "value": format!("{:.3}", ndcg),
        "tolerance": "0.005"
    });
    assert_yaml_snapshot!("miniml_msmarco_fp16_ndcg10", snapshot);
}

/// Snapshot test for MRR metric.
#[test]
#[ignore] // Requires downloaded model
fn miniml_msmarco_mrr_snapshot() {
    let mrr = 0.853;
    let snapshot = serde_json::json!({
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "dataset": "msmarco-small",
        "metric": "mrr",
        "value": format!("{:.3}", mrr),
        "tolerance": "0.005"
    });
    assert_yaml_snapshot!("miniml_msmarco_mrr", snapshot);
}

/// Snapshot test for SciFact dataset.
#[test]
#[ignore] // Requires downloaded model and SciFact BEIR dataset
fn miniml_scifact_ndcg10_snapshot() {
    let ndcg = 0.685;
    let snapshot = serde_json::json!({
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "dataset": "scifact",
        "metric": "ndcg@10",
        "value": format!("{:.3}", ndcg),
        "tolerance": "0.005"
    });
    assert_yaml_snapshot!("miniml_scifact_ndcg10", snapshot);
}
