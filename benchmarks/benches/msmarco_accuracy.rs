use criterion::{Criterion, criterion_group, criterion_main};
use std::path::PathBuf;
use std::time::Duration;

/// MSMARCO accuracy benchmark using the MsmarcoEvaluator.
///
/// Requires:
/// - MSMARCO_PATH env var pointing to the MSMARCO dataset directory
/// - Downloaded model in HF cache
///
/// Evaluates NDCG@10 at two scales:
/// - 100 queries (quick feedback loop)
/// - 1000 queries (more statistically robust)

fn default_cache_dir() -> PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return PathBuf::from(cache);
    }
    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        return PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    PathBuf::from(".cache/huggingface/hub")
}

fn get_msmarco_path() -> Option<PathBuf> {
    std::env::var("MSMARCO_PATH").ok().map(PathBuf::from)
}

fn msmarco_ndcg_100(c: &mut Criterion) {
    let msmarco_path = match get_msmarco_path() {
        Some(p) => p,
        None => {
            eprintln!("MSMARCO_PATH not set, skipping msmarco_ndcg_100 benchmark");
            return;
        }
    };

    let config = flash_rerank::ModelConfig::default();
    let cache_dir = default_cache_dir();
    let registry = flash_rerank::models::ModelRegistry::new(cache_dir);
    let model_dir = match registry.load(&config.model_id) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping msmarco_ndcg_100: {e}");
            return;
        }
    };
    let scorer = match flash_rerank::engine::ort_backend::OrtScorer::new(config, &model_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Skipping msmarco_ndcg_100: {e}");
            return;
        }
    };

    let evaluator =
        match flash_rerank_benchmarks::msmarco::MsmarcoEvaluator::load(&msmarco_path, Some(100)) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to load MSMARCO dataset: {e}");
                return;
            }
        };

    println!(
        "Loaded MSMARCO: {} queries, {} corpus docs",
        evaluator.queries.len(),
        evaluator.corpus.len()
    );

    let mut group = c.benchmark_group("msmarco_accuracy");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    group.bench_function("ndcg10_100q", |b| {
        b.iter(|| evaluator.evaluate_ndcg(&scorer, 10, 100));
    });

    // Also report the actual NDCG@10 value
    let ndcg = evaluator.evaluate_ndcg(&scorer, 10, 100);
    println!("MSMARCO NDCG@10 (100 queries, top-100 candidates): {ndcg:.4}");

    let mrr = evaluator.evaluate_mrr(&scorer, 100);
    println!("MSMARCO MRR (100 queries, top-100 candidates): {mrr:.4}");

    // Save results
    let results_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("results");
    let _ = std::fs::create_dir_all(&results_dir);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let result = serde_json::json!({
        "benchmark": "msmarco_accuracy",
        "queries": 100,
        "top_candidates": 100,
        "ndcg_at_10": ndcg,
        "mrr": mrr,
        "timestamp": timestamp,
    });
    let path = results_dir.join(format!("msmarco_100q_{timestamp}.json"));
    if let Ok(json) = serde_json::to_string_pretty(&result) {
        let _ = std::fs::write(&path, json);
        println!("Results saved to: {}", path.display());
    }

    group.finish();
}

fn msmarco_ndcg_1000(c: &mut Criterion) {
    let msmarco_path = match get_msmarco_path() {
        Some(p) => p,
        None => {
            eprintln!("MSMARCO_PATH not set, skipping msmarco_ndcg_1000 benchmark");
            return;
        }
    };

    let config = flash_rerank::ModelConfig::default();
    let cache_dir = default_cache_dir();
    let registry = flash_rerank::models::ModelRegistry::new(cache_dir);
    let model_dir = match registry.load(&config.model_id) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping msmarco_ndcg_1000: {e}");
            return;
        }
    };
    let scorer = match flash_rerank::engine::ort_backend::OrtScorer::new(config, &model_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Skipping msmarco_ndcg_1000: {e}");
            return;
        }
    };

    let evaluator =
        match flash_rerank_benchmarks::msmarco::MsmarcoEvaluator::load(&msmarco_path, Some(1000)) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to load MSMARCO dataset: {e}");
                return;
            }
        };

    println!(
        "Loaded MSMARCO: {} queries, {} corpus docs",
        evaluator.queries.len(),
        evaluator.corpus.len()
    );

    let mut group = c.benchmark_group("msmarco_accuracy");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10);

    group.bench_function("ndcg10_1000q", |b| {
        b.iter(|| evaluator.evaluate_ndcg(&scorer, 10, 100));
    });

    let ndcg = evaluator.evaluate_ndcg(&scorer, 10, 100);
    println!("MSMARCO NDCG@10 (1000 queries, top-100 candidates): {ndcg:.4}");

    let mrr = evaluator.evaluate_mrr(&scorer, 100);
    println!("MSMARCO MRR (1000 queries, top-100 candidates): {mrr:.4}");

    // Save results
    let results_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("results");
    let _ = std::fs::create_dir_all(&results_dir);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let result = serde_json::json!({
        "benchmark": "msmarco_accuracy",
        "queries": 1000,
        "top_candidates": 100,
        "ndcg_at_10": ndcg,
        "mrr": mrr,
        "timestamp": timestamp,
    });
    let path = results_dir.join(format!("msmarco_1000q_{timestamp}.json"));
    if let Ok(json) = serde_json::to_string_pretty(&result) {
        let _ = std::fs::write(&path, json);
        println!("Results saved to: {}", path.display());
    }

    group.finish();
}

criterion_group!(benches, msmarco_ndcg_100, msmarco_ndcg_1000);
criterion_main!(benches);
