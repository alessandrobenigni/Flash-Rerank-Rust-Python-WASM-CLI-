use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn rerank_latency_benchmark(c: &mut Criterion) {
    // Setup: load model outside benchmark loop.
    // This requires a downloaded model -- the benchmark will panic if not available.
    let config = flash_rerank::ModelConfig::default();

    let cache_dir = default_cache_dir();
    let registry = flash_rerank::models::ModelRegistry::new(cache_dir);
    let model_dir = registry
        .load(&config.model_id)
        .expect("Model not downloaded. Run: flash-rerank download --model cross-encoder/ms-marco-MiniLM-L-6-v2");

    let scorer = flash_rerank::engine::ort_backend::OrtScorer::new(config, &model_dir)
        .expect("Failed to create scorer");

    let query = "What is machine learning?";

    let mut group = c.benchmark_group("rerank");

    for doc_count in [1, 10, 50, 100, 500, 1000] {
        let docs: Vec<String> = (0..doc_count)
            .map(|i| format!("Document number {i} about various topics in science and technology"))
            .collect();

        group.bench_with_input(BenchmarkId::new("docs", doc_count), &docs, |b, docs| {
            b.iter(|| {
                use flash_rerank::engine::Scorer;
                scorer.score(query, docs).unwrap()
            });
        });
    }

    group.finish();
}

fn default_cache_dir() -> std::path::PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return std::path::PathBuf::from(cache);
    }
    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        return std::path::PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    std::path::PathBuf::from(".cache/huggingface/hub")
}

criterion_group!(benches, rerank_latency_benchmark);
criterion_main!(benches);
