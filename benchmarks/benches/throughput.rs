use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::sync::Arc;
use std::time::Duration;

/// Sustained throughput benchmark: sequential vs. batched concurrent queries.
///
/// Measures QPS (queries per second) for:
/// - Sequential baseline: one query at a time through the scorer
/// - Concurrent via DynamicBatcher: multiple submitters contending
///
/// Both require a downloaded model at runtime.

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

fn load_scorer() -> flash_rerank::engine::ort_backend::OrtScorer {
    let config = flash_rerank::ModelConfig::default();
    let cache_dir = default_cache_dir();
    let registry = flash_rerank::models::ModelRegistry::new(cache_dir);
    let model_dir = registry
        .load(&config.model_id)
        .expect("Model not downloaded. Run: flash-rerank download --model cross-encoder/ms-marco-MiniLM-L-6-v2");

    flash_rerank::engine::ort_backend::OrtScorer::new(config, &model_dir)
        .expect("Failed to create scorer")
}

fn generate_docs(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("Document number {i} about various topics in science and technology"))
        .collect()
}

fn sequential_throughput(c: &mut Criterion) {
    let scorer = load_scorer();
    let query = "What is machine learning?";
    let docs = generate_docs(10);

    let mut group = c.benchmark_group("throughput_sequential");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    group.bench_function("10_docs", |b| {
        b.iter(|| {
            use flash_rerank::engine::Scorer;
            scorer.score(query, &docs).unwrap()
        });
    });

    group.finish();
}

fn batched_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let config = flash_rerank::ModelConfig::default();
    let cache_dir = default_cache_dir();
    let registry = flash_rerank::models::ModelRegistry::new(cache_dir);
    let model_dir = registry
        .load(&config.model_id)
        .expect("Model not downloaded");
    let scorer = flash_rerank::engine::ort_backend::OrtScorer::new(config, &model_dir)
        .expect("Failed to create scorer");

    let batcher = rt.block_on(async {
        Arc::new(flash_rerank::batch::DynamicBatcher::new(
            16,
            Duration::from_millis(5),
            Box::new(scorer),
        ))
    });

    let query = "What is machine learning?";
    let docs = generate_docs(10);

    let mut group = c.benchmark_group("throughput_batched");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for concurrency in [1, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("concurrent_submitters", concurrency),
            &concurrency,
            |b, &n_submitters| {
                b.to_async(&rt).iter(|| {
                    let batcher = batcher.clone();
                    let query = query.to_string();
                    let docs = docs.clone();
                    async move {
                        let mut handles = Vec::with_capacity(n_submitters);
                        for _ in 0..n_submitters {
                            let b = batcher.clone();
                            let q = query.clone();
                            let d = docs.clone();
                            handles.push(tokio::spawn(async move {
                                b.submit(q, d, None).await.unwrap()
                            }));
                        }
                        for h in handles {
                            h.await.unwrap();
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, sequential_throughput, batched_throughput);
criterion_main!(benches);
