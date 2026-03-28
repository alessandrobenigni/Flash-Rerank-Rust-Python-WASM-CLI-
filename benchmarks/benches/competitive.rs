use criterion::{Criterion, criterion_group, criterion_main};
use std::time::Duration;

/// Competitive benchmark: Flash-Rerank vs published API baselines.
///
/// Runs reranking on 100 documents and records latency for CPU and CUDA.
/// Published baselines for comparison (100 documents):
/// - Jina Reranker API:   ~188ms
/// - Cohere Rerank API:   ~595ms
/// - Voyage Rerank API:   ~603ms
///
/// Results are saved to benchmarks/results/ for tracking over time.

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

fn generate_docs(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            format!(
                "Document {i}: This is a comprehensive article about topic number {i} \
                 covering various aspects of science, technology, and their applications \
                 in modern society. It discusses the fundamental principles and recent \
                 advances in the field."
            )
        })
        .collect()
}

fn competitive_cpu(c: &mut Criterion) {
    let config = flash_rerank::ModelConfig {
        device: flash_rerank::Device::Cpu,
        ..flash_rerank::ModelConfig::default()
    };
    let cache_dir = default_cache_dir();
    let registry = flash_rerank::models::ModelRegistry::new(cache_dir);
    let model_dir = match registry.load(&config.model_id) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping CPU competitive benchmark: {e}");
            return;
        }
    };
    let scorer = match flash_rerank::engine::ort_backend::OrtScorer::new(config, &model_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Skipping CPU competitive benchmark: {e}");
            return;
        }
    };

    let query = "What are the latest advances in machine learning and artificial intelligence?";
    let docs = generate_docs(100);

    let mut group = c.benchmark_group("competitive");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    group.bench_function("cpu_100docs", |b| {
        b.iter(|| {
            use flash_rerank::engine::Scorer;
            scorer.score(query, &docs).unwrap()
        });
    });

    group.finish();

    // Save results summary
    save_results("cpu", &scorer, query, &docs);
}

fn competitive_cuda(c: &mut Criterion) {
    let config = flash_rerank::ModelConfig {
        device: flash_rerank::Device::Cuda(0),
        ..flash_rerank::ModelConfig::default()
    };
    let cache_dir = default_cache_dir();
    let registry = flash_rerank::models::ModelRegistry::new(cache_dir);
    let model_dir = match registry.load(&config.model_id) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping CUDA competitive benchmark: {e}");
            return;
        }
    };
    let scorer = match flash_rerank::engine::ort_backend::OrtScorer::new(config, &model_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Skipping CUDA competitive benchmark (no GPU?): {e}");
            return;
        }
    };

    let query = "What are the latest advances in machine learning and artificial intelligence?";
    let docs = generate_docs(100);

    let mut group = c.benchmark_group("competitive");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    group.bench_function("cuda_100docs", |b| {
        b.iter(|| {
            use flash_rerank::engine::Scorer;
            scorer.score(query, &docs).unwrap()
        });
    });

    group.finish();

    save_results("cuda", &scorer, query, &docs);
}

/// Run a quick measurement and save results to benchmarks/results/.
fn save_results(
    device_label: &str,
    scorer: &flash_rerank::engine::ort_backend::OrtScorer,
    query: &str,
    docs: &[String],
) {
    use flash_rerank::engine::Scorer;
    use std::time::Instant;

    // Warmup
    let _ = scorer.score(query, docs);

    // Measure 10 iterations
    let mut durations = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        let _ = scorer.score(query, docs);
        durations.push(start.elapsed());
    }
    durations.sort();

    let median = durations[durations.len() / 2];
    let mean: Duration = durations.iter().sum::<Duration>() / durations.len() as u32;

    let results_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("results");
    let _ = std::fs::create_dir_all(&results_dir);

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let result = serde_json::json!({
        "benchmark": "competitive_100docs",
        "device": device_label,
        "timestamp": timestamp,
        "docs_count": docs.len(),
        "iterations": durations.len(),
        "median_ms": median.as_secs_f64() * 1000.0,
        "mean_ms": mean.as_secs_f64() * 1000.0,
        "p95_ms": durations[(durations.len() as f64 * 0.95) as usize].as_secs_f64() * 1000.0,
        "baselines": {
            "jina_api_ms": 188.0,
            "cohere_api_ms": 595.0,
            "voyage_api_ms": 603.0,
        }
    });

    let filename = format!("competitive_{device_label}_{timestamp}.json");
    let path = results_dir.join(&filename);
    if let Ok(json) = serde_json::to_string_pretty(&result) {
        let _ = std::fs::write(&path, json);
        println!("\nResults saved to: {}", path.display());
        println!(
            "Flash-Rerank {device_label}: median={:.1}ms, mean={:.1}ms",
            median.as_secs_f64() * 1000.0,
            mean.as_secs_f64() * 1000.0,
        );
        println!("Baselines: Jina=188ms, Cohere=595ms, Voyage=603ms");
    }
}

fn competitive_parallel_cpu(c: &mut Criterion) {
    let config = flash_rerank::ModelConfig {
        device: flash_rerank::Device::Cpu,
        ..flash_rerank::ModelConfig::default()
    };
    let cache_dir = default_cache_dir();
    let registry = flash_rerank::models::ModelRegistry::new(cache_dir);
    let model_dir = match registry.load(&config.model_id) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping parallel CPU benchmark: {e}");
            return;
        }
    };
    let scorer = match flash_rerank::engine::parallel::ParallelScorer::new(config, &model_dir, None)
    {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Skipping parallel CPU benchmark: {e}");
            return;
        }
    };

    let query = "What are the latest advances in machine learning and artificial intelligence?";
    let docs = generate_docs(100);

    // Warm up
    for _ in 0..5 {
        use flash_rerank::engine::Scorer;
        let _ = scorer.score(query, &docs);
    }

    let mut group = c.benchmark_group("competitive");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    group.bench_function("parallel_cpu_100docs", |b| {
        b.iter(|| {
            use flash_rerank::engine::Scorer;
            scorer.score(query, &docs).unwrap()
        });
    });

    group.finish();

    // Save results manually
    {
        use flash_rerank::engine::Scorer;
        use std::time::Instant;
        let mut durations = Vec::with_capacity(10);
        for _ in 0..10 {
            let start = Instant::now();
            let _ = scorer.score(query, &docs);
            durations.push(start.elapsed());
        }
        durations.sort();
        let median = durations[5];
        let mean: Duration = durations.iter().sum::<Duration>() / 10;
        let results_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("results");
        let _ = std::fs::create_dir_all(&results_dir);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let result = serde_json::json!({
            "benchmark": "competitive_100docs",
            "device": "parallel_cpu",
            "timestamp": timestamp,
            "docs_count": docs.len(),
            "median_ms": median.as_secs_f64() * 1000.0,
            "mean_ms": mean.as_secs_f64() * 1000.0,
            "baselines": { "jina_api_ms": 188.0, "cohere_api_ms": 595.0, "voyage_api_ms": 603.0 }
        });
        let path = results_dir.join(format!("competitive_parallel_cpu_{timestamp}.json"));
        let _ = std::fs::write(&path, serde_json::to_string_pretty(&result).unwrap());
        println!("\nResults saved to: {}", path.display());
        println!(
            "Flash-Rerank parallel_cpu: median={:.1}ms, mean={:.1}ms",
            median.as_secs_f64() * 1000.0,
            mean.as_secs_f64() * 1000.0
        );
        println!("Baselines: Jina=188ms, Cohere=595ms, Voyage=603ms");
    }
}

criterion_group!(
    benches,
    competitive_parallel_cpu,
    competitive_cpu,
    competitive_cuda
);
criterion_main!(benches);
