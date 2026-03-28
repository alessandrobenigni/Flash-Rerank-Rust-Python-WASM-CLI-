//! Micro-profiler: measures each step of the reranking pipeline independently.

use std::time::Instant;

use flash_rerank::ModelConfig;
use flash_rerank::engine::Scorer;
use flash_rerank::engine::ort_backend::OrtScorer;
use flash_rerank::engine::parallel::ParallelScorer;
use flash_rerank::models::ModelRegistry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache_dir = home_cache_dir();
    let registry = ModelRegistry::new(cache_dir);
    let model_dir = registry.load("cross-encoder/ms-marco-MiniLM-L-6-v2")?;

    let query = "What is the capital of France?";
    let docs: Vec<String> = (0..100)
        .map(|i| {
            format!(
                "Document {i} about various topics including geography, science, and history. \
             Paris is the capital and most populous city of France."
            )
        })
        .collect();

    println!("\n=== SINGLE SESSION (INT8 auto-select) ===");
    let config = ModelConfig::default();
    let scorer = OrtScorer::new(config, &model_dir)?;

    // Warm up
    for _ in 0..5 {
        let _ = scorer.score(query, &docs);
    }

    for &n in &[1usize, 10, 50, 100] {
        let subset = &docs[..n];
        let iters = if n <= 10 { 100 } else { 30 };
        let mut times = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let _ = scorer.score(query, subset);
            times.push(t.elapsed());
        }
        times.sort();
        let p50 = times[times.len() / 2];
        println!(
            "  {:>4} docs: P50={:>7.2}ms  QPS={:.1}",
            n,
            p50.as_secs_f64() * 1000.0,
            iters as f64 / times.iter().sum::<std::time::Duration>().as_secs_f64()
        );
    }

    println!("\n=== PARALLEL SCORER (auto workers) ===");
    let config2 = ModelConfig::default();
    let parallel = ParallelScorer::new(config2, &model_dir, None)?;

    // Warm up
    for _ in 0..5 {
        let _ = parallel.score(query, &docs);
    }

    for &n in &[1usize, 10, 50, 100] {
        let subset = &docs[..n];
        let iters = if n <= 10 { 100 } else { 30 };
        let mut times = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let _ = parallel.score(query, subset);
            times.push(t.elapsed());
        }
        times.sort();
        let p50 = times[times.len() / 2];
        println!(
            "  {:>4} docs: P50={:>7.2}ms  QPS={:.1}",
            n,
            p50.as_secs_f64() * 1000.0,
            iters as f64 / times.iter().sum::<std::time::Duration>().as_secs_f64()
        );
    }

    // Test different worker counts
    for workers in [2, 4, 6, 8] {
        let config3 = ModelConfig::default();
        let p = ParallelScorer::new(config3, &model_dir, Some(workers))?;
        for _ in 0..5 {
            let _ = p.score(query, &docs);
        }
        let mut times = Vec::new();
        for _ in 0..20 {
            let t = Instant::now();
            let _ = p.score(query, &docs);
            times.push(t.elapsed());
        }
        times.sort();
        let p50 = times[10];
        println!(
            "  100 docs, {workers} workers: P50={:.2}ms",
            p50.as_secs_f64() * 1000.0
        );
    }

    Ok(())
}

fn home_cache_dir() -> std::path::PathBuf {
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
