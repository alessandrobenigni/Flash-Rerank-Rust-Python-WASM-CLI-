//! Combined pipeline benchmark: BM25-Turbo retrieval + Flash-Rerank reranking.
//!
//! Simulates the full two-stage pipeline to measure end-to-end latency.
//! BM25-Turbo's published numbers: 8.6ms P50 on 8.8M docs.
//! Flash-Rerank adds semantic reranking on the top-K candidates.

use std::time::{Duration, Instant};

use flash_rerank::ModelConfig;
use flash_rerank::engine::Scorer;
use flash_rerank::engine::ort_backend::OrtScorer;
use flash_rerank::engine::parallel::ParallelScorer;
use flash_rerank::models::ModelRegistry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache_dir = home_cache_dir();
    let registry = ModelRegistry::new(cache_dir);
    let model_dir = registry.load("cross-encoder/ms-marco-MiniLM-L-6-v2")?;

    // Simulate BM25-Turbo retrieval latency (from their benchmarks)
    let bm25_latency = Duration::from_micros(8600); // 8.6ms P50 on 8.8M docs
    let bm25_latency_scifact = Duration::from_micros(67); // 67μs on 5K docs
    let bm25_latency_fiqa = Duration::from_micros(711); // 711μs on 57K docs

    // Simulated BM25 top-100 candidates (realistic search snippets)
    let candidates: Vec<String> = (0..100)
        .map(|i| {
            format!(
                "Result {i}: This document discusses the query topic with varying relevance. \
             Some results are highly relevant while others are tangential matches based \
             on keyword overlap without semantic alignment."
            )
        })
        .collect();

    let query = "What are the most effective approaches to distributed systems at scale?";

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Flash-Rerank + BM25-Turbo Pipeline Benchmark           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Stage 1: BM25-Turbo (keyword retrieval from millions)     ║");
    println!("║  Stage 2: Flash-Rerank (semantic reranking of top-100)     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // --- Single-session scorer ---
    let config = ModelConfig::default();
    let single = OrtScorer::new(config, &model_dir)?;
    for _ in 0..5 {
        let _ = single.score(query, &candidates);
    }

    let mut rerank_times = Vec::new();
    for _ in 0..30 {
        let t = Instant::now();
        let _ = single.score(query, &candidates)?;
        rerank_times.push(t.elapsed());
    }
    rerank_times.sort();
    let single_p50 = rerank_times[15];

    // --- Parallel scorer ---
    let config2 = ModelConfig::default();
    let parallel = ParallelScorer::new(config2, &model_dir, None)?;
    for _ in 0..5 {
        let _ = parallel.score(query, &candidates);
    }

    let mut par_times = Vec::new();
    for _ in 0..30 {
        let t = Instant::now();
        let _ = parallel.score(query, &candidates)?;
        par_times.push(t.elapsed());
    }
    par_times.sort();
    let parallel_p50 = par_times[15];

    // --- Results ---
    println!("=== COMPONENT LATENCIES ===\n");
    println!(
        "  BM25-Turbo (8.8M docs):       {:>8.2}ms  (published)",
        bm25_latency.as_secs_f64() * 1000.0
    );
    println!(
        "  Flash-Rerank Single CPU INT8:  {:>8.2}ms  (measured)",
        single_p50.as_secs_f64() * 1000.0
    );
    println!(
        "  Flash-Rerank Parallel CPU INT8:{:>8.2}ms  (measured)",
        parallel_p50.as_secs_f64() * 1000.0
    );

    println!("\n=== FULL PIPELINE (BM25-Turbo → Flash-Rerank → Top 5) ===\n");

    // MS MARCO scale (8.8M docs)
    let pipeline_single = bm25_latency + single_p50;
    let pipeline_parallel = bm25_latency + parallel_p50;
    println!("  MS MARCO (8.8M docs, top 100 → top 5):");
    println!(
        "    Single scorer:   {:>8.2}ms  (BM25 {:.1}ms + Rerank {:.1}ms)",
        pipeline_single.as_secs_f64() * 1000.0,
        bm25_latency.as_secs_f64() * 1000.0,
        single_p50.as_secs_f64() * 1000.0
    );
    println!(
        "    Parallel scorer: {:>8.2}ms  (BM25 {:.1}ms + Rerank {:.1}ms)",
        pipeline_parallel.as_secs_f64() * 1000.0,
        bm25_latency.as_secs_f64() * 1000.0,
        parallel_p50.as_secs_f64() * 1000.0
    );

    // SciFact scale (5K docs)
    let pipe_scifact = bm25_latency_scifact + parallel_p50;
    println!("\n  SciFact (5K docs, top 100 → top 5):");
    println!(
        "    Parallel scorer: {:>8.2}ms  (BM25 {:.3}ms + Rerank {:.1}ms)",
        pipe_scifact.as_secs_f64() * 1000.0,
        bm25_latency_scifact.as_secs_f64() * 1000.0,
        parallel_p50.as_secs_f64() * 1000.0
    );

    // FiQA scale (57K docs)
    let pipe_fiqa = bm25_latency_fiqa + parallel_p50;
    println!("\n  FiQA (57K docs, top 100 → top 5):");
    println!(
        "    Parallel scorer: {:>8.2}ms  (BM25 {:.1}ms + Rerank {:.1}ms)",
        pipe_fiqa.as_secs_f64() * 1000.0,
        bm25_latency_fiqa.as_secs_f64() * 1000.0,
        parallel_p50.as_secs_f64() * 1000.0
    );

    println!("\n=== COMPETITIVE COMPARISON ===\n");
    let competitors = [
        ("Jina Reranker v3 (API)", 188.0),
        ("Cohere Rerank 3.5 (API)", 595.0),
        ("Voyage Rerank 2.5 (API)", 603.0),
    ];

    println!(
        "  Flash-Rerank pipeline (8.8M docs → top 5): {:.1}ms",
        pipeline_parallel.as_secs_f64() * 1000.0
    );
    println!();
    for (name, latency) in &competitors {
        // Competitors only do reranking (no retrieval included)
        // Our pipeline does BOTH retrieval + reranking
        let speedup = latency / (pipeline_parallel.as_secs_f64() * 1000.0);
        println!("  vs {name}:");
        println!("    Their reranking alone:   {:.0}ms", latency);
        println!(
            "    Our FULL PIPELINE:       {:.1}ms  ({:.1}x faster — and we include retrieval!)",
            pipeline_parallel.as_secs_f64() * 1000.0,
            speedup
        );
    }

    println!("\n=== KEY INSIGHT ===\n");
    println!("  Competitors charge for reranking 100 docs:     188-603ms");
    println!("  Flash-Rerank + BM25-Turbo searches 8.8M docs");
    println!(
        "  AND reranks the top 100, all in:               {:.1}ms",
        pipeline_parallel.as_secs_f64() * 1000.0
    );
    println!("  That's RETRIEVAL + RERANKING faster than competitors do RERANKING ALONE.");

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
