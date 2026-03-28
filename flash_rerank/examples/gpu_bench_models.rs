//! GPU vs CPU benchmark across multiple model sizes.
use std::time::Instant;

use flash_rerank::engine::Scorer;
use flash_rerank::engine::ort_backend::OrtScorer;
use flash_rerank::engine::parallel::ParallelScorer;
use flash_rerank::models::ModelRegistry;
use flash_rerank::{Device, ModelConfig, Precision};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache_dir = home_cache_dir();
    let registry = ModelRegistry::new(cache_dir);

    let query = "What are the latest advances in distributed systems?";
    let docs: Vec<String> = (0..100)
        .map(|i| {
            format!(
                "Document {i} covering various aspects of computer science, distributed computing, \
             and system architecture with detailed analysis of modern approaches."
            )
        })
        .collect();

    let models = [
        ("MiniLM-L-6 (22M)", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        ("MiniLM-L-12 (33M)", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
    ];

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║          GPU vs CPU Benchmark — Multiple Models              ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    for (label, model_id) in &models {
        let model_dir = match registry.load(model_id) {
            Ok(d) => d,
            Err(e) => {
                println!("{label}: SKIP ({e})\n");
                continue;
            }
        };

        println!("=== {label} ===\n");

        // CPU INT8 (single)
        let config = ModelConfig {
            model_id: model_id.to_string(),
            device: Device::Cpu,
            ..Default::default()
        };
        match OrtScorer::new(config, &model_dir) {
            Ok(scorer) => {
                let p50 = bench_scorer(&scorer, query, &docs);
                println!("  CPU INT8 (single):   {:>7.2}ms", p50);
            }
            Err(e) => println!("  CPU INT8: FAILED ({e})"),
        }

        // CPU INT8 (parallel)
        let config = ModelConfig {
            model_id: model_id.to_string(),
            device: Device::Cpu,
            ..Default::default()
        };
        match ParallelScorer::new(config, &model_dir, None) {
            Ok(scorer) => {
                let p50 = bench_scorer(&scorer, query, &docs);
                println!("  CPU INT8 (parallel): {:>7.2}ms", p50);
            }
            Err(e) => println!("  CPU parallel: FAILED ({e})"),
        }

        // CUDA FP32
        let config = ModelConfig {
            model_id: model_id.to_string(),
            device: Device::Cuda(0),
            precision: Precision::FP32,
            ..Default::default()
        };
        match OrtScorer::new(config, &model_dir) {
            Ok(scorer) => {
                let p50 = bench_scorer(&scorer, query, &docs);
                println!("  CUDA FP32:           {:>7.2}ms", p50);
            }
            Err(e) => println!("  CUDA FP32: FAILED ({e})"),
        }

        // CUDA FP16
        let config = ModelConfig {
            model_id: model_id.to_string(),
            device: Device::Cuda(0),
            precision: Precision::FP16,
            ..Default::default()
        };
        match OrtScorer::new(config, &model_dir) {
            Ok(scorer) => {
                let p50 = bench_scorer(&scorer, query, &docs);
                println!("  CUDA FP16:           {:>7.2}ms", p50);
            }
            Err(e) => println!("  CUDA FP16: FAILED ({e})"),
        }

        // TensorRT FP16
        let config = ModelConfig {
            model_id: model_id.to_string(),
            device: Device::TensorRT(0),
            precision: Precision::FP16,
            ..Default::default()
        };
        match OrtScorer::new(config, &model_dir) {
            Ok(scorer) => {
                println!("  TensorRT FP16:       (warming up TRT engine...)");
                let p50 = bench_scorer(&scorer, query, &docs);
                println!("  TensorRT FP16:       {:>7.2}ms", p50);
            }
            Err(e) => println!("  TensorRT FP16: FAILED ({e})"),
        }

        println!();
    }

    // Check BGE model if ONNX exists
    let bge_id = "BAAI/bge-reranker-v2-m3";
    if let Ok(model_dir) = registry.load(bge_id) {
        println!("=== BGE-Reranker-v2-m3 (568M) ===\n");

        let config = ModelConfig {
            model_id: bge_id.to_string(),
            device: Device::Cpu,
            ..Default::default()
        };
        match OrtScorer::new(config, &model_dir) {
            Ok(scorer) => {
                let p50 = bench_scorer(&scorer, query, &docs);
                println!("  CPU (single):        {:>7.2}ms", p50);
            }
            Err(e) => println!("  CPU: FAILED ({e})"),
        }

        let config = ModelConfig {
            model_id: bge_id.to_string(),
            device: Device::Cuda(0),
            ..Default::default()
        };
        match OrtScorer::new(config, &model_dir) {
            Ok(scorer) => {
                let p50 = bench_scorer(&scorer, query, &docs);
                println!("  CUDA FP32:           {:>7.2}ms", p50);
            }
            Err(e) => println!("  CUDA FP32: FAILED ({e})"),
        }

        let config = ModelConfig {
            model_id: bge_id.to_string(),
            device: Device::TensorRT(0),
            precision: Precision::FP16,
            ..Default::default()
        };
        match OrtScorer::new(config, &model_dir) {
            Ok(scorer) => {
                println!("  TensorRT FP16:       (warming up TRT engine...)");
                let p50 = bench_scorer(&scorer, query, &docs);
                println!("  TensorRT FP16:       {:>7.2}ms", p50);
            }
            Err(e) => println!("  TensorRT FP16: FAILED ({e})"),
        }
        println!();
    }

    Ok(())
}

fn bench_scorer(scorer: &dyn Scorer, query: &str, docs: &[String]) -> f64 {
    // Warm up
    for _ in 0..5 {
        let _ = scorer.score(query, docs);
    }

    let mut times = Vec::new();
    for _ in 0..20 {
        let t = Instant::now();
        let _ = scorer.score(query, docs);
        times.push(t.elapsed());
    }
    times.sort();
    times[10].as_secs_f64() * 1000.0
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
