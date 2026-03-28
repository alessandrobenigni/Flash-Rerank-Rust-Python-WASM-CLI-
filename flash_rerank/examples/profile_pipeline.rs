use std::time::Instant;

use flash_rerank::engine::Scorer;
use flash_rerank::engine::ort_backend::OrtScorer;
use flash_rerank::models::ModelRegistry;
use flash_rerank::tokenize::Tokenizer;
use flash_rerank::{Device, ModelConfig, Precision};

fn main() -> anyhow::Result<()> {
    let cache_dir = home_cache_dir();
    let registry = ModelRegistry::new(cache_dir);
    let model_dir = registry.load("cross-encoder/ms-marco-MiniLM-L-6-v2")?;

    let query = "What is the capital of France?";
    let docs: Vec<String> = (0..100)
        .map(|i| {
            format!(
                "Document {i} about various topics including geography, science, and history. \
                 Paris is the capital and most populous city of France, with an area of 105 square km."
            )
        })
        .collect();

    // --- Profile tokenization alone ---
    let tok_path = model_dir.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tok_path)?;

    // Warm up tokenizer
    for _ in 0..5 {
        let _ = tokenizer.tokenize_pairs(query, &docs, 512);
    }

    let mut tok_times = Vec::new();
    for _ in 0..50 {
        let t = Instant::now();
        let _ = tokenizer.tokenize_pairs(query, &docs, 512)?;
        tok_times.push(t.elapsed());
    }
    tok_times.sort();
    println!(
        "=== TOKENIZATION (100 pairs, max_length=512) ===\n  P50: {:.2}ms\n  Mean: {:.2}ms\n  Min: {:.2}ms",
        tok_times[25].as_secs_f64() * 1000.0,
        tok_times.iter().sum::<std::time::Duration>().as_secs_f64() / 50.0 * 1000.0,
        tok_times[0].as_secs_f64() * 1000.0,
    );

    // With shorter max_length
    let mut tok128_times = Vec::new();
    for _ in 0..50 {
        let t = Instant::now();
        let _ = tokenizer.tokenize_pairs(query, &docs, 128)?;
        tok128_times.push(t.elapsed());
    }
    tok128_times.sort();
    println!(
        "\n=== TOKENIZATION (100 pairs, max_length=128) ===\n  P50: {:.2}ms\n  Mean: {:.2}ms",
        tok128_times[25].as_secs_f64() * 1000.0,
        tok128_times
            .iter()
            .sum::<std::time::Duration>()
            .as_secs_f64()
            / 50.0
            * 1000.0,
    );

    // --- Profile tensor creation ---
    let encodings = tokenizer.tokenize_pairs(query, &docs, 128)?;
    let batch_size = encodings.len();
    let seq_len = encodings[0].get_ids().len();
    println!("\n=== TENSOR SHAPE: [{batch_size}, {seq_len}] ===");

    let mut tensor_times = Vec::new();
    for _ in 0..50 {
        let t = Instant::now();
        let _input_ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|&id| id as i64))
            .collect();
        let _attention_mask: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|&m| m as i64))
            .collect();
        let _token_type_ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_type_ids().iter().map(|&t| t as i64))
            .collect();
        tensor_times.push(t.elapsed());
    }
    tensor_times.sort();
    println!(
        "  Tensor build P50: {:.3}ms",
        tensor_times[25].as_secs_f64() * 1000.0
    );

    // --- Profile parallel scorer (CPU) ---
    {
        let config = ModelConfig {
            device: Device::Cpu,
            ..Default::default()
        };
        let t0 = Instant::now();
        let parallel =
            flash_rerank::engine::parallel::ParallelScorer::new(config, &model_dir, None).unwrap();
        println!(
            "\n=== PARALLEL CPU (load: {:.0}ms) ===",
            t0.elapsed().as_secs_f64() * 1000.0
        );
        // Warm up
        for _ in 0..5 {
            let _ = parallel.score(query, &docs);
        }
        for &n in &[1usize, 10, 50, 100] {
            let subset = &docs[..n];
            let mut times = Vec::new();
            let iters = if n <= 10 { 100 } else { 30 };
            for _ in 0..iters {
                let t = Instant::now();
                let _ = parallel.score(query, subset);
                times.push(t.elapsed());
            }
            times.sort();
            let p50 = times[times.len() / 2];
            let p99 = times[((times.len() as f64 * 0.99) as usize).min(times.len() - 1)];
            println!(
                "  {n:>4} docs: P50={:>7.2}ms  P99={:>7.2}ms  QPS={:.1}",
                p50.as_secs_f64() * 1000.0,
                p99.as_secs_f64() * 1000.0,
                iters as f64 / times.iter().sum::<std::time::Duration>().as_secs_f64()
            );
        }
    }

    // --- Profile full pipeline per device ---
    for (device_name, device) in [
        ("CPU_SINGLE", Device::Cpu),
        ("CUDA", Device::Cuda(0)),
        ("CUDA_FP16", Device::Cuda(0)),
    ] {
        let config = ModelConfig {
            device: device.clone(),
            precision: if device_name == "CUDA_FP16" {
                Precision::FP16
            } else {
                Precision::FP32
            },
            ..Default::default()
        };

        let t0 = Instant::now();
        let scorer = match OrtScorer::new(config, &model_dir) {
            Ok(s) => s,
            Err(e) => {
                println!("\n=== {device_name}: SKIPPED ({e}) ===\n");
                continue;
            }
        };
        println!(
            "\n=== {device_name} (load: {:.0}ms) ===",
            t0.elapsed().as_secs_f64() * 1000.0
        );

        // Warm up
        for _ in 0..5 {
            let _ = scorer.score(query, &docs);
        }

        // Profile: 1, 10, 50, 100 docs
        for &n in &[1usize, 10, 50, 100] {
            let subset = &docs[..n];
            let mut times = Vec::new();
            let iters = if n <= 10 { 100 } else { 30 };
            for _ in 0..iters {
                let t = Instant::now();
                let _ = scorer.score(query, subset);
                times.push(t.elapsed());
            }
            times.sort();
            let p50 = times[times.len() / 2];
            let p99 = times[((times.len() as f64 * 0.99) as usize).min(times.len() - 1)];
            println!(
                "  {n:>4} docs: P50={:>7.2}ms  P99={:>7.2}ms  QPS={:.1}",
                p50.as_secs_f64() * 1000.0,
                p99.as_secs_f64() * 1000.0,
                iters as f64 / times.iter().sum::<std::time::Duration>().as_secs_f64()
            );
        }
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
