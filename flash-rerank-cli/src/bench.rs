use clap::Args;
use console::style;
use flash_rerank::engine::Scorer;
use flash_rerank::models::ModelRegistry;

#[derive(Args)]
pub struct BenchArgs {
    /// Model to benchmark (HuggingFace ID or local path).
    #[arg(short, long)]
    pub model: String,

    /// Number of queries to benchmark.
    #[arg(short, long, default_value = "100")]
    pub queries: usize,

    /// Number of documents per query.
    #[arg(short, long, default_value = "100")]
    pub documents: usize,

    /// BEIR dataset directory (optional). If provided, uses real queries/docs.
    #[arg(long)]
    pub dataset: Option<String>,

    /// Output format: "table" or "json".
    #[arg(long, default_value = "table")]
    pub output: String,

    /// P50 latency target in milliseconds for pass/fail.
    #[arg(long, default_value = "20")]
    pub target_ms: u64,
}

pub async fn run(args: BenchArgs) -> anyhow::Result<()> {
    println!(
        "{} Benchmarking model: {}",
        style("[bench]").cyan().bold(),
        style(&args.model).green()
    );

    // Load model
    let config = flash_rerank::ModelConfig {
        model_id: args.model.clone(),
        ..Default::default()
    };

    let cache_dir = default_cache_dir();
    let registry = ModelRegistry::new(cache_dir);
    let model_dir = registry.load(&config.model_id)?;
    let scorer = flash_rerank::engine::ort_backend::OrtScorer::new(config, &model_dir)?;

    // Prepare queries and documents
    let (queries, documents, relevance_data) = if let Some(ref dataset_path) = args.dataset {
        // Load BEIR dataset
        let path = std::path::PathBuf::from(dataset_path);
        let dataset_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        let parent = path.parent().unwrap_or(&path);

        println!(
            "  {} Loading BEIR dataset: {}",
            style(">").dim(),
            dataset_name
        );

        let dataset = flash_rerank_benchmarks::beir::BeirDataset::load(parent, dataset_name)
            .map_err(|e| anyhow::anyhow!("Failed to load dataset: {e}"))?;

        let num_q = args.queries.min(dataset.num_queries());
        let queries: Vec<String> = dataset.queries[..num_q].to_vec();
        let documents = dataset.documents.clone();
        let relevance = Some(dataset);

        (queries, documents, relevance)
    } else {
        // Generate synthetic data
        let queries: Vec<String> = (0..args.queries)
            .map(|i| format!("What is the meaning of query number {i}?"))
            .collect();
        let documents: Vec<String> = (0..args.documents)
            .map(|i| {
                format!(
                    "Document number {i} discussing various scientific and technological topics"
                )
            })
            .collect();
        (queries, documents, None)
    };

    let num_queries = queries.len();
    println!(
        "  {} {} queries, {} documents",
        style(">").dim(),
        num_queries,
        documents.len()
    );

    // Warm-up: 10 queries (excluded from measurements)
    let warmup_count = 10.min(num_queries);
    println!(
        "  {} Warming up ({warmup_count} queries)...",
        style(">").dim()
    );
    for q in queries.iter().take(warmup_count) {
        let _ = scorer.score(q, &documents);
    }

    // Benchmark with progress bar
    let pb = indicatif::ProgressBar::new(num_queries as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("=> "),
    );

    let stats = flash_rerank_benchmarks::latency::measure_latency(num_queries, || {
        // Use a rotating query index
        let idx = rand_index(num_queries);
        let _ = scorer.score(&queries[idx], &documents);
        pb.inc(1);
    });

    pb.finish_and_clear();

    // Compute accuracy metrics if BEIR dataset available
    let (ndcg_10, mrr_val) = if let Some(ref dataset) = relevance_data {
        let mut ndcg_sum = 0.0;
        let mut mrr_sum = 0.0;
        let eval_count = num_queries.min(dataset.num_queries());

        for q_idx in 0..eval_count {
            let results = scorer.score(&dataset.queries[q_idx], &dataset.documents)?;
            let ranked: Vec<usize> = results.iter().map(|r| r.index).collect();
            let rels = dataset.relevance_vector(q_idx);
            let relevant = dataset.relevant_docs(q_idx);

            ndcg_sum += flash_rerank_benchmarks::accuracy::ndcg_at_k(&ranked, &rels, 10);
            mrr_sum += flash_rerank_benchmarks::accuracy::mrr(&ranked, &relevant);
        }

        (
            Some(ndcg_sum / eval_count as f64),
            Some(mrr_sum / eval_count as f64),
        )
    } else {
        (None, None)
    };

    // Determine pass/fail
    let target = std::time::Duration::from_millis(args.target_ms);
    let passed = stats.p50 <= target;

    // Display results
    if args.output == "json" {
        let mut result = serde_json::json!({
            "model": args.model,
            "queries": num_queries,
            "documents": documents.len(),
            "p50_ms": stats.p50.as_secs_f64() * 1000.0,
            "p95_ms": stats.p95.as_secs_f64() * 1000.0,
            "p99_ms": stats.p99.as_secs_f64() * 1000.0,
            "mean_ms": stats.mean.as_secs_f64() * 1000.0,
            "throughput_qps": stats.throughput_qps,
            "target_ms": args.target_ms,
            "passed": passed,
        });
        if let Some(ndcg) = ndcg_10 {
            result["ndcg@10"] = serde_json::json!(ndcg);
        }
        if let Some(mrr) = mrr_val {
            result["mrr"] = serde_json::json!(mrr);
        }
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        // Table output
        println!();
        println!("{}", style("=== Benchmark Results ===").bold().underlined());
        println!("  Model:      {}", style(&args.model).green());
        println!("  Queries:    {num_queries}");
        println!("  Documents:  {}", documents.len());
        println!();
        println!("{}", style("--- Latency ---").bold());
        println!("  P50:        {:.3} ms", stats.p50.as_secs_f64() * 1000.0);
        println!("  P95:        {:.3} ms", stats.p95.as_secs_f64() * 1000.0);
        println!("  P99:        {:.3} ms", stats.p99.as_secs_f64() * 1000.0);
        println!("  Mean:       {:.3} ms", stats.mean.as_secs_f64() * 1000.0);
        println!("  QPS:        {:.1}", stats.throughput_qps);

        if let Some(ndcg) = ndcg_10 {
            println!();
            println!("{}", style("--- Accuracy ---").bold());
            println!("  NDCG@10:    {:.4}", ndcg);
        }
        if let Some(mrr) = mrr_val {
            println!("  MRR:        {:.4}", mrr);
        }

        println!();
        let target_label = format!("Target:     P50 < {} ms", args.target_ms);
        if passed {
            println!("  {} {}", style("PASS").green().bold(), target_label);
        } else {
            println!("  {} {}", style("FAIL").red().bold(), target_label);
        }
        println!();
    }

    Ok(())
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

/// Simple rotating index using a thread-local counter.
fn rand_index(max: usize) -> usize {
    use std::cell::Cell;
    thread_local! {
        static COUNTER: Cell<usize> = const { Cell::new(0) };
    }
    COUNTER.with(|c| {
        let val = c.get();
        c.set(val + 1);
        val % max
    })
}
