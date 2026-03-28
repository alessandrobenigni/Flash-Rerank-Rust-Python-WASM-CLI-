//! Two-stage retrieval pipeline example.
//!
//! Demonstrates the conceptual flow of a BM25-Turbo first-stage retrieval
//! followed by Flash-Rerank neural reranking for precision refinement.

use flash_rerank::engine::ort_backend::OrtScorer;
use flash_rerank::engine::Scorer;
use flash_rerank::models::ModelRegistry;
use flash_rerank::ModelConfig;

fn main() -> anyhow::Result<()> {
    // ── Stage 1: BM25 retrieval (simulated) ──────────────────────────
    //
    // In production, this would be a BM25-Turbo index search returning
    // the top-100 candidates. Here we simulate it with a static corpus.
    let query = "How does photosynthesis work?";

    // Simulated BM25 top-10 results (pre-sorted by BM25 score)
    let bm25_candidates = vec![
        "Photosynthesis is the process by which green plants convert sunlight into chemical energy.".to_string(),
        "Plants use chlorophyll to absorb light energy during photosynthesis.".to_string(),
        "The Calvin cycle is the light-independent stage of photosynthesis.".to_string(),
        "Cellular respiration is the reverse process of photosynthesis.".to_string(),
        "Chloroplasts are the organelles where photosynthesis takes place.".to_string(),
        "Solar panels convert sunlight into electricity, similar in concept to photosynthesis.".to_string(),
        "Leaves are green because chlorophyll reflects green wavelengths of light.".to_string(),
        "Carbon dioxide and water are the primary inputs to photosynthesis.".to_string(),
        "The Amazon rainforest produces about 20% of the world's oxygen.".to_string(),
        "ATP and NADPH are produced during the light-dependent reactions.".to_string(),
    ];

    println!("=== Two-Stage Retrieval Pipeline ===\n");
    println!("Query: {query}\n");
    println!("--- Stage 1: BM25 candidates (top {}) ---", bm25_candidates.len());
    for (i, doc) in bm25_candidates.iter().enumerate() {
        println!("  BM25 #{i}: {doc}");
    }

    // ── Stage 2: Neural reranking with Flash-Rerank ──────────────────
    let cache_dir = home_cache_dir();
    let registry = ModelRegistry::new(cache_dir);
    let model_dir = registry.load("cross-encoder/ms-marco-MiniLM-L-6-v2")?;

    let scorer = OrtScorer::new(ModelConfig::default(), &model_dir)?;
    let reranked = scorer.score(query, &bm25_candidates)?;

    println!("\n--- Stage 2: Neural reranked (top 5) ---");
    for result in reranked.iter().take(5) {
        println!(
            "  Rerank #{} (score: {:.4}): {}",
            result.index, result.score, bm25_candidates[result.index]
        );
    }

    println!("\nPipeline complete. Neural reranking reordered BM25 candidates by semantic relevance.");

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
