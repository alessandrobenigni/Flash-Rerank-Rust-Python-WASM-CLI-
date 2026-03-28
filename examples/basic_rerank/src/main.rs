//! Basic reranking example using Flash-Rerank.
//!
//! Downloads (or loads from cache) a cross-encoder model and reranks
//! a set of documents against a query, printing scored results.

use flash_rerank::engine::ort_backend::OrtScorer;
use flash_rerank::engine::Scorer;
use flash_rerank::models::ModelRegistry;
use flash_rerank::ModelConfig;

fn main() -> anyhow::Result<()> {
    // Load model from HuggingFace cache
    let cache_dir = home_cache_dir();
    let registry = ModelRegistry::new(cache_dir);
    let model_dir = registry.load("cross-encoder/ms-marco-MiniLM-L-6-v2")?;

    let scorer = OrtScorer::new(ModelConfig::default(), &model_dir)?;

    let query = "What is the capital of France?";
    let documents = vec![
        "Paris is the capital of France.".to_string(),
        "Berlin is the capital of Germany.".to_string(),
        "London is the capital of England.".to_string(),
        "The Eiffel Tower is in Paris.".to_string(),
        "France is a country in Europe.".to_string(),
    ];

    let results = scorer.score(query, &documents)?;

    println!("Query: {query}\n");
    for result in &results {
        println!(
            "  #{} (score: {:.4}): {}",
            result.index, result.score, documents[result.index]
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
