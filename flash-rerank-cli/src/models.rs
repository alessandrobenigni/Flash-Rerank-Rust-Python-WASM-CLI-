use clap::{Args, Subcommand};
use console::style;
use flash_rerank::models::ModelRegistry;

#[derive(Args)]
pub struct ModelsArgs {
    #[command(subcommand)]
    pub command: ModelsCommand,
}

#[derive(Subcommand)]
pub enum ModelsCommand {
    /// List all cached models.
    List,
    /// Show details for a cached model.
    Inspect {
        /// HuggingFace model ID.
        #[arg(short, long)]
        model: String,
    },
    /// Delete a cached model.
    Delete {
        /// HuggingFace model ID.
        #[arg(short, long)]
        model: String,
    },
}

pub async fn run(args: ModelsArgs) -> anyhow::Result<()> {
    let cache_dir = default_cache_dir();
    let registry = ModelRegistry::new(cache_dir);

    match args.command {
        ModelsCommand::List => {
            let models = registry.list()?;
            if models.is_empty() {
                println!(
                    "{} No models cached. Download one with: flash-rerank download --model <model_id>",
                    style("(empty)").dim()
                );
            } else {
                println!("{}", style("Cached models:").bold());
                for meta in &models {
                    let size_mb = meta.file_size_bytes as f64 / 1_048_576.0;
                    println!(
                        "  {} {} ({:.1} MB, downloaded {})",
                        style("*").cyan(),
                        style(&meta.model_id).green(),
                        size_mb,
                        meta.download_date,
                    );
                }
                println!("\n{} model(s) cached.", models.len());
            }
        }
        ModelsCommand::Inspect { model } => {
            // Try to load from cache to verify existence
            let path = registry.load(&model)?;
            let meta_path = path.join("flash_rerank_cache.json");
            if meta_path.exists() {
                let content = std::fs::read_to_string(&meta_path)?;
                let meta: flash_rerank::CacheMetadata = serde_json::from_str(&content)?;
                println!("{}", style("Model details:").bold());
                println!("  Model ID:      {}", meta.model_id);
                println!("  Source:        {}", meta.source_url);
                println!("  Downloaded:    {}", meta.download_date);
                println!(
                    "  Size:          {:.1} MB",
                    meta.file_size_bytes as f64 / 1_048_576.0
                );
                println!("  Path:          {}", path.display());
            } else {
                println!("Model found at {} (no metadata file)", path.display());
            }
        }
        ModelsCommand::Delete { model } => {
            registry.delete(&model)?;
            println!("{} Model '{}' deleted.", style("OK").green().bold(), model);
        }
    }

    Ok(())
}

fn default_cache_dir() -> std::path::PathBuf {
    // Use HuggingFace Hub default cache location
    if let Ok(cache) = std::env::var("HF_HOME") {
        return std::path::PathBuf::from(cache);
    }
    if let Some(home) = dirs_fallback() {
        return home.join(".cache").join("huggingface").join("hub");
    }
    std::path::PathBuf::from(".cache/huggingface/hub")
}

fn dirs_fallback() -> Option<std::path::PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(std::path::PathBuf::from)
}
