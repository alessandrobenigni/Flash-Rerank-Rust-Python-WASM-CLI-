use clap::Args;
use console::style;
use flash_rerank::models::ModelRegistry;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Args)]
pub struct DownloadArgs {
    /// HuggingFace model ID to download.
    #[arg(short, long)]
    pub model: String,

    /// Local cache directory (defaults to HuggingFace Hub cache).
    #[arg(long)]
    pub cache_dir: Option<String>,

    /// Skip confirmation prompt.
    #[arg(short = 'y', long)]
    pub yes: bool,
}

pub async fn run(args: DownloadArgs) -> anyhow::Result<()> {
    let cache_dir = if let Some(dir) = &args.cache_dir {
        std::path::PathBuf::from(dir)
    } else {
        default_cache_dir()
    };

    let registry = ModelRegistry::new(cache_dir);

    println!(
        "{} Downloading model: {}",
        style("=>").cyan().bold(),
        style(&args.model).green()
    );

    // Show a spinner during download (hf-hub manages the actual transfer)
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    pb.set_message(format!("Downloading {}...", &args.model));
    pb.enable_steady_tick(std::time::Duration::from_millis(120));

    let model_path = registry.download(&args.model).await?;

    pb.finish_and_clear();

    // Print summary
    let size_bytes = dir_size(&model_path);
    let size_mb = size_bytes as f64 / 1_048_576.0;

    println!();
    println!("{}", style("Download complete!").green().bold());
    println!("  Model:   {}", args.model);
    println!("  Size:    {:.1} MB", size_mb);
    println!("  Path:    {}", model_path.display());
    println!();
    println!(
        "{}",
        style("Next: flash-rerank compile --model <path> to optimize for your GPU").dim()
    );

    Ok(())
}

fn default_cache_dir() -> std::path::PathBuf {
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

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
    }
    total
}
