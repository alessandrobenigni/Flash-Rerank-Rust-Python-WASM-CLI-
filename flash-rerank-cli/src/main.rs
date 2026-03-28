mod bench;
mod compile;
mod download;
mod models;
mod serve;

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::Shell;
use console::style;

#[derive(Parser)]
#[command(name = "flash-rerank")]
#[command(about = "Flash-Rerank CLI — compile, benchmark, serve, and manage reranking models")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output (info level logging).
    #[arg(long, global = true)]
    verbose: bool,

    /// Enable debug output (trace level logging).
    #[arg(long, global = true)]
    debug: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile an ONNX model to a TensorRT engine.
    Compile(compile::CompileArgs),
    /// Run benchmarks against a model.
    Bench(bench::BenchArgs),
    /// Start the HTTP server.
    Serve(serve::ServeArgs),
    /// Download a model from HuggingFace Hub.
    Download(download::DownloadArgs),
    /// Manage cached models (list, inspect, delete).
    Models(models::ModelsArgs),
    /// Generate shell completions.
    Completions {
        /// Shell to generate completions for.
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Configure tracing subscriber based on verbosity flags
    let filter = if cli.debug {
        "trace"
    } else if cli.verbose {
        "info"
    } else {
        "warn"
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .init();

    let result = match cli.command {
        Commands::Compile(args) => compile::run(args).await,
        Commands::Bench(args) => bench::run(args).await,
        Commands::Serve(args) => serve::run(args).await,
        Commands::Download(args) => download::run(args).await,
        Commands::Models(args) => models::run(args).await,
        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            clap_complete::generate(shell, &mut cmd, name, &mut std::io::stdout());
            Ok(())
        }
    };

    if let Err(err) = result {
        eprintln!("{} {err:#}", style("error:").red().bold());
        std::process::exit(1);
    }
}
