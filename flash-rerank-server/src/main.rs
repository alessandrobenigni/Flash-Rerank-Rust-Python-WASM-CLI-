use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use flash_rerank_server::ServerConfig;

#[derive(Parser)]
#[command(name = "flash-rerank-server")]
#[command(about = "Flash-Rerank HTTP inference server")]
struct ServerArgs {
    /// Model to serve (HuggingFace ID or local path).
    #[arg(short, long)]
    model: String,

    /// Host to bind to.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on.
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Maximum batch size for dynamic batching.
    #[arg(long, default_value = "32")]
    max_batch: usize,

    /// Maximum wait time (ms) before dispatching a partial batch.
    #[arg(long, default_value = "10")]
    max_wait_ms: u64,

    /// Comma-separated GPU IDs to use (e.g., "0,1,2").
    #[arg(long)]
    gpus: Option<String>,

    /// OTLP endpoint for OpenTelemetry tracing.
    #[arg(long)]
    otlp_endpoint: Option<String>,

    /// Maximum sequence length for tokenization.
    #[arg(long, default_value = "512")]
    max_length: usize,
}

impl From<ServerArgs> for ServerConfig {
    fn from(args: ServerArgs) -> Self {
        let gpus = args.gpus.map(|s| {
            s.split(',')
                .filter_map(|id| id.trim().parse::<usize>().ok())
                .collect()
        });

        Self {
            model: args.model,
            host: args.host,
            port: args.port,
            max_batch: args.max_batch,
            max_wait_ms: args.max_wait_ms,
            gpus,
            otlp_endpoint: args.otlp_endpoint,
            max_length: args.max_length,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = ServerArgs::parse();
    let config: ServerConfig = args.into();
    flash_rerank_server::start_server(config).await
}
