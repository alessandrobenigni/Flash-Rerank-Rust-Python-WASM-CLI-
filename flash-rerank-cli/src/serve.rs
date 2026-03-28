use clap::Args;

#[derive(Args)]
pub struct ServeArgs {
    /// Host to bind to.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on.
    #[arg(short, long, default_value = "8080")]
    pub port: u16,

    /// Model to serve (HuggingFace ID or local path).
    #[arg(short, long)]
    pub model: String,

    /// Maximum batch size for dynamic batching.
    #[arg(long, default_value = "32")]
    pub max_batch: usize,

    /// Maximum wait time (ms) before dispatching a partial batch.
    #[arg(long, default_value = "10")]
    pub max_wait_ms: u64,

    /// Comma-separated GPU IDs to use (e.g., "0,1,2").
    #[arg(long)]
    pub gpus: Option<String>,

    /// OTLP endpoint for OpenTelemetry tracing.
    #[arg(long)]
    pub otlp_endpoint: Option<String>,

    /// Maximum sequence length for tokenization.
    #[arg(long, default_value = "512")]
    pub max_length: usize,
}

pub async fn run(args: ServeArgs) -> anyhow::Result<()> {
    tracing::info!(
        model = %args.model,
        host = %args.host,
        port = args.port,
        max_batch = args.max_batch,
        max_wait_ms = args.max_wait_ms,
        "Starting Flash-Rerank server via CLI"
    );

    let gpus = args.gpus.map(|s| {
        s.split(',')
            .filter_map(|id| id.trim().parse::<usize>().ok())
            .collect()
    });

    let config = flash_rerank_server::ServerConfig {
        model: args.model,
        host: args.host,
        port: args.port,
        max_batch: args.max_batch,
        max_wait_ms: args.max_wait_ms,
        gpus,
        otlp_endpoint: args.otlp_endpoint,
        max_length: args.max_length,
    };

    flash_rerank_server::start_server(config).await
}
