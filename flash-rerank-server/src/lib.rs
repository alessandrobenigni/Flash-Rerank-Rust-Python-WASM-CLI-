//! Flash-Rerank HTTP inference server library.
//!
//! Provides the server startup function and configuration types that can be
//! used from the CLI or programmatically.

pub mod ab_test;
pub mod canary;
pub mod drift;
pub mod routes;
mod telemetry;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use axum::Router;
use tokio::sync::RwLock;
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use flash_rerank::batch::DynamicBatcher;
use flash_rerank::engine::ort_backend::OrtScorer;
use flash_rerank::multi_gpu::GpuRouter;
use flash_rerank::types::ModelConfig;

use crate::ab_test::AbTestRouter;
use crate::canary::CanaryDeployer;
use crate::drift::DriftDetector;
use crate::routes::{AppState, ServerMetrics};

/// Server configuration that can be constructed from CLI args or programmatically.
pub struct ServerConfig {
    pub model: String,
    pub host: String,
    pub port: u16,
    pub max_batch: usize,
    pub max_wait_ms: u64,
    pub gpus: Option<Vec<usize>>,
    pub otlp_endpoint: Option<String>,
    pub max_length: usize,
}

/// Start the server with the given configuration.
///
/// This function is the main entry point for the server. It loads the model,
/// creates the dynamic batcher, sets up routes, and starts the HTTP server
/// with graceful shutdown support.
pub async fn start_server(config: ServerConfig) -> Result<()> {
    // Initialize telemetry if configured
    telemetry::init_telemetry(config.otlp_endpoint.as_deref())?;

    tracing::info!(
        model = %config.model,
        host = %config.host,
        port = config.port,
        max_batch = config.max_batch,
        max_wait_ms = config.max_wait_ms,
        "Starting Flash-Rerank server"
    );

    // Build model config
    let model_config = ModelConfig {
        model_id: config.model.clone(),
        device: flash_rerank::Device::Cpu,
        precision: flash_rerank::Precision::FP32,
        scorer_type: flash_rerank::ScorerType::CrossEncoder,
        max_length: config.max_length,
    };

    let model_dir = {
        // Try as HuggingFace model ID first
        let cache_dir = default_cache_dir();
        let registry = flash_rerank::models::ModelRegistry::new(cache_dir);
        match registry.load(&config.model) {
            Ok(path) => path,
            Err(_) => {
                // Fall back to direct filesystem path
                let path = PathBuf::from(&config.model);
                if !path.exists() {
                    anyhow::bail!(
                        "Model '{}' not found. Download with: flash-rerank download --model {}",
                        config.model,
                        config.model
                    );
                }
                path
            }
        }
    };
    let max_wait = Duration::from_millis(config.max_wait_ms);

    // Create batcher: multi-GPU path when multiple GPUs specified, single-GPU otherwise
    let gpu_ids = config.gpus.as_deref().unwrap_or(&[]);
    let batcher = if gpu_ids.len() > 1 {
        // Multi-GPU: create a GpuRouter with one inference thread per GPU,
        // then bridge it to the DynamicBatcher via a dispatcher thread.
        tracing::info!(?gpu_ids, "Starting multi-GPU mode");
        let gpu_router = Arc::new(GpuRouter::new(gpu_ids, &model_config, &model_dir)?);

        // Bridge channel: DynamicBatcher sends batches here, dispatcher routes them
        let (bridge_tx, bridge_rx) = std::sync::mpsc::channel();
        let router_clone = gpu_router.clone();
        std::thread::Builder::new()
            .name("gpu-dispatcher".into())
            .spawn(move || {
                while let Ok(batch) = bridge_rx.recv() {
                    if let Err(e) = router_clone.route_batch(batch) {
                        tracing::error!(error = %e, "GPU router dispatch failed");
                    }
                }
                tracing::info!("GPU dispatcher thread exiting");
            })
            .expect("Failed to spawn GPU dispatcher thread");

        Arc::new(DynamicBatcher::with_gpu_sender(
            config.max_batch,
            max_wait,
            bridge_tx,
        ))
    } else {
        // Single GPU (or CPU): create scorer directly with a single batcher thread
        let scorer = OrtScorer::new(model_config, &model_dir)?;
        Arc::new(DynamicBatcher::new(
            config.max_batch,
            max_wait,
            Box::new(scorer),
        ))
    };

    // Create drift detector (warm up after 1000 samples, threshold 0.5)
    let drift_detector = Arc::new(RwLock::new(DriftDetector::new(0.5, 1000)));

    // Create A/B router (50/50 default, inactive until configured)
    let ab_router = Arc::new(AbTestRouter::new(50, 50));

    // Create canary deployer (inactive until started via endpoint)
    let canary = Arc::new(RwLock::new(CanaryDeployer::new()));

    // Build application state
    let state = Arc::new(AppState {
        batcher,
        model_name: config.model.clone(),
        start_time: Instant::now(),
        drift_detector,
        ab_router,
        canary,
        metrics: Arc::new(RwLock::new(ServerMetrics::default())),
    });

    // Build Axum app with middleware stack
    let app = Router::new()
        .merge(routes::router(state))
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive());

    let bind_addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    tracing::info!(
        "Flash-Rerank server listening on {}",
        listener.local_addr()?
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Server shut down gracefully");
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

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C handler");
    tracing::info!("Shutdown signal received");
}
