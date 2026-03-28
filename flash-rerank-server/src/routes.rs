use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use flash_rerank::batch::DynamicBatcher;

use crate::ab_test::{AbTestConfig, AbTestRouter, Variant};
use crate::canary::CanaryDeployer;
use crate::drift::DriftDetector;

/// Shared application state passed to all route handlers.
pub struct AppState {
    pub batcher: Arc<DynamicBatcher>,
    pub model_name: String,
    pub start_time: Instant,
    pub drift_detector: Arc<RwLock<DriftDetector>>,
    pub ab_router: Arc<AbTestRouter>,
    pub canary: Arc<RwLock<CanaryDeployer>>,
    pub metrics: Arc<RwLock<ServerMetrics>>,
}

/// Server-level metrics for Prometheus export.
pub struct ServerMetrics {
    pub requests_total: u64,
    pub errors_total: u64,
    pub total_latency_seconds: f64,
    pub batch_sizes: Vec<usize>,
}

impl Default for ServerMetrics {
    fn default() -> Self {
        Self {
            requests_total: 0,
            errors_total: 0,
            total_latency_seconds: 0.0,
            batch_sizes: Vec::new(),
        }
    }
}

/// Build the application router with all routes and management endpoints.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        // Core endpoints
        .route("/rerank", post(rerank))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        // Admin management endpoints
        .route("/admin/status", get(admin_status))
        .route("/admin/drift/status", get(drift_status))
        .route("/admin/drift/reset-baseline", post(drift_reset_baseline))
        .route("/admin/ab", post(ab_create))
        .route("/admin/ab", put(ab_adjust))
        .route("/admin/ab", delete(ab_stop))
        .route("/admin/ab/metrics", get(ab_metrics))
        // Canary management endpoints
        .route("/admin/canary", post(canary_start))
        .route("/admin/canary", delete(canary_abort))
        .route("/admin/canary/status", get(canary_status))
        .route("/admin/canary/advance", post(canary_advance))
        .with_state(state)
}

// --- Request/Response types ---

#[derive(Debug, Deserialize)]
pub struct RerankPayload {
    pub query: String,
    pub documents: Vec<String>,
    pub top_k: Option<usize>,
    #[serde(default)]
    pub request_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RerankResponse {
    pub results: Vec<ScoredDoc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    pub latency_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct ScoredDoc {
    pub index: usize,
    pub score: f32,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub model: String,
    pub uptime_seconds: f64,
}

#[derive(Debug, Serialize)]
pub struct AdminStatusResponse {
    pub status: &'static str,
    pub model: String,
    pub uptime_seconds: f64,
    pub total_requests: u64,
    pub total_errors: u64,
    pub ab_test_active: bool,
}

// --- Core route handlers ---

async fn rerank(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RerankPayload>,
) -> std::result::Result<Json<RerankResponse>, (StatusCode, String)> {
    let start = Instant::now();

    // Determine A/B variant
    let variant = if state.ab_router.is_active() {
        Some(state.ab_router.route(payload.request_id.as_deref()))
    } else {
        None
    };

    // Submit to batcher
    let results = state
        .batcher
        .submit(payload.query, payload.documents, payload.top_k)
        .await
        .map_err(|e| {
            let _ = state.metrics.try_write().map(|mut m| m.errors_total += 1);
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Record drift scores
    {
        let mut drift = state.drift_detector.write().await;
        for result in &results {
            drift.record_score(result.score);
        }
    }

    // Record A/B metrics
    if let Some(v) = variant {
        state.ab_router.record(v, latency_ms);
    }

    // Update server metrics
    {
        let mut m = state.metrics.write().await;
        m.requests_total += 1;
        m.total_latency_seconds += latency_ms / 1000.0;
        m.batch_sizes.push(results.len());
    }

    let scored_docs: Vec<ScoredDoc> = results
        .into_iter()
        .map(|r| ScoredDoc {
            index: r.index,
            score: r.score,
        })
        .collect();

    Ok(Json(RerankResponse {
        results: scored_docs,
        variant: variant.map(|v| v.to_string()),
        latency_ms,
    }))
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        model: state.model_name.clone(),
        uptime_seconds: state.start_time.elapsed().as_secs_f64(),
    })
}

async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let m = state.metrics.read().await;
    let uptime = state.start_time.elapsed().as_secs_f64();

    let avg_latency = if m.requests_total > 0 {
        m.total_latency_seconds / m.requests_total as f64
    } else {
        0.0
    };

    let avg_batch = if m.batch_sizes.is_empty() {
        0.0
    } else {
        m.batch_sizes.iter().sum::<usize>() as f64 / m.batch_sizes.len() as f64
    };

    // Prometheus text format
    let body = format!(
        "# HELP flash_rerank_requests_total Total number of rerank requests.\n\
         # TYPE flash_rerank_requests_total counter\n\
         flash_rerank_requests_total {}\n\
         # HELP flash_rerank_errors_total Total number of errors.\n\
         # TYPE flash_rerank_errors_total counter\n\
         flash_rerank_errors_total {}\n\
         # HELP flash_rerank_latency_seconds Average request latency in seconds.\n\
         # TYPE flash_rerank_latency_seconds gauge\n\
         flash_rerank_latency_seconds {avg_latency:.6}\n\
         # HELP flash_rerank_batch_size Average batch size.\n\
         # TYPE flash_rerank_batch_size gauge\n\
         flash_rerank_batch_size {avg_batch:.2}\n\
         # HELP flash_rerank_uptime_seconds Server uptime in seconds.\n\
         # TYPE flash_rerank_uptime_seconds gauge\n\
         flash_rerank_uptime_seconds {uptime:.2}\n",
        m.requests_total, m.errors_total,
    );

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

// --- Admin endpoints ---

async fn admin_status(State(state): State<Arc<AppState>>) -> Json<AdminStatusResponse> {
    let m = state.metrics.read().await;
    Json(AdminStatusResponse {
        status: "ok",
        model: state.model_name.clone(),
        uptime_seconds: state.start_time.elapsed().as_secs_f64(),
        total_requests: m.requests_total,
        total_errors: m.errors_total,
        ab_test_active: state.ab_router.is_active(),
    })
}

// --- Drift detection endpoints ---

async fn drift_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let drift = state.drift_detector.read().await;
    let summary = drift.status_summary();
    Json(serde_json::to_value(summary).unwrap_or_default())
}

async fn drift_reset_baseline(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mut drift = state.drift_detector.write().await;
    drift.reset_baseline();
    Json(serde_json::json!({ "status": "baseline_reset" }))
}

// --- A/B test management endpoints ---

async fn ab_create(
    State(state): State<Arc<AppState>>,
    Json(config): Json<AbTestConfig>,
) -> Json<serde_json::Value> {
    state.ab_router.configure(config);
    Json(serde_json::json!({ "status": "ab_test_created" }))
}

#[derive(Deserialize)]
struct AbAdjustPayload {
    variant_a_weight: u32,
    variant_b_weight: u32,
}

async fn ab_adjust(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<AbAdjustPayload>,
) -> Json<serde_json::Value> {
    state
        .ab_router
        .update_split(payload.variant_a_weight, payload.variant_b_weight);
    Json(serde_json::json!({ "status": "split_adjusted" }))
}

#[derive(Deserialize)]
struct AbStopQuery {
    retain: Option<String>,
}

async fn ab_stop(
    State(state): State<Arc<AppState>>,
    Query(query): Query<AbStopQuery>,
) -> Json<serde_json::Value> {
    let retain = match query.retain.as_deref() {
        Some("b") | Some("B") => Variant::B,
        _ => Variant::A,
    };
    state.ab_router.stop(retain);
    Json(serde_json::json!({ "status": "ab_test_stopped", "retained": retain.to_string() }))
}

async fn ab_metrics(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let metrics = state.ab_router.metrics();
    Json(serde_json::to_value(metrics).unwrap_or_default())
}

// --- Canary management endpoints ---

#[derive(Deserialize)]
struct CanaryStartPayload {
    canary_model: String,
    stages: Option<Vec<u8>>,
    observation_secs: Option<u64>,
    tolerance: Option<f64>,
}

async fn canary_start(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CanaryStartPayload>,
) -> Json<serde_json::Value> {
    let mut canary = state.canary.write().await;
    canary.start(
        payload.canary_model,
        payload.stages,
        payload.observation_secs,
        payload.tolerance,
    );
    Json(serde_json::json!({ "status": "canary_started" }))
}

async fn canary_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let canary = state.canary.read().await;
    let status = canary.status();
    Json(serde_json::to_value(status).unwrap_or_default())
}

async fn canary_abort(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mut canary = state.canary.write().await;
    canary.abort();
    Json(serde_json::json!({ "status": "canary_aborted" }))
}

async fn canary_advance(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mut canary = state.canary.write().await;
    let action = canary.manual_advance();
    Json(serde_json::json!({
        "status": "advanced",
        "action": format!("{:?}", action),
        "current_percentage": canary.canary_percentage(),
    }))
}
