use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

/// Variant identifier for A/B testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Variant {
    A,
    B,
}

impl std::fmt::Display for Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variant::A => write!(f, "a"),
            Variant::B => write!(f, "b"),
        }
    }
}

/// Per-variant metrics for A/B test analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantMetrics {
    pub request_count: u64,
    pub total_latency_ms: f64,
    pub mean_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub error_count: u64,
}

impl Default for VariantMetrics {
    fn default() -> Self {
        Self {
            request_count: 0,
            total_latency_ms: 0.0,
            mean_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            error_count: 0,
        }
    }
}

/// Internal mutable metrics state.
struct VariantState {
    request_count: u64,
    total_latency_ms: f64,
    error_count: u64,
    latency_samples: Vec<f64>,
}

impl Default for VariantState {
    fn default() -> Self {
        Self {
            request_count: 0,
            total_latency_ms: 0.0,
            error_count: 0,
            latency_samples: Vec::with_capacity(1024),
        }
    }
}

impl VariantState {
    fn to_metrics(&self) -> VariantMetrics {
        let mean = if self.request_count > 0 {
            self.total_latency_ms / self.request_count as f64
        } else {
            0.0
        };

        let (p50, p95, p99) = if self.latency_samples.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let mut sorted = self.latency_samples.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let len = sorted.len();
            (
                sorted[len * 50 / 100],
                sorted[(len * 95 / 100).min(len - 1)],
                sorted[(len * 99 / 100).min(len - 1)],
            )
        };

        VariantMetrics {
            request_count: self.request_count,
            total_latency_ms: self.total_latency_ms,
            mean_latency_ms: mean,
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            error_count: self.error_count,
        }
    }
}

/// A/B test router for comparing model variants in production.
///
/// Routes requests to variant A or B based on a configurable split ratio.
/// Supports deterministic routing via request ID hashing for consistency,
/// and weighted round-robin as fallback.
pub struct AbTestRouter {
    split_ratio: RwLock<(u32, u32)>,
    counter: AtomicU64,
    metrics_a: RwLock<VariantState>,
    metrics_b: RwLock<VariantState>,
    active: RwLock<bool>,
    test_name: RwLock<String>,
}

/// Configuration for creating or adjusting an A/B test.
#[derive(Debug, Clone, Deserialize)]
pub struct AbTestConfig {
    pub name: String,
    pub variant_a_weight: u32,
    pub variant_b_weight: u32,
}

/// Response for A/B test metrics endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct AbTestMetricsResponse {
    pub name: String,
    pub active: bool,
    pub split: (u32, u32),
    pub variant_a: VariantMetrics,
    pub variant_b: VariantMetrics,
}

impl AbTestRouter {
    /// Create a new A/B test router with the given split ratio.
    ///
    /// Split ratio is (a_weight, b_weight). For example, (80, 20) means
    /// 80% of traffic goes to variant A and 20% to variant B.
    pub fn new(a_weight: u32, b_weight: u32) -> Self {
        Self {
            split_ratio: RwLock::new((a_weight, b_weight)),
            counter: AtomicU64::new(0),
            metrics_a: RwLock::new(VariantState::default()),
            metrics_b: RwLock::new(VariantState::default()),
            active: RwLock::new(true),
            test_name: RwLock::new("default".to_string()),
        }
    }

    /// Route a request to variant A or B.
    ///
    /// If `request_id` is provided, routing is deterministic (same ID always
    /// routes to the same variant). Otherwise, uses weighted round-robin.
    pub fn route(&self, request_id: Option<&str>) -> Variant {
        let (a_weight, b_weight) = *self.split_ratio.read().unwrap();
        let total = (a_weight + b_weight) as u64;

        if total == 0 {
            return Variant::A;
        }

        let value = match request_id {
            Some(id) => {
                let mut hasher = DefaultHasher::new();
                id.hash(&mut hasher);
                hasher.finish()
            }
            None => self.counter.fetch_add(1, Ordering::Relaxed),
        };

        if (value % total) < a_weight as u64 {
            Variant::A
        } else {
            Variant::B
        }
    }

    /// Update the traffic split ratio.
    pub fn update_split(&self, a_weight: u32, b_weight: u32) {
        *self.split_ratio.write().unwrap() = (a_weight, b_weight);
        tracing::info!(a_weight, b_weight, "A/B test split updated");
    }

    /// Record a completed request for the given variant.
    pub fn record(&self, variant: Variant, latency_ms: f64) {
        let state = match variant {
            Variant::A => &self.metrics_a,
            Variant::B => &self.metrics_b,
        };
        let mut s = state.write().unwrap();
        s.request_count += 1;
        s.total_latency_ms += latency_ms;
        s.latency_samples.push(latency_ms);
    }

    /// Record an error for the given variant.
    pub fn record_error(&self, variant: Variant) {
        let state = match variant {
            Variant::A => &self.metrics_a,
            Variant::B => &self.metrics_b,
        };
        let mut s = state.write().unwrap();
        s.error_count += 1;
    }

    /// Get metrics for both variants.
    pub fn metrics(&self) -> AbTestMetricsResponse {
        let name = self.test_name.read().unwrap().clone();
        let active = *self.active.read().unwrap();
        let split = *self.split_ratio.read().unwrap();

        AbTestMetricsResponse {
            name,
            active,
            split,
            variant_a: self.metrics_a.read().unwrap().to_metrics(),
            variant_b: self.metrics_b.read().unwrap().to_metrics(),
        }
    }

    /// Configure the A/B test with new settings.
    pub fn configure(&self, config: AbTestConfig) {
        *self.test_name.write().unwrap() = config.name.clone();
        self.update_split(config.variant_a_weight, config.variant_b_weight);
        *self.active.write().unwrap() = true;
        tracing::info!(name = config.name, "A/B test configured");
    }

    /// Stop the A/B test, routing all traffic to the retained variant.
    pub fn stop(&self, retain: Variant) {
        *self.active.write().unwrap() = false;
        match retain {
            Variant::A => self.update_split(1, 0),
            Variant::B => self.update_split(0, 1),
        }
        tracing::info!(?retain, "A/B test stopped");
    }

    /// Whether the A/B test is active.
    pub fn is_active(&self) -> bool {
        *self.active.read().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_routing_same_request_id() {
        let router = AbTestRouter::new(50, 50);
        let id = "test-request-123";

        let variant = router.route(Some(id));
        for _ in 0..100 {
            assert_eq!(router.route(Some(id)), variant);
        }
    }

    #[test]
    fn test_split_ratio_distribution() {
        let router = AbTestRouter::new(70, 30);
        let mut a_count = 0u32;

        for _ in 0..1000 {
            match router.route(None) {
                Variant::A => a_count += 1,
                Variant::B => {}
            }
        }

        // With 70/30 split, A should get ~700 requests
        let a_ratio = a_count as f64 / 1000.0;
        assert!(
            (a_ratio - 0.7).abs() < 0.05,
            "A ratio {a_ratio} not close to 0.7"
        );
    }

    #[test]
    fn test_update_split() {
        let router = AbTestRouter::new(50, 50);
        router.update_split(90, 10);
        let (a, b) = *router.split_ratio.read().unwrap();
        assert_eq!(a, 90);
        assert_eq!(b, 10);
    }

    #[test]
    fn test_record_metrics() {
        let router = AbTestRouter::new(50, 50);
        router.record(Variant::A, 10.0);
        router.record(Variant::A, 20.0);
        router.record(Variant::B, 15.0);

        let metrics = router.metrics();
        assert_eq!(metrics.variant_a.request_count, 2);
        assert_eq!(metrics.variant_b.request_count, 1);
        assert!((metrics.variant_a.mean_latency_ms - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_stop_retains_variant() {
        let router = AbTestRouter::new(50, 50);
        router.stop(Variant::A);

        assert!(!router.is_active());
        // All requests should now go to A
        for _ in 0..100 {
            assert_eq!(router.route(None), Variant::A);
        }
    }

    // --- Additional A/B test tests ---

    #[test]
    fn test_variant_display() {
        assert_eq!(format!("{}", Variant::A), "a");
        assert_eq!(format!("{}", Variant::B), "b");
    }

    #[test]
    fn test_variant_serde_roundtrip() {
        let a = Variant::A;
        let json = serde_json::to_string(&a).unwrap();
        let back: Variant = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }

    #[test]
    fn test_zero_total_weight_routes_to_a() {
        let router = AbTestRouter::new(0, 0);
        for _ in 0..10 {
            assert_eq!(
                router.route(None),
                Variant::A,
                "Zero total weight should default to A"
            );
        }
    }

    #[test]
    fn test_100_percent_a() {
        let router = AbTestRouter::new(100, 0);
        for _ in 0..100 {
            assert_eq!(router.route(None), Variant::A);
        }
    }

    #[test]
    fn test_100_percent_b() {
        let router = AbTestRouter::new(0, 100);
        for _ in 0..100 {
            assert_eq!(router.route(None), Variant::B);
        }
    }

    #[test]
    fn test_record_error() {
        let router = AbTestRouter::new(50, 50);
        router.record_error(Variant::A);
        router.record_error(Variant::A);
        router.record_error(Variant::B);

        let metrics = router.metrics();
        assert_eq!(metrics.variant_a.error_count, 2);
        assert_eq!(metrics.variant_b.error_count, 1);
    }

    #[test]
    fn test_configure_updates_name_and_split() {
        let router = AbTestRouter::new(50, 50);
        router.configure(AbTestConfig {
            name: "experiment-1".to_string(),
            variant_a_weight: 90,
            variant_b_weight: 10,
        });
        let metrics = router.metrics();
        assert_eq!(metrics.name, "experiment-1");
        assert_eq!(metrics.split, (90, 10));
        assert!(metrics.active);
    }

    #[test]
    fn test_stop_retains_variant_b() {
        let router = AbTestRouter::new(50, 50);
        router.stop(Variant::B);

        assert!(!router.is_active());
        for _ in 0..100 {
            assert_eq!(router.route(None), Variant::B);
        }
    }

    #[test]
    fn test_metrics_percentiles_with_samples() {
        let router = AbTestRouter::new(50, 50);
        // Record 100 latencies for variant A
        for i in 1..=100 {
            router.record(Variant::A, i as f64);
        }

        let metrics = router.metrics();
        assert_eq!(metrics.variant_a.request_count, 100);
        assert!(metrics.variant_a.p50_latency_ms > 0.0);
        assert!(metrics.variant_a.p95_latency_ms > metrics.variant_a.p50_latency_ms);
        assert!(metrics.variant_a.p99_latency_ms >= metrics.variant_a.p95_latency_ms);
    }

    #[test]
    fn test_variant_metrics_default() {
        let m = VariantMetrics::default();
        assert_eq!(m.request_count, 0);
        assert_eq!(m.total_latency_ms, 0.0);
        assert_eq!(m.mean_latency_ms, 0.0);
        assert_eq!(m.error_count, 0);
    }
}
