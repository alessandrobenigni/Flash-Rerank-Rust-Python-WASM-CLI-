use serde::Serialize;

const NUM_BINS: usize = 100;

/// Score drift detector using exponentially weighted histograms and KL divergence.
///
/// Monitors the distribution of reranking scores over time. When the current
/// distribution diverges significantly from the baseline (established during
/// warm-up), alerts are raised with escalation from Warning to Critical.
pub struct DriftDetector {
    baseline: Option<Histogram>,
    current: Histogram,
    threshold: f64,
    consecutive_alerts: usize,
    warm_up_remaining: usize,
    min_sample_size: usize,
    decay_factor: f64,
}

/// Internal histogram with exponential decay.
#[derive(Clone)]
struct Histogram {
    bins: Vec<f64>,
    count: usize,
}

impl Histogram {
    fn new() -> Self {
        Self {
            bins: vec![0.0; NUM_BINS],
            count: 0,
        }
    }

    /// Normalize bins to form a probability distribution.
    fn normalized(&self) -> Vec<f64> {
        let total: f64 = self.bins.iter().sum();
        if total == 0.0 {
            return vec![1.0 / NUM_BINS as f64; NUM_BINS];
        }
        self.bins.iter().map(|&b| b / total).collect()
    }

    /// Apply exponential decay to all bins.
    fn decay(&mut self, factor: f64) {
        for bin in &mut self.bins {
            *bin *= factor;
        }
    }
}

/// Status of drift detection.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum DriftStatus {
    /// Score distribution is stable (within threshold).
    Stable,
    /// Score distribution has drifted (KL divergence above threshold).
    Warning(f64),
    /// Score distribution drift has persisted (5+ consecutive alerts).
    Critical(f64),
    /// Score distribution has recovered after a period of drift.
    Recovered,
    /// Not enough data to evaluate drift.
    InsufficientData,
}

impl DriftDetector {
    /// Create a new DriftDetector.
    ///
    /// # Arguments
    /// * `threshold` - KL divergence threshold for triggering a warning
    /// * `warm_up` - Number of samples to collect before establishing baseline
    pub fn new(threshold: f64, warm_up: usize) -> Self {
        Self {
            baseline: None,
            current: Histogram::new(),
            threshold,
            consecutive_alerts: 0,
            warm_up_remaining: warm_up,
            min_sample_size: 50,
            decay_factor: 0.99,
        }
    }

    /// Record a score observation, updating the current histogram.
    ///
    /// During warm-up, the baseline is established from the first `warm_up` samples.
    /// After warm-up, scores are recorded with exponential decay.
    pub fn record_score(&mut self, score: f32) {
        // Clamp score to [0.0, 1.0] range
        let clamped = score.clamp(0.0, 1.0);
        let bin = ((clamped * NUM_BINS as f32) as usize).min(NUM_BINS - 1);

        // Apply decay before adding new observation
        if self.current.count > 0 {
            self.current.decay(self.decay_factor);
        }
        self.current.bins[bin] += 1.0;
        self.current.count += 1;

        if self.warm_up_remaining > 0 {
            self.warm_up_remaining -= 1;
            if self.warm_up_remaining == 0 {
                self.baseline = Some(self.current.clone());
                tracing::info!(
                    samples = self.current.count,
                    "Drift detection baseline established"
                );
            }
        }
    }

    /// Record a batch of scores.
    pub fn record_scores(&mut self, scores: &[f32]) {
        for &score in scores {
            self.record_score(score);
        }
    }

    /// Evaluate the current drift status.
    ///
    /// Returns:
    /// - `InsufficientData` if baseline is not yet established or not enough samples
    /// - `Stable` if KL divergence is below threshold
    /// - `Warning` if KL divergence exceeds threshold (< 5 consecutive)
    /// - `Critical` if KL divergence exceeds threshold (>= 5 consecutive)
    /// - `Recovered` if previously drifted but now stable
    pub fn check(&mut self) -> DriftStatus {
        let baseline = match &self.baseline {
            Some(b) => b,
            None => return DriftStatus::InsufficientData,
        };

        if self.current.count < self.min_sample_size {
            return DriftStatus::InsufficientData;
        }

        let kl = kl_divergence(&self.current, baseline);

        if kl > self.threshold {
            self.consecutive_alerts += 1;
            if self.consecutive_alerts >= 5 {
                DriftStatus::Critical(kl)
            } else {
                DriftStatus::Warning(kl)
            }
        } else if self.consecutive_alerts > 0 {
            self.consecutive_alerts = 0;
            DriftStatus::Recovered
        } else {
            DriftStatus::Stable
        }
    }

    /// Reset the baseline to the current distribution.
    pub fn reset_baseline(&mut self) {
        self.baseline = Some(self.current.clone());
        self.consecutive_alerts = 0;
        tracing::info!("Drift detection baseline reset");
    }

    /// Get the current drift status summary.
    pub fn status_summary(&self) -> DriftStatusSummary {
        let baseline_established = self.baseline.is_some();
        let kl = if let Some(ref baseline) = self.baseline {
            Some(kl_divergence(&self.current, baseline))
        } else {
            None
        };

        DriftStatusSummary {
            baseline_established,
            current_samples: self.current.count,
            warm_up_remaining: self.warm_up_remaining,
            consecutive_alerts: self.consecutive_alerts,
            kl_divergence: kl,
            threshold: self.threshold,
        }
    }

    /// Get the consecutive alert count.
    pub fn consecutive_alerts(&self) -> usize {
        self.consecutive_alerts
    }
}

/// Summary of drift detector status for reporting.
#[derive(Debug, Clone, Serialize)]
pub struct DriftStatusSummary {
    pub baseline_established: bool,
    pub current_samples: usize,
    pub warm_up_remaining: usize,
    pub consecutive_alerts: usize,
    pub kl_divergence: Option<f64>,
    pub threshold: f64,
}

/// Compute KL divergence: KL(current || baseline).
///
/// Uses Laplace smoothing to avoid division by zero.
fn kl_divergence(current: &Histogram, baseline: &Histogram) -> f64 {
    let p = current.normalized();
    let q = baseline.normalized();

    let epsilon = 1e-10;
    let mut kl = 0.0;
    for i in 0..NUM_BINS {
        let pi = p[i] + epsilon;
        let qi = q[i] + epsilon;
        kl += pi * (pi / qi).ln();
    }

    kl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warm_up_establishes_baseline() {
        let mut detector = DriftDetector::new(0.1, 10);
        for _ in 0..9 {
            detector.record_score(0.5);
        }
        assert!(detector.baseline.is_none());
        detector.record_score(0.5);
        assert!(detector.baseline.is_some());
    }

    #[test]
    fn test_stable_distribution() {
        let mut detector = DriftDetector::new(0.5, 100);
        // Warm up with uniform-ish scores
        for i in 0..100 {
            detector.record_score((i as f32) / 100.0);
        }
        // Continue with similar distribution
        for i in 0..100 {
            detector.record_score((i as f32) / 100.0);
        }
        let status = detector.check();
        assert!(
            matches!(status, DriftStatus::Stable),
            "Expected Stable, got {status:?}"
        );
    }

    #[test]
    fn test_drift_warning_and_critical() {
        let mut detector = DriftDetector::new(0.1, 100);
        detector.min_sample_size = 10;

        // Warm up with scores around 0.4-0.6
        for _ in 0..100 {
            detector.record_score(0.5);
        }

        // Shift distribution to 0.9-1.0
        for _ in 0..200 {
            detector.record_score(0.95);
        }

        // Check for warnings escalating to critical
        for i in 0..5 {
            let status = detector.check();
            if i < 4 {
                assert!(
                    matches!(status, DriftStatus::Warning(_)),
                    "Expected Warning at check {i}, got {status:?}"
                );
            } else {
                assert!(
                    matches!(status, DriftStatus::Critical(_)),
                    "Expected Critical at check {i}, got {status:?}"
                );
            }
        }
    }

    #[test]
    fn test_recovery() {
        let mut detector = DriftDetector::new(0.1, 50);
        detector.min_sample_size = 10;

        // Warm up
        for _ in 0..50 {
            detector.record_score(0.5);
        }

        // Cause drift
        for _ in 0..100 {
            detector.record_score(0.95);
        }
        let status = detector.check();
        assert!(matches!(status, DriftStatus::Warning(_)));

        // Return to baseline-like distribution
        // Reset baseline to simulate recovery
        detector.reset_baseline();
        let status = detector.check();
        assert!(
            matches!(status, DriftStatus::Stable),
            "Expected Stable after reset, got {status:?}"
        );
    }

    #[test]
    fn test_insufficient_data() {
        let mut detector = DriftDetector::new(0.1, 1000);
        detector.record_score(0.5);
        assert_eq!(detector.check(), DriftStatus::InsufficientData);
    }

    #[test]
    fn test_score_clamping() {
        let mut detector = DriftDetector::new(0.5, 5);
        // Scores outside [0, 1] should be clamped
        detector.record_score(-0.5);
        detector.record_score(1.5);
        detector.record_score(0.5);
        detector.record_score(0.5);
        detector.record_score(0.5);
        assert!(detector.baseline.is_some());
    }

    // --- Additional drift tests ---

    #[test]
    fn test_record_scores_batch() {
        let mut detector = DriftDetector::new(0.5, 5);
        detector.record_scores(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        assert!(
            detector.baseline.is_some(),
            "Batch of 5 should complete warm-up of 5"
        );
    }

    #[test]
    fn test_consecutive_alerts_counter() {
        let mut detector = DriftDetector::new(0.01, 50);
        detector.min_sample_size = 10;

        // Warm up with 0.5
        for _ in 0..50 {
            detector.record_score(0.5);
        }

        // Shift distribution heavily
        for _ in 0..200 {
            detector.record_score(0.99);
        }

        assert_eq!(detector.consecutive_alerts(), 0, "Before first check");
        let _ = detector.check();
        assert!(detector.consecutive_alerts() > 0, "After drift check");
    }

    #[test]
    fn test_reset_baseline_clears_alerts() {
        let mut detector = DriftDetector::new(0.01, 50);
        detector.min_sample_size = 10;

        for _ in 0..50 {
            detector.record_score(0.5);
        }
        for _ in 0..200 {
            detector.record_score(0.99);
        }
        let _ = detector.check();
        assert!(detector.consecutive_alerts() > 0);

        detector.reset_baseline();
        assert_eq!(
            detector.consecutive_alerts(),
            0,
            "Reset should clear alerts"
        );
    }

    #[test]
    fn test_status_summary_before_baseline() {
        let detector = DriftDetector::new(0.5, 100);
        let summary = detector.status_summary();
        assert!(!summary.baseline_established);
        assert!(summary.kl_divergence.is_none());
        assert_eq!(summary.warm_up_remaining, 100);
    }

    #[test]
    fn test_status_summary_after_baseline() {
        let mut detector = DriftDetector::new(0.5, 10);
        for _ in 0..10 {
            detector.record_score(0.5);
        }
        let summary = detector.status_summary();
        assert!(summary.baseline_established);
        assert!(summary.kl_divergence.is_some());
        assert_eq!(summary.warm_up_remaining, 0);
    }

    #[test]
    fn test_kl_divergence_identical_distributions() {
        // Two identical histograms should have KL ~ 0
        let mut h1 = Histogram::new();
        let mut h2 = Histogram::new();
        for i in 0..100 {
            let bin = i % NUM_BINS;
            h1.bins[bin] += 1.0;
            h2.bins[bin] += 1.0;
            h1.count += 1;
            h2.count += 1;
        }
        let kl = kl_divergence(&h1, &h2);
        assert!(
            kl < 1e-6,
            "Identical distributions should have KL ~ 0, got {kl}"
        );
    }

    #[test]
    fn test_histogram_normalized_empty() {
        let h = Histogram::new();
        let norm = h.normalized();
        assert_eq!(norm.len(), NUM_BINS);
        // Should return uniform distribution
        let expected = 1.0 / NUM_BINS as f64;
        for val in &norm {
            assert!((val - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_histogram_decay() {
        let mut h = Histogram::new();
        h.bins[0] = 100.0;
        h.decay(0.5);
        assert!((h.bins[0] - 50.0).abs() < 1e-6);
    }
}
