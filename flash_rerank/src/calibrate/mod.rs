use crate::Result;

/// Trait for score calibration (raw logits -> calibrated probabilities).
pub trait Calibrator: Send + Sync {
    fn calibrate(&self, raw_score: f32) -> f32;
}

/// Sigmoid calibration: maps raw scores to [0, 1] via sigmoid function.
pub struct SigmoidCalibrator;

impl Calibrator for SigmoidCalibrator {
    fn calibrate(&self, raw_score: f32) -> f32 {
        1.0 / (1.0 + (-raw_score).exp())
    }
}

/// Platt scaling calibration with learned parameters A and B.
/// P(relevant | score) = 1 / (1 + exp(A * score + B))
#[derive(Debug)]
pub struct PlattCalibrator {
    pub a: f64,
    pub b: f64,
}

impl PlattCalibrator {
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }

    /// Fit Platt scaling parameters from labeled data using Newton's method.
    ///
    /// Implements the algorithm from Lin, Lin & Weng (2007):
    /// "A Note on Platt's Probabilistic Outputs for Support Vector Machines".
    ///
    /// Fits `P(y=1|f) = 1/(1 + exp(A*f + B))` by minimizing the negative
    /// log-likelihood via Newton's method with regularized target probabilities.
    ///
    /// # Arguments
    /// * `scores` - Raw model output scores for each sample.
    /// * `labels` - Binary relevance labels (true = relevant, false = not relevant).
    ///
    /// # Errors
    /// Returns `Error::Calibration` if fewer than 10 samples are provided.
    pub fn fit(scores: &[f32], labels: &[bool]) -> Result<Self> {
        if scores.len() < 10 {
            return Err(crate::Error::Calibration(format!(
                "Insufficient data: need >= 10 samples, got {}",
                scores.len()
            )));
        }
        if scores.len() != labels.len() {
            return Err(crate::Error::Calibration(format!(
                "Mismatched lengths: {} scores vs {} labels",
                scores.len(),
                labels.len()
            )));
        }

        let n_pos = labels.iter().filter(|&&l| l).count() as f64;
        let n_neg = labels.len() as f64 - n_pos;

        // Regularized target probabilities (Lin et al. 2007)
        let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
        let t_neg = 1.0 / (n_neg + 2.0);
        let targets: Vec<f64> = labels
            .iter()
            .map(|&l| if l { t_pos } else { t_neg })
            .collect();

        // Initialize: A=0, B=ln((n_neg+1)/(n_pos+1))
        let mut a: f64 = 0.0;
        let mut b: f64 = (n_neg + 1.0).ln() - (n_pos + 1.0).ln();

        // Newton's method (100 iterations max)
        for _ in 0..100 {
            // Regularization terms for Hessian diagonal
            let mut h11: f64 = 1e-3;
            let mut h22: f64 = 1e-3;
            let mut h21: f64 = 0.0;
            let mut g1: f64 = 0.0;
            let mut g2: f64 = 0.0;

            for i in 0..scores.len() {
                let f_i = scores[i] as f64;
                let fapb = a * f_i + b;

                // Numerically stable sigmoid computation
                let p = if fapb >= 0.0 {
                    let e = (-fapb).exp();
                    e / (1.0 + e)
                } else {
                    1.0 / (1.0 + fapb.exp())
                };

                let q = 1.0 - p;
                let d2 = p * q;
                let d1 = targets[i] - p;

                h11 += f_i * f_i * d2;
                h22 += d2;
                h21 += f_i * d2;
                g1 += f_i * d1;
                g2 += d1;
            }

            // Solve Newton step: H * [da, db] = -[g1, g2]
            let det = h11 * h22 - h21 * h21;
            if det.abs() < 1e-15 {
                // Hessian is singular, stop iteration
                break;
            }

            let da = -(h22 * g1 - h21 * g2) / det;
            let db = -(-h21 * g1 + h11 * g2) / det;

            a += da;
            b += db;

            // Convergence check
            if da.abs() < 1e-5 && db.abs() < 1e-5 {
                break;
            }
        }

        Ok(Self { a, b })
    }

    /// Compute Expected Calibration Error (ECE) over binned predictions.
    ///
    /// Partitions predictions into `n_bins` equally-spaced bins by predicted
    /// probability, then computes the weighted average of |mean_predicted - mean_actual|
    /// per bin, weighted by the fraction of samples in each bin.
    ///
    /// # Arguments
    /// * `scores` - Raw model scores (will be calibrated internally).
    /// * `labels` - Binary relevance labels.
    /// * `n_bins` - Number of calibration bins (default: 10).
    pub fn expected_calibration_error(
        &self,
        scores: &[f32],
        labels: &[bool],
        n_bins: usize,
    ) -> f64 {
        let n = scores.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let mut bin_sums = vec![0.0f64; n_bins]; // sum of predicted probabilities
        let mut bin_true = vec![0.0f64; n_bins]; // sum of actual labels
        let mut bin_counts = vec![0usize; n_bins];

        for (i, &score) in scores.iter().enumerate() {
            let prob = self.calibrate(score) as f64;
            // Clamp to [0, 1) for binning, with 1.0 going into the last bin
            let bin = ((prob * n_bins as f64) as usize).min(n_bins - 1);
            bin_sums[bin] += prob;
            bin_true[bin] += if labels[i] { 1.0 } else { 0.0 };
            bin_counts[bin] += 1;
        }

        let mut ece = 0.0;
        for bin in 0..n_bins {
            if bin_counts[bin] > 0 {
                let count = bin_counts[bin] as f64;
                let avg_pred = bin_sums[bin] / count;
                let avg_actual = bin_true[bin] / count;
                ece += (count / n) * (avg_pred - avg_actual).abs();
            }
        }

        ece
    }
}

impl Calibrator for PlattCalibrator {
    fn calibrate(&self, raw_score: f32) -> f32 {
        let logit = self.a * raw_score as f64 + self.b;
        (1.0 / (1.0 + logit.exp())) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_calibrator() {
        let cal = SigmoidCalibrator;
        assert!((cal.calibrate(0.0) - 0.5).abs() < 1e-6);
        assert!(cal.calibrate(10.0) > 0.99);
        assert!(cal.calibrate(-10.0) < 0.01);
    }

    #[test]
    fn test_platt_fit_converges_on_synthetic_data() {
        // Generate synthetic data with known sigmoid parameters
        // True model: P(y=1|f) = 1/(1 + exp(-2*f + 1))  =>  A=-2, B=1
        let true_a = -2.0_f64;
        let true_b = 1.0_f64;

        let mut scores = Vec::new();
        let mut labels = Vec::new();

        // Generate 200 samples with deterministic "random" pattern
        for i in 0..200 {
            let f = (i as f64 - 100.0) / 50.0; // scores in [-2, 2]
            let p = 1.0 / (1.0 + (true_a * f + true_b).exp());

            // Deterministic labeling: label = 1 if p > threshold based on index
            let threshold = (i as f64 * 7.0 % 100.0) / 100.0;
            scores.push(f as f32);
            labels.push(p > threshold);
        }

        let calibrator = PlattCalibrator::fit(&scores, &labels).unwrap();

        // The fitted parameters should be reasonably close to true values
        // (not exact due to deterministic noise in labeling)
        assert!(
            (calibrator.a - true_a).abs() < 1.5,
            "A={} should be close to {}",
            calibrator.a,
            true_a
        );
        assert!(
            (calibrator.b - true_b).abs() < 1.5,
            "B={} should be close to {}",
            calibrator.b,
            true_b
        );

        // Calibrated probabilities should be in [0, 1]
        for &s in &scores {
            let prob = calibrator.calibrate(s);
            assert!(prob >= 0.0 && prob <= 1.0, "prob={prob} out of range");
        }
    }

    #[test]
    fn test_platt_fit_insufficient_data() {
        let scores = vec![0.1, 0.2, 0.3];
        let labels = vec![true, false, true];
        let result = PlattCalibrator::fit(&scores, &labels);
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::Error::Calibration(msg) => {
                assert!(msg.contains("Insufficient data"));
            }
            other => panic!("Expected Calibration error, got: {other:?}"),
        }
    }

    #[test]
    fn test_platt_fit_mismatched_lengths() {
        let scores = vec![0.1; 20];
        let labels = vec![true; 15];
        let result = PlattCalibrator::fit(&scores, &labels);
        assert!(result.is_err());
    }

    #[test]
    fn test_platt_calibrate() {
        let cal = PlattCalibrator::new(-1.0, 0.0);
        // P = 1/(1 + exp(-1*f + 0)) = 1/(1+exp(-f)) = sigmoid(f)
        let p = cal.calibrate(0.0);
        assert!((p - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_expected_calibration_error() {
        // Perfect calibrator should have low ECE
        let cal = PlattCalibrator::new(-1.0, 0.0);
        let scores: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 25.0).collect();
        let labels: Vec<bool> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let p = cal.calibrate(s);
                // Deterministic labeling based on predicted probability
                let threshold = (i as f32 * 7.0 % 100.0) / 100.0;
                p > threshold
            })
            .collect();

        let ece = cal.expected_calibration_error(&scores, &labels, 10);
        // ECE should be a reasonable value (not perfect but bounded)
        assert!(ece >= 0.0 && ece <= 1.0, "ECE={ece} should be in [0,1]");
    }

    // --- Additional sigmoid tests ---

    #[test]
    fn test_sigmoid_at_positive_infinity() {
        let cal = SigmoidCalibrator;
        let result = cal.calibrate(f32::INFINITY);
        assert!(
            (result - 1.0).abs() < 1e-6,
            "sigmoid(+inf) should be 1.0, got {result}"
        );
    }

    #[test]
    fn test_sigmoid_at_negative_infinity() {
        let cal = SigmoidCalibrator;
        let result = cal.calibrate(f32::NEG_INFINITY);
        assert!(
            result.abs() < 1e-6,
            "sigmoid(-inf) should be 0.0, got {result}"
        );
    }

    #[test]
    fn test_sigmoid_at_nan() {
        let cal = SigmoidCalibrator;
        let result = cal.calibrate(f32::NAN);
        assert!(result.is_nan(), "sigmoid(NaN) should be NaN");
    }

    #[test]
    fn test_sigmoid_monotonicity() {
        let cal = SigmoidCalibrator;
        let scores: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        for window in scores.windows(2) {
            let a = cal.calibrate(window[0]);
            let b = cal.calibrate(window[1]);
            assert!(
                b >= a,
                "Sigmoid should be monotonically non-decreasing: f({}) = {} > f({}) = {}",
                window[0],
                a,
                window[1],
                b
            );
        }
    }

    // --- Additional Platt fit edge cases ---

    #[test]
    fn test_platt_fit_all_positive_labels() {
        let scores: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let labels = vec![true; 20];
        let result = PlattCalibrator::fit(&scores, &labels);
        assert!(result.is_ok(), "All-positive labels should still converge");
        let cal = result.unwrap();
        // All-positive: calibrated outputs should be high
        let p = cal.calibrate(1.0);
        assert!(
            p > 0.5,
            "High score with all-positive labels should yield p > 0.5, got {p}"
        );
    }

    #[test]
    fn test_platt_fit_all_negative_labels() {
        let scores: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let labels = vec![false; 20];
        let result = PlattCalibrator::fit(&scores, &labels);
        assert!(result.is_ok(), "All-negative labels should still converge");
        let cal = result.unwrap();
        let p = cal.calibrate(0.5);
        assert!(
            p < 0.5,
            "Mid score with all-negative labels should yield p < 0.5, got {p}"
        );
    }

    #[test]
    fn test_platt_fit_exactly_10_samples() {
        let scores: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let labels: Vec<bool> = (0..10).map(|i| i >= 5).collect();
        let result = PlattCalibrator::fit(&scores, &labels);
        assert!(result.is_ok(), "Exactly 10 samples should succeed");
    }

    #[test]
    fn test_platt_fit_9_samples_fails() {
        let scores: Vec<f32> = (0..9).map(|i| i as f32 * 0.1).collect();
        let labels: Vec<bool> = (0..9).map(|i| i >= 5).collect();
        let result = PlattCalibrator::fit(&scores, &labels);
        assert!(result.is_err(), "9 samples should fail");
    }

    #[test]
    fn test_platt_fit_zero_samples_fails() {
        let result = PlattCalibrator::fit(&[], &[]);
        assert!(result.is_err(), "0 samples should fail");
    }

    // --- Platt output range ---

    #[test]
    fn test_platt_calibrate_output_range() {
        let cal = PlattCalibrator::new(-2.0, 0.5);
        for i in -100..=100 {
            let score = i as f32 * 0.1;
            let p = cal.calibrate(score);
            assert!(
                p >= 0.0 && p <= 1.0,
                "Platt calibrate({score}) = {p} should be in [0, 1]"
            );
        }
    }

    // --- ECE edge cases ---

    #[test]
    fn test_ece_empty_data() {
        let cal = PlattCalibrator::new(-1.0, 0.0);
        let ece = cal.expected_calibration_error(&[], &[], 10);
        assert_eq!(ece, 0.0, "ECE on empty data should be 0.0");
    }

    #[test]
    fn test_ece_single_bin() {
        let cal = PlattCalibrator::new(-1.0, 0.0);
        let scores = vec![0.0f32; 20];
        let labels = vec![false; 20];
        let ece = cal.expected_calibration_error(&scores, &labels, 1);
        assert!(
            ece >= 0.0 && ece <= 1.0,
            "ECE with 1 bin should be in [0, 1], got {ece}"
        );
    }

    #[test]
    fn test_ece_many_bins() {
        let cal = PlattCalibrator::new(-1.0, 0.0);
        let scores: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 25.0).collect();
        let labels: Vec<bool> = (0..100).map(|i| i >= 50).collect();
        let ece = cal.expected_calibration_error(&scores, &labels, 100);
        assert!(
            ece >= 0.0 && ece <= 1.0,
            "ECE with 100 bins should be in [0, 1], got {ece}"
        );
    }
}
