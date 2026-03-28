use std::time::{Duration, Instant};

use serde::Serialize;

use crate::drift::DriftStatus;

/// Progressive rollout stages as traffic percentages.
const DEFAULT_STAGES: &[u8] = &[1, 5, 25, 100];

/// Default observation period per stage in seconds.
const DEFAULT_OBSERVATION_SECS: u64 = 300;

/// Default latency tolerance multiplier (canary P95 must be within this
/// factor of the baseline P95).
const DEFAULT_LATENCY_TOLERANCE: f64 = 1.2;

/// State of the canary deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CanaryState {
    /// No canary deployment in progress.
    Inactive,
    /// Canary is active and being observed at the current stage.
    Active,
    /// Canary is being rolled back due to drift or manual abort.
    RollingBack,
    /// Canary completed all stages successfully (100% traffic).
    Completed,
}

/// Action recommended by `check_advance()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanaryAction {
    /// No action needed -- continue observing.
    None,
    /// Advance to the next stage with the given percentage.
    Advance(u8),
    /// Rollback: route 100% to baseline.
    Rollback,
    /// Canary completed all stages.
    Complete,
}

/// Status response for the canary status endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct CanaryStatusResponse {
    pub state: CanaryState,
    pub canary_model: Option<String>,
    pub current_stage_index: usize,
    pub current_percentage: u8,
    pub stages: Vec<u8>,
    pub observation_period_secs: u64,
    pub elapsed_in_stage_secs: f64,
    pub time_remaining_secs: f64,
    pub latency_tolerance: f64,
}

/// Canary deployment manager for gradual model rollouts.
///
/// Progresses through configurable traffic stages (e.g., [1%, 5%, 25%, 100%])
/// with observation periods at each stage. Integrates with `DriftDetector`
/// for automatic rollback when score drift is detected.
pub struct CanaryDeployer {
    state: CanaryState,
    canary_model: Option<String>,
    stages: Vec<u8>,
    current_stage: usize,
    observation_period: Duration,
    stage_started_at: Option<Instant>,
    latency_tolerance: f64,
    counter: u64,
}

impl CanaryDeployer {
    pub fn new() -> Self {
        Self {
            state: CanaryState::Inactive,
            canary_model: None,
            stages: DEFAULT_STAGES.to_vec(),
            current_stage: 0,
            observation_period: Duration::from_secs(DEFAULT_OBSERVATION_SECS),
            stage_started_at: None,
            latency_tolerance: DEFAULT_LATENCY_TOLERANCE,
            counter: 0,
        }
    }

    /// Start a canary deployment for the given model.
    ///
    /// # Arguments
    /// * `canary_model` - identifier of the canary model
    /// * `stages` - traffic percentages per stage (e.g., [1, 5, 25, 100])
    /// * `observation_secs` - seconds to observe at each stage before advancing
    /// * `tolerance` - latency tolerance multiplier vs baseline
    pub fn start(
        &mut self,
        canary_model: String,
        stages: Option<Vec<u8>>,
        observation_secs: Option<u64>,
        tolerance: Option<f64>,
    ) {
        self.canary_model = Some(canary_model.clone());
        self.stages = stages.unwrap_or_else(|| DEFAULT_STAGES.to_vec());
        self.observation_period =
            Duration::from_secs(observation_secs.unwrap_or(DEFAULT_OBSERVATION_SECS));
        self.latency_tolerance = tolerance.unwrap_or(DEFAULT_LATENCY_TOLERANCE);
        self.current_stage = 0;
        self.state = CanaryState::Active;
        self.stage_started_at = Some(Instant::now());
        self.counter = 0;

        tracing::info!(
            canary_model,
            stages = ?self.stages,
            observation_secs = self.observation_period.as_secs(),
            tolerance = self.latency_tolerance,
            "Canary deployment started at stage 0 ({}%)",
            self.stages.first().copied().unwrap_or(0)
        );
    }

    /// Return the current canary traffic percentage.
    pub fn canary_percentage(&self) -> u8 {
        match self.state {
            CanaryState::Active => self.stages.get(self.current_stage).copied().unwrap_or(0),
            _ => 0,
        }
    }

    /// Determine if the current request should be routed to the canary model.
    ///
    /// Uses a simple counter-based approach for deterministic percentage routing.
    pub fn is_canary(&mut self) -> bool {
        if self.state != CanaryState::Active {
            return false;
        }

        let pct = self.canary_percentage();
        if pct == 0 {
            return false;
        }
        if pct >= 100 {
            return true;
        }

        self.counter = self.counter.wrapping_add(1);
        (self.counter % 100) < pct as u64
    }

    /// Evaluate whether the canary should advance, rollback, or continue observing.
    ///
    /// # Arguments
    /// * `drift_status` - current drift status from the DriftDetector
    /// * `latency_ok` - whether canary P95 latency is within tolerance of baseline
    pub fn check_advance(&mut self, drift_status: &DriftStatus, latency_ok: bool) -> CanaryAction {
        if self.state != CanaryState::Active {
            return CanaryAction::None;
        }

        // Rollback on drift warning or critical
        match drift_status {
            DriftStatus::Warning(_) | DriftStatus::Critical(_) => {
                tracing::warn!(
                    drift = ?drift_status,
                    stage = self.current_stage,
                    "Drift detected during canary -- rolling back"
                );
                self.state = CanaryState::RollingBack;
                return CanaryAction::Rollback;
            }
            _ => {}
        }

        // Rollback if latency is out of tolerance
        if !latency_ok {
            tracing::warn!(
                stage = self.current_stage,
                "Canary latency exceeds tolerance -- rolling back"
            );
            self.state = CanaryState::RollingBack;
            return CanaryAction::Rollback;
        }

        // Check if observation period has elapsed
        let elapsed = self
            .stage_started_at
            .map(|t| t.elapsed())
            .unwrap_or_default();

        if elapsed < self.observation_period {
            return CanaryAction::None;
        }

        // Advance to next stage
        self.current_stage += 1;

        if self.current_stage >= self.stages.len() {
            // All stages completed
            self.state = CanaryState::Completed;
            tracing::info!("Canary deployment completed -- all stages passed");
            return CanaryAction::Complete;
        }

        // Reset observation timer for new stage
        self.stage_started_at = Some(Instant::now());
        let new_pct = self.stages[self.current_stage];

        tracing::info!(
            stage = self.current_stage,
            percentage = new_pct,
            "Canary advancing to next stage"
        );

        CanaryAction::Advance(new_pct)
    }

    /// Manually advance to the next stage, bypassing the observation timer.
    pub fn manual_advance(&mut self) -> CanaryAction {
        if self.state != CanaryState::Active {
            return CanaryAction::None;
        }

        self.current_stage += 1;

        if self.current_stage >= self.stages.len() {
            self.state = CanaryState::Completed;
            tracing::info!("Canary manually completed -- all stages passed");
            return CanaryAction::Complete;
        }

        self.stage_started_at = Some(Instant::now());
        let new_pct = self.stages[self.current_stage];

        tracing::info!(
            stage = self.current_stage,
            percentage = new_pct,
            "Canary manually advanced to next stage"
        );

        CanaryAction::Advance(new_pct)
    }

    /// Abort the canary deployment, routing 100% traffic back to baseline.
    pub fn abort(&mut self) {
        tracing::warn!("Canary deployment aborted manually");
        self.state = CanaryState::RollingBack;
        self.current_stage = 0;
    }

    /// Get the current canary state.
    pub fn state(&self) -> CanaryState {
        self.state
    }

    /// Get the canary status for the management endpoint.
    pub fn status(&self) -> CanaryStatusResponse {
        let elapsed = self
            .stage_started_at
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        let remaining = if self.state == CanaryState::Active {
            (self.observation_period.as_secs_f64() - elapsed).max(0.0)
        } else {
            0.0
        };

        CanaryStatusResponse {
            state: self.state,
            canary_model: self.canary_model.clone(),
            current_stage_index: self.current_stage,
            current_percentage: self.canary_percentage(),
            stages: self.stages.clone(),
            observation_period_secs: self.observation_period.as_secs(),
            elapsed_in_stage_secs: elapsed,
            time_remaining_secs: remaining,
            latency_tolerance: self.latency_tolerance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canary_starts_inactive() {
        let deployer = CanaryDeployer::new();
        assert_eq!(deployer.state(), CanaryState::Inactive);
        assert_eq!(deployer.canary_percentage(), 0);
    }

    #[test]
    fn test_canary_start_sets_active() {
        let mut deployer = CanaryDeployer::new();
        deployer.start("new-model".into(), None, None, None);
        assert_eq!(deployer.state(), CanaryState::Active);
        assert_eq!(deployer.canary_percentage(), 1); // first default stage
    }

    #[test]
    fn test_canary_stage_progression() {
        let mut deployer = CanaryDeployer::new();
        deployer.start(
            "new-model".into(),
            Some(vec![1, 5, 25, 100]),
            Some(0), // zero observation period for instant advancement
            None,
        );

        assert_eq!(deployer.canary_percentage(), 1);

        // Advance through stages with stable drift and good latency
        let action = deployer.check_advance(&DriftStatus::Stable, true);
        assert_eq!(action, CanaryAction::Advance(5));
        assert_eq!(deployer.canary_percentage(), 5);

        let action = deployer.check_advance(&DriftStatus::Stable, true);
        assert_eq!(action, CanaryAction::Advance(25));
        assert_eq!(deployer.canary_percentage(), 25);

        let action = deployer.check_advance(&DriftStatus::Stable, true);
        assert_eq!(action, CanaryAction::Advance(100));
        assert_eq!(deployer.canary_percentage(), 100);

        let action = deployer.check_advance(&DriftStatus::Stable, true);
        assert_eq!(action, CanaryAction::Complete);
        assert_eq!(deployer.state(), CanaryState::Completed);
    }

    #[test]
    fn test_canary_rollback_on_drift() {
        let mut deployer = CanaryDeployer::new();
        deployer.start("new-model".into(), Some(vec![1, 5, 25, 100]), Some(0), None);

        let action = deployer.check_advance(&DriftStatus::Warning(0.8), true);
        assert_eq!(action, CanaryAction::Rollback);
        assert_eq!(deployer.state(), CanaryState::RollingBack);
        assert_eq!(deployer.canary_percentage(), 0);
    }

    #[test]
    fn test_canary_rollback_on_critical_drift() {
        let mut deployer = CanaryDeployer::new();
        deployer.start("new-model".into(), Some(vec![1, 5, 25, 100]), Some(0), None);

        let action = deployer.check_advance(&DriftStatus::Critical(1.5), true);
        assert_eq!(action, CanaryAction::Rollback);
        assert_eq!(deployer.state(), CanaryState::RollingBack);
    }

    #[test]
    fn test_canary_rollback_on_latency() {
        let mut deployer = CanaryDeployer::new();
        deployer.start("new-model".into(), Some(vec![1, 5, 25, 100]), Some(0), None);

        let action = deployer.check_advance(&DriftStatus::Stable, false);
        assert_eq!(action, CanaryAction::Rollback);
        assert_eq!(deployer.state(), CanaryState::RollingBack);
    }

    #[test]
    fn test_canary_manual_abort() {
        let mut deployer = CanaryDeployer::new();
        deployer.start("new-model".into(), None, None, None);

        deployer.abort();
        assert_eq!(deployer.state(), CanaryState::RollingBack);
        assert_eq!(deployer.canary_percentage(), 0);
    }

    #[test]
    fn test_canary_manual_advance() {
        let mut deployer = CanaryDeployer::new();
        deployer.start("new-model".into(), Some(vec![1, 5, 25, 100]), None, None);

        assert_eq!(deployer.canary_percentage(), 1);
        let action = deployer.manual_advance();
        assert_eq!(action, CanaryAction::Advance(5));
        assert_eq!(deployer.canary_percentage(), 5);
    }

    #[test]
    fn test_is_canary_routing() {
        let mut deployer = CanaryDeployer::new();
        assert!(!deployer.is_canary()); // inactive

        deployer.start("new-model".into(), Some(vec![100]), Some(300), None);
        // At 100%, all requests should be canary
        for _ in 0..10 {
            assert!(deployer.is_canary());
        }
    }

    #[test]
    fn test_is_canary_inactive_returns_false() {
        let mut deployer = CanaryDeployer::new();
        deployer.start("m".into(), Some(vec![1, 5]), Some(0), None);
        deployer.abort();
        assert!(!deployer.is_canary());
    }

    #[test]
    fn test_status_response() {
        let mut deployer = CanaryDeployer::new();
        deployer.start(
            "test-model".into(),
            Some(vec![1, 5, 25, 100]),
            Some(60),
            Some(1.5),
        );

        let status = deployer.status();
        assert_eq!(status.state, CanaryState::Active);
        assert_eq!(status.canary_model, Some("test-model".into()));
        assert_eq!(status.current_stage_index, 0);
        assert_eq!(status.current_percentage, 1);
        assert_eq!(status.stages, vec![1, 5, 25, 100]);
        assert_eq!(status.observation_period_secs, 60);
        assert!((status.latency_tolerance - 1.5).abs() < f64::EPSILON);
    }

    // --- Additional canary tests ---

    #[test]
    fn test_canary_check_advance_when_inactive() {
        let mut deployer = CanaryDeployer::new();
        let action = deployer.check_advance(&DriftStatus::Stable, true);
        assert_eq!(
            action,
            CanaryAction::None,
            "Inactive deployer should return None action"
        );
    }

    #[test]
    fn test_canary_manual_advance_when_inactive() {
        let mut deployer = CanaryDeployer::new();
        let action = deployer.manual_advance();
        assert_eq!(
            action,
            CanaryAction::None,
            "Inactive deployer should return None on manual advance"
        );
    }

    #[test]
    fn test_canary_custom_stages() {
        let mut deployer = CanaryDeployer::new();
        deployer.start("m".into(), Some(vec![10, 50, 100]), Some(0), None);
        assert_eq!(deployer.canary_percentage(), 10);

        let action = deployer.check_advance(&DriftStatus::Stable, true);
        assert_eq!(action, CanaryAction::Advance(50));

        let action = deployer.check_advance(&DriftStatus::Stable, true);
        assert_eq!(action, CanaryAction::Advance(100));

        let action = deployer.check_advance(&DriftStatus::Stable, true);
        assert_eq!(action, CanaryAction::Complete);
    }

    #[test]
    fn test_canary_manual_advance_completes() {
        let mut deployer = CanaryDeployer::new();
        deployer.start("m".into(), Some(vec![50, 100]), None, None);

        let action = deployer.manual_advance();
        assert_eq!(action, CanaryAction::Advance(100));

        let action = deployer.manual_advance();
        assert_eq!(action, CanaryAction::Complete);
        assert_eq!(deployer.state(), CanaryState::Completed);
    }

    #[test]
    fn test_canary_is_canary_percentage_routing() {
        let mut deployer = CanaryDeployer::new();
        deployer.start("m".into(), Some(vec![50]), Some(300), None);

        let mut canary_count = 0;
        for _ in 0..200 {
            if deployer.is_canary() {
                canary_count += 1;
            }
        }
        // At 50%, roughly 100 out of 200 should be canary
        assert!(
            canary_count >= 80 && canary_count <= 120,
            "Expected ~100 canary requests at 50%, got {canary_count}"
        );
    }

    #[test]
    fn test_canary_status_when_inactive() {
        let deployer = CanaryDeployer::new();
        let status = deployer.status();
        assert_eq!(status.state, CanaryState::Inactive);
        assert!(status.canary_model.is_none());
        assert_eq!(status.current_percentage, 0);
        assert_eq!(status.time_remaining_secs, 0.0);
    }

    #[test]
    fn test_canary_rollback_on_recovered_drift_is_stable() {
        let mut deployer = CanaryDeployer::new();
        deployer.start("m".into(), Some(vec![1, 100]), Some(0), None);

        // Recovered drift should NOT trigger rollback
        let action = deployer.check_advance(&DriftStatus::Recovered, true);
        assert_eq!(
            action,
            CanaryAction::Advance(100),
            "Recovered drift should allow advancement"
        );
    }
}
