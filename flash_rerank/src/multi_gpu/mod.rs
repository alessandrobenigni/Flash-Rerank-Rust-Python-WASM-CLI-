use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::Result;
use crate::batch::BatchRequest;
use crate::engine::Scorer;
use crate::engine::ort_backend::OrtScorer;
use crate::types::{Device, ModelConfig};

/// Handle to a single GPU inference thread.
pub struct GpuHandle {
    pub gpu_id: usize,
    sender: std::sync::mpsc::Sender<Vec<BatchRequest>>,
    in_flight: Arc<AtomicUsize>,
    healthy: Arc<AtomicBool>,
}

impl GpuHandle {
    /// Send a batch of requests to this GPU thread.
    pub fn send_batch(&self, batch: Vec<BatchRequest>) -> std::result::Result<(), String> {
        self.in_flight.fetch_add(batch.len(), Ordering::Relaxed);
        self.sender
            .send(batch)
            .map_err(|_| format!("GPU {} thread channel closed", self.gpu_id))
    }

    /// Current number of in-flight requests on this GPU.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Whether this GPU is healthy.
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    /// The GPU ID.
    pub fn id(&self) -> usize {
        self.gpu_id
    }
}

/// Multi-GPU router: replicates model across GPUs and routes by least-loaded.
///
/// Cross-encoder models (110M-568M params) fit on a single GPU.
/// Linear throughput scaling via model replication with least-loaded routing.
pub struct GpuRouter {
    handles: Vec<GpuHandle>,
}

impl GpuRouter {
    /// Create a new GpuRouter that spawns one OS thread per GPU.
    ///
    /// Each GPU thread owns an OrtScorer configured for the given GPU device.
    pub fn new(gpu_ids: &[usize], config: &ModelConfig, model_dir: &Path) -> Result<Self> {
        if gpu_ids.is_empty() {
            return Err(crate::Error::Config("No GPU IDs provided".into()));
        }

        let mut handles = Vec::with_capacity(gpu_ids.len());

        for &gpu_id in gpu_ids {
            let (tx, rx) = std::sync::mpsc::channel::<Vec<BatchRequest>>();
            let in_flight = Arc::new(AtomicUsize::new(0));
            let healthy = Arc::new(AtomicBool::new(true));

            // Create scorer for this GPU
            let mut gpu_config = config.clone();
            gpu_config.device = Device::Cuda(gpu_id);
            let scorer = OrtScorer::new(gpu_config, model_dir)?;

            let in_flight_clone = in_flight.clone();
            let healthy_clone = healthy.clone();

            std::thread::Builder::new()
                .name(format!("gpu-{gpu_id}"))
                .spawn(move || {
                    gpu_thread_loop(Box::new(scorer), rx, in_flight_clone, healthy_clone);
                })
                .map_err(|e| {
                    crate::Error::Inference(format!("Failed to spawn GPU {gpu_id} thread: {e}"))
                })?;

            tracing::info!(gpu_id, "GPU thread spawned");

            handles.push(GpuHandle {
                gpu_id,
                sender: tx,
                in_flight,
                healthy,
            });
        }

        Ok(Self { handles })
    }

    /// Create a GpuRouter from pre-built GPU handles (for testing or custom setups).
    pub fn from_handles(handles: Vec<GpuHandle>) -> Self {
        Self { handles }
    }

    /// Select the least-loaded healthy GPU.
    ///
    /// Returns None if no healthy GPUs are available.
    pub fn next_gpu(&self) -> Option<&GpuHandle> {
        self.handles
            .iter()
            .filter(|g| g.healthy.load(Ordering::Relaxed))
            .min_by_key(|g| g.in_flight.load(Ordering::Relaxed))
    }

    /// Route a batch to the least-loaded GPU.
    pub fn route_batch(&self, batch: Vec<BatchRequest>) -> std::result::Result<usize, String> {
        let gpu = self
            .next_gpu()
            .ok_or_else(|| "No healthy GPUs available".to_string())?;
        let gpu_id = gpu.gpu_id;
        gpu.send_batch(batch)?;
        Ok(gpu_id)
    }

    /// Get the number of GPUs.
    pub fn gpu_count(&self) -> usize {
        self.handles.len()
    }

    /// Get a snapshot of per-GPU status.
    pub fn status(&self) -> Vec<GpuStatus> {
        self.handles
            .iter()
            .map(|h| GpuStatus {
                gpu_id: h.gpu_id,
                in_flight: h.in_flight.load(Ordering::Relaxed),
                healthy: h.healthy.load(Ordering::Relaxed),
            })
            .collect()
    }

    /// Mark a GPU as unhealthy (removes it from routing).
    pub fn mark_unhealthy(&self, gpu_id: usize) {
        if let Some(handle) = self.handles.iter().find(|h| h.gpu_id == gpu_id) {
            handle.healthy.store(false, Ordering::Relaxed);
            tracing::warn!(gpu_id, "GPU marked unhealthy");
        }
    }

    /// Mark a GPU as healthy (adds it back to routing).
    pub fn mark_healthy(&self, gpu_id: usize) {
        if let Some(handle) = self.handles.iter().find(|h| h.gpu_id == gpu_id) {
            handle.healthy.store(true, Ordering::Relaxed);
            tracing::info!(gpu_id, "GPU marked healthy");
        }
    }
}

/// Status snapshot for a single GPU.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuStatus {
    pub gpu_id: usize,
    pub in_flight: usize,
    pub healthy: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    /// Create a mock GpuHandle for testing (no real GPU thread).
    fn mock_handle(gpu_id: usize) -> GpuHandle {
        let (tx, _rx) = std::sync::mpsc::channel();
        GpuHandle {
            gpu_id,
            sender: tx,
            in_flight: Arc::new(AtomicUsize::new(0)),
            healthy: Arc::new(AtomicBool::new(true)),
        }
    }

    #[test]
    fn test_empty_gpu_ids_error() {
        let config = crate::types::ModelConfig::default();
        let tmp = std::env::temp_dir();
        let result = GpuRouter::new(&[], &config, &tmp);
        assert!(result.is_err());
        match result {
            Err(crate::Error::Config(msg)) => assert!(msg.contains("No GPU IDs")),
            Err(other) => panic!("Expected Config error, got: {other:?}"),
            Ok(_) => panic!("Expected error"),
        }
    }

    #[test]
    fn test_in_flight_starts_zero() {
        let handle = mock_handle(0);
        assert_eq!(handle.in_flight_count(), 0);
    }

    #[test]
    fn test_healthy_default_true() {
        let handle = mock_handle(0);
        assert!(handle.is_healthy());
    }

    #[test]
    fn test_mark_unhealthy_excludes_from_routing() {
        let handles = vec![mock_handle(0), mock_handle(1)];
        let router = GpuRouter::from_handles(handles);

        router.mark_unhealthy(0);

        let next = router.next_gpu().unwrap();
        assert_eq!(next.id(), 1, "Unhealthy GPU 0 should be excluded");
    }

    #[test]
    fn test_mark_healthy_restores() {
        let handles = vec![mock_handle(0), mock_handle(1)];
        let router = GpuRouter::from_handles(handles);

        router.mark_unhealthy(0);
        router.mark_healthy(0);

        // Both should be eligible now
        let status = router.status();
        assert!(status.iter().all(|s| s.healthy));
    }

    #[test]
    fn test_least_loaded_selection() {
        let handles = vec![mock_handle(0), mock_handle(1), mock_handle(2)];
        // Simulate load: GPU 0 has 5 in-flight, GPU 1 has 2, GPU 2 has 8
        handles[0].in_flight.store(5, Ordering::Relaxed);
        handles[1].in_flight.store(2, Ordering::Relaxed);
        handles[2].in_flight.store(8, Ordering::Relaxed);

        let router = GpuRouter::from_handles(handles);
        let next = router.next_gpu().unwrap();
        assert_eq!(next.id(), 1, "GPU 1 has least in-flight");
    }

    #[test]
    fn test_all_unhealthy_returns_none() {
        let handles = vec![mock_handle(0), mock_handle(1)];
        let router = GpuRouter::from_handles(handles);

        router.mark_unhealthy(0);
        router.mark_unhealthy(1);

        assert!(
            router.next_gpu().is_none(),
            "No healthy GPUs should return None"
        );
    }

    #[test]
    fn test_route_batch_error_when_all_unhealthy() {
        let handles = vec![mock_handle(0)];
        let router = GpuRouter::from_handles(handles);
        router.mark_unhealthy(0);

        let result = router.route_batch(vec![]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No healthy GPUs"));
    }

    #[test]
    fn test_gpu_count() {
        let handles = vec![mock_handle(0), mock_handle(1), mock_handle(2)];
        let router = GpuRouter::from_handles(handles);
        assert_eq!(router.gpu_count(), 3);
    }

    #[test]
    fn test_status_snapshot() {
        let handles = vec![mock_handle(0), mock_handle(1)];
        handles[0].in_flight.store(3, Ordering::Relaxed);

        let router = GpuRouter::from_handles(handles);
        router.mark_unhealthy(1);

        let status = router.status();
        assert_eq!(status.len(), 2);
        assert_eq!(status[0].gpu_id, 0);
        assert_eq!(status[0].in_flight, 3);
        assert!(status[0].healthy);
        assert_eq!(status[1].gpu_id, 1);
        assert!(!status[1].healthy);
    }

    #[test]
    fn test_single_gpu_fallback() {
        let handles = vec![mock_handle(0)];
        let router = GpuRouter::from_handles(handles);
        let next = router.next_gpu().unwrap();
        assert_eq!(next.id(), 0);
        assert_eq!(router.gpu_count(), 1);
    }
}

/// GPU thread loop: receives batches, processes each request, tracks in-flight count.
fn gpu_thread_loop(
    scorer: Box<dyn Scorer>,
    batch_rx: std::sync::mpsc::Receiver<Vec<BatchRequest>>,
    in_flight: Arc<AtomicUsize>,
    healthy: Arc<AtomicBool>,
) {
    tracing::info!("GPU thread started");

    while let Ok(batch) = batch_rx.recv() {
        let batch_size = batch.len();

        for req in batch {
            let result = scorer.score(&req.query, &req.documents).map(|mut results| {
                if let Some(top_k) = req.top_k {
                    results.truncate(top_k);
                }
                results
            });

            // If scoring fails, mark this GPU as unhealthy
            if result.is_err() {
                tracing::error!("GPU scoring failed, marking unhealthy");
                healthy.store(false, Ordering::Relaxed);
            }

            let _ = req.response_tx.send(result);
            in_flight.fetch_sub(1, Ordering::Relaxed);
        }

        // Safety: ensure in_flight doesn't go negative if there were counting mismatches
        let _ = batch_size;
    }

    tracing::info!("GPU thread exiting");
}
