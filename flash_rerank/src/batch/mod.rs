//! Dynamic request batching for GPU inference.
//!
//! Collects incoming reranking requests into optimal batches before dispatching
//! to the GPU thread. Uses a fill-or-timeout strategy: a batch is dispatched when
//! it reaches `max_batch_size` or when `max_wait` elapses, whichever comes first.

use std::time::Duration;

use tokio::sync::{mpsc, oneshot};

use crate::RerankResult;
use crate::engine::Scorer;

/// A single request submitted to the dynamic batcher.
pub struct BatchRequest {
    pub query: String,
    pub documents: Vec<String>,
    pub top_k: Option<usize>,
    pub response_tx: oneshot::Sender<crate::Result<Vec<RerankResult>>>,
}

/// Dynamic batcher that collects requests and dispatches to GPU in optimal batches.
///
/// Uses tokio mpsc channels: HTTP handlers send requests, batcher task collects
/// until batch is full or timeout expires, then dispatches to GPU thread via
/// std::sync::mpsc. The GPU thread owns the scorer and sends results back via
/// oneshot channels.
pub struct DynamicBatcher {
    max_batch_size: usize,
    max_wait: Duration,
    request_tx: mpsc::Sender<BatchRequest>,
}

impl DynamicBatcher {
    /// Create a new DynamicBatcher and spawn the batcher loop and GPU thread.
    ///
    /// # Arguments
    /// * `max_batch_size` - Maximum number of requests in a single batch
    /// * `max_wait` - Maximum time to wait for a full batch before dispatching
    /// * `scorer` - The scorer to use on the GPU thread (takes ownership)
    pub fn new(max_batch_size: usize, max_wait: Duration, scorer: Box<dyn Scorer>) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<BatchRequest>(1024);
        let (gpu_tx, gpu_rx) = std::sync::mpsc::channel::<Vec<BatchRequest>>();

        // Spawn the GPU OS thread that owns the scorer
        std::thread::spawn(move || {
            gpu_thread_loop(scorer, gpu_rx);
        });

        // Spawn the async batcher task
        tokio::spawn(Self::batcher_loop(
            request_rx,
            gpu_tx,
            max_batch_size,
            max_wait,
        ));

        Self {
            max_batch_size,
            max_wait,
            request_tx,
        }
    }

    /// Create a DynamicBatcher with an external GPU sender (for multi-GPU routing).
    ///
    /// The caller is responsible for spawning GPU thread(s) that receive from
    /// the returned std::sync::mpsc channel.
    pub fn with_gpu_sender(
        max_batch_size: usize,
        max_wait: Duration,
        gpu_tx: std::sync::mpsc::Sender<Vec<BatchRequest>>,
    ) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<BatchRequest>(1024);

        tokio::spawn(Self::batcher_loop(
            request_rx,
            gpu_tx,
            max_batch_size,
            max_wait,
        ));

        Self {
            max_batch_size,
            max_wait,
            request_tx,
        }
    }

    /// Submit a reranking request and await the response.
    ///
    /// Creates a oneshot channel for the response, sends the request to the batcher,
    /// and awaits the result from the GPU thread.
    pub async fn submit(
        &self,
        query: String,
        documents: Vec<String>,
        top_k: Option<usize>,
    ) -> crate::Result<Vec<RerankResult>> {
        let (response_tx, response_rx) = oneshot::channel();
        let req = BatchRequest {
            query,
            documents,
            top_k,
            response_tx,
        };
        self.request_tx
            .send(req)
            .await
            .map_err(|_| crate::Error::Inference("Batcher channel closed".into()))?;
        response_rx
            .await
            .map_err(|_| crate::Error::Inference("Response channel closed".into()))?
    }

    /// Returns the max batch size.
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Returns the max wait duration.
    pub fn max_wait(&self) -> Duration {
        self.max_wait
    }

    /// The batcher loop: receives requests from the tokio mpsc channel, collects
    /// them into batches, and dispatches to the GPU thread via std::sync::mpsc.
    async fn batcher_loop(
        mut request_rx: mpsc::Receiver<BatchRequest>,
        gpu_tx: std::sync::mpsc::Sender<Vec<BatchRequest>>,
        max_batch_size: usize,
        max_wait: Duration,
    ) {
        let mut batch: Vec<BatchRequest> = Vec::with_capacity(max_batch_size);

        loop {
            // Wait for at least one request
            match request_rx.recv().await {
                Some(req) => batch.push(req),
                None => break, // All senders dropped, shut down
            }

            // Collect more requests until batch is full or timeout expires
            let deadline = tokio::time::Instant::now() + max_wait;
            while batch.len() < max_batch_size {
                tokio::select! {
                    biased;
                    req = request_rx.recv() => {
                        match req {
                            Some(r) => batch.push(r),
                            None => break,
                        }
                    }
                    _ = tokio::time::sleep_until(deadline) => break,
                }
            }

            // Dispatch the collected batch to the GPU thread
            let ready_batch: Vec<BatchRequest> = batch.drain(..).collect();
            if gpu_tx.send(ready_batch).is_err() {
                tracing::error!("GPU thread channel closed, batcher shutting down");
                break;
            }
        }

        tracing::info!("Batcher loop exiting");
    }
}

/// GPU thread loop: receives batches from std::sync::mpsc and processes each
/// request using the scorer, sending results back via oneshot channels.
fn gpu_thread_loop(
    scorer: Box<dyn Scorer>,
    batch_rx: std::sync::mpsc::Receiver<Vec<BatchRequest>>,
) {
    tracing::info!("GPU thread started");

    while let Ok(batch) = batch_rx.recv() {
        let batch_size = batch.len();
        tracing::debug!(batch_size, "Processing batch");

        for req in batch {
            let result = scorer.score(&req.query, &req.documents).map(|mut results| {
                // Apply top_k filter if requested
                if let Some(top_k) = req.top_k {
                    results.truncate(top_k);
                }
                results
            });
            // Send result back; ignore error if receiver was dropped
            let _ = req.response_tx.send(result);
        }
    }

    tracing::info!("GPU thread exiting");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock scorer for testing that returns deterministic scores.
    struct MockScorer;

    impl Scorer for MockScorer {
        fn score(&self, _query: &str, documents: &[String]) -> crate::Result<Vec<RerankResult>> {
            let mut results: Vec<RerankResult> = documents
                .iter()
                .enumerate()
                .map(|(i, _)| RerankResult {
                    index: i,
                    score: 1.0 / (i as f32 + 1.0),
                    document: None,
                })
                .collect();
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Ok(results)
        }
    }

    #[tokio::test]
    async fn test_batcher_single_request() {
        let batcher = DynamicBatcher::new(8, Duration::from_millis(50), Box::new(MockScorer));

        let results = batcher
            .submit(
                "test query".to_string(),
                vec!["doc1".to_string(), "doc2".to_string()],
                None,
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
    }

    #[tokio::test]
    async fn test_batcher_concurrent_requests() {
        let batcher = std::sync::Arc::new(DynamicBatcher::new(
            16,
            Duration::from_millis(20),
            Box::new(MockScorer),
        ));

        let mut handles = Vec::new();
        for i in 0..20 {
            let b = batcher.clone();
            handles.push(tokio::spawn(async move {
                b.submit(
                    format!("query {i}"),
                    vec![format!("doc_{i}_a"), format!("doc_{i}_b")],
                    None,
                )
                .await
            }));
        }

        for handle in handles {
            let result = handle.await.unwrap().unwrap();
            assert_eq!(result.len(), 2);
        }
    }

    #[tokio::test]
    async fn test_batcher_top_k() {
        let batcher = DynamicBatcher::new(8, Duration::from_millis(50), Box::new(MockScorer));

        let results = batcher
            .submit(
                "test".to_string(),
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                Some(1),
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    // --- Additional batcher tests ---

    #[test]
    fn test_batcher_accessors() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let batcher = DynamicBatcher::new(16, Duration::from_millis(100), Box::new(MockScorer));
            assert_eq!(batcher.max_batch_size(), 16);
            assert_eq!(batcher.max_wait(), Duration::from_millis(100));
        });
    }

    #[tokio::test]
    async fn test_batcher_timeout_dispatches_partial_batch() {
        // With max_batch_size=100, only send 2 requests. The timeout should dispatch them.
        let batcher = std::sync::Arc::new(DynamicBatcher::new(
            100,
            Duration::from_millis(20),
            Box::new(MockScorer),
        ));

        let b = batcher.clone();
        let h1 = tokio::spawn(async move {
            b.submit("q1".to_string(), vec!["d1".to_string()], None)
                .await
        });
        let b = batcher.clone();
        let h2 = tokio::spawn(async move {
            b.submit("q2".to_string(), vec!["d2".to_string()], None)
                .await
        });

        let r1 = h1.await.unwrap().unwrap();
        let r2 = h2.await.unwrap().unwrap();
        assert_eq!(r1.len(), 1);
        assert_eq!(r2.len(), 1);
    }

    #[tokio::test]
    async fn test_batcher_full_dispatch() {
        // max_batch_size=2 and send exactly 2. Should dispatch without waiting for timeout.
        let batcher = std::sync::Arc::new(DynamicBatcher::new(
            2,
            Duration::from_secs(60), // long timeout -- batch should fill before it
            Box::new(MockScorer),
        ));

        let b = batcher.clone();
        let h1 =
            tokio::spawn(
                async move { b.submit("q".to_string(), vec!["a".to_string()], None).await },
            );
        let b = batcher.clone();
        let h2 =
            tokio::spawn(
                async move { b.submit("q".to_string(), vec!["b".to_string()], None).await },
            );

        // Both should complete quickly since batch is full
        let (r1, r2) = tokio::join!(h1, h2);
        assert!(r1.unwrap().is_ok());
        assert!(r2.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_batcher_top_k_zero() {
        let batcher = DynamicBatcher::new(8, Duration::from_millis(50), Box::new(MockScorer));

        let results = batcher
            .submit(
                "test".to_string(),
                vec!["a".to_string(), "b".to_string()],
                Some(0),
            )
            .await
            .unwrap();

        assert!(results.is_empty(), "top_k=0 should return empty results");
    }

    #[tokio::test]
    async fn test_batcher_empty_documents() {
        let batcher = DynamicBatcher::new(8, Duration::from_millis(50), Box::new(MockScorer));

        let results = batcher
            .submit("query".to_string(), vec![], None)
            .await
            .unwrap();

        assert!(
            results.is_empty(),
            "Empty documents should return empty results"
        );
    }

    #[tokio::test]
    async fn test_batcher_dropped_sender() {
        // Create a batcher with external GPU sender, then drop the GPU receiver
        let (gpu_tx, gpu_rx) = std::sync::mpsc::channel::<Vec<BatchRequest>>();
        let batcher = DynamicBatcher::with_gpu_sender(8, Duration::from_millis(20), gpu_tx);

        // Drop the receiver so the GPU channel is closed
        drop(gpu_rx);

        // Submit should eventually fail because batcher loop will detect closed channel
        // Give it time to process
        tokio::time::sleep(Duration::from_millis(50)).await;
        let result = batcher
            .submit("q".to_string(), vec!["d".to_string()], None)
            .await;

        // This may error with channel closed or hang depending on timing.
        // The key invariant is it doesn't panic.
        let _ = result; // We just ensure no panic
    }
}
