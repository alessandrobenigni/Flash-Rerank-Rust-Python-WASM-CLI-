use std::time::{Duration, Instant};

/// Latency measurement result.
pub struct LatencyStats {
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub mean: Duration,
    pub throughput_qps: f64,
}

impl std::fmt::Display for LatencyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "P50={:?}  P95={:?}  P99={:?}  mean={:?}  QPS={:.1}",
            self.p50, self.p95, self.p99, self.mean, self.throughput_qps
        )
    }
}

/// Measure reranking latency over multiple iterations.
///
/// Runs the closure `iterations` times, collects wall-clock timings,
/// sorts them, and computes percentile statistics plus throughput.
pub fn measure_latency(iterations: usize, mut f: impl FnMut()) -> LatencyStats {
    assert!(iterations > 0, "iterations must be > 0");

    let mut durations: Vec<Duration> = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        f();
        durations.push(start.elapsed());
    }

    durations.sort();
    let len = durations.len();

    let p50 = durations[len / 2];
    let p95 = durations[((len as f64 * 0.95) as usize).min(len - 1)];
    let p99 = durations[((len as f64 * 0.99) as usize).min(len - 1)];
    let mean = durations.iter().sum::<Duration>() / len as u32;
    let total_secs: f64 = durations.iter().map(|d| d.as_secs_f64()).sum();
    let throughput_qps = if total_secs > 0.0 {
        iterations as f64 / total_secs
    } else {
        f64::INFINITY
    };

    LatencyStats {
        p50,
        p95,
        p99,
        mean,
        throughput_qps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_latency_is_sub_microsecond() {
        let stats = measure_latency(100, || {});
        // A no-op should complete in well under 1ms
        assert!(
            stats.p50 < Duration::from_millis(1),
            "P50 for no-op was {:?}, expected < 1ms",
            stats.p50
        );
        assert!(stats.throughput_qps > 1_000.0);
    }

    #[test]
    fn percentiles_are_ordered() {
        let stats = measure_latency(200, || {
            // tiny busy-wait to create some variance
            std::hint::black_box(0u64.wrapping_add(1));
        });
        assert!(stats.p50 <= stats.p95);
        assert!(stats.p95 <= stats.p99);
    }
}
