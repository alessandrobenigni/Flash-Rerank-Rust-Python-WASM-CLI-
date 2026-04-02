# Changelog

## [0.2.0] - 2026-03-28

### Changed
- License changed from MIT/Apache-2.0 to AGPL-3.0-or-later
- Commercial license available at https://alessandrobenigni.com for enterprise/proprietary use
- See COMMERCIAL_LICENSE.md for details

### Fixed
- All clippy lints resolved for clean CI
- models/ gitignore pattern no longer excludes source code modules
- CUDA/TensorRT features now optional (CPU builds work without NVIDIA SDK)
- Python arerank() uses get_running_loop() instead of deprecated get_event_loop()
- Server `--model` flag now accepts HuggingFace model IDs (not just filesystem paths)

## [0.1.1] - 2026-03-28

### Fixed
- Clippy lint fixes
- models/mod.rs missing from git (overly broad gitignore)

## [0.1.0] - 2026-03-28

### Added
- Core reranking engine with ORT backend (CPU, CUDA, TensorRT)
- Parallel sub-batch scorer for CPU (2.6x faster than Jina)
- Auto INT8 model selection for CPU inference
- Dynamic model input detection (supports 2-input and 3-input models)
- Python bindings via PyO3 (sync + async)
- WASM browser reranking via tract-onnx
- CLI with download, compile, bench, serve, models commands
- HTTP server with dynamic batching and SLA-aware dispatch
- Multi-GPU scaling with least-loaded routing
- A/B model comparison with deterministic routing
- Canary deployment with progressive rollout
- Score drift detection with KL divergence
- Score calibration (sigmoid + custom Platt scaling)
- Hybrid RRF fusion for multi-source ranking
- Cascade reranking pipeline (fast -> accurate)
- ColBERT late interaction scoring
- OpenTelemetry tracing (behind feature flag)
- BEIR benchmarking suite with MSMARCO evaluation
- 235 tests: unit, integration, property-based, stress, CLI, snapshot
- BM25-Turbo pipeline bridge
