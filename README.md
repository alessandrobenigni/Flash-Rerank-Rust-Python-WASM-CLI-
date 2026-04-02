<p align="center">
  <h1 align="center">Flash-Rerank <sup>⚡</sup></h1>
  <p align="center"><strong>Rust &middot; Python &middot; WASM &middot; CLI</strong></p>
  <p align="center">The fastest reranker in the world. Pure Rust. No GPU required.</p>
</p>

<p align="center">
  <a href="https://github.com/TheSauceSuite/Flash-Rerank-Rust-Python-WASM-CLI-/actions"><img src="https://img.shields.io/github/actions/workflow/status/TheSauceSuite/Flash-Rerank-Rust-Python-WASM-CLI-/ci.yml?label=CI" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License"></a>
  <a href="https://crates.io/crates/flash_rerank"><img src="https://img.shields.io/crates/v/flash_rerank" alt="crates.io"></a>
  <a href="https://pypi.org/project/flash-rerank/"><img src="https://img.shields.io/pypi/v/flash-rerank" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/flash-rerank-wasm"><img src="https://img.shields.io/npm/v/flash-rerank-wasm" alt="npm"></a>
  <a href="https://github.com/TheSauceSuite/Flash-Rerank-Rust-Python-WASM-CLI-/stargazers"><img src="https://img.shields.io/github/stars/TheSauceSuite/Flash-Rerank-Rust-Python-WASM-CLI-?style=social" alt="Stars"></a>
</p>

---

**72ms to rerank 100 documents on CPU.** No GPU, no cloud API, no Python in the inference path. Pure Rust with INT8 quantization, parallel sub-batch scoring, and zero-copy tokenization.

Combined with [**BM25-Turbo**](https://github.com/TheSauceSuite/BM25-Turbo-Rust-Python-WASM-CLI), Flash-Rerank searches **8.8 million documents and semantically reranks the top 100 in 80ms** — faster than any competitor can rerank 100 pre-selected documents alone.

```python
import flash_rerank

reranker = flash_rerank.load("cross-encoder/ms-marco-MiniLM-L-6-v2")
results = reranker.rerank("what is machine learning?", [
    "ML is a subset of artificial intelligence",
    "The weather today is sunny",
    "Deep learning uses neural networks",
], top_k=2)
```

---

## Performance

> Benchmarked on Intel i9-12900H (16 threads), RTX 3080 Ti Laptop GPU, 32GB RAM. Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, INT8 quantized). Parallel sub-batch inference with 8 workers. All benchmarks reproducible via `cargo bench`.

### Competitive Comparison — Reranking 100 Documents

<p align="center">
  <img src="assets/competitive-chart.svg" alt="Competitive comparison chart" width="700">
</p>

| Provider | Latency | Type | vs Flash-Rerank |
|----------|---------|------|-----------------|
| **Flash-Rerank** (Parallel CPU INT8) | **72ms** | Local, open-source | — |
| **Flash-Rerank + BM25-Turbo** | **80ms** | Search 8.8M docs + rerank | **Entire pipeline** |
| Jina Reranker v3 | 188ms | Cloud API, GPU | 2.6x slower |
| Cohere Rerank 3.5 | 595ms | Cloud API | 8.3x slower |
| Voyage Rerank 2.5 | 603ms | Cloud API | 8.4x slower |

> **Key insight:** Our full pipeline (keyword search across 8.8M documents + semantic reranking of top 100) completes in **80ms** — faster than Jina, Cohere, or Voyage can rerank 100 pre-selected documents alone. And we're running on a laptop CPU.

### Latency Scaling

<p align="center">
  <img src="assets/scaling-chart.svg" alt="Latency scaling chart" width="700">
</p>

| Documents | P50 Latency | Per-Doc Cost | QPS |
|-----------|------------|-------------|-----|
| 1 | **2.7ms** | 2.7ms | 353 |
| 10 | **10.3ms** | 1.0ms | 96 |
| 50 | **39.5ms** | 0.8ms | 25 |
| 100 | **72ms** | 0.7ms | 14 |

Per-document cost decreases with batch size — parallel sub-batch scoring saturates all CPU cores as the batch grows.

### Two-Stage Pipeline Performance

<p align="center">
  <img src="assets/pipeline-chart.svg" alt="Pipeline comparison chart" width="700">
</p>

### Why Flash-Rerank Beats GPU-Accelerated APIs on CPU

For models under ~100M parameters, CPU INT8 with AVX-512 + parallel workers **outperforms GPU** because:

| Factor | CPU (Flash-Rerank) | GPU (competitors) |
|--------|-------------------|-------------------|
| Model fits in cache | L2/L3 cache (~30MB) holds entire model | VRAM access adds latency |
| Kernel overhead | None — direct computation | GPU kernel launch costs 2-5ms |
| Data transfer | Zero — model lives in CPU memory | Host→Device copies add latency |
| Quantization | INT8 on AVX-512 is extremely efficient | FP16/FP32 wastes precision headroom |
| Parallelism | N workers × M threads = all cores busy | Single model instance, fixed parallelism |

---

## Quick Start

### Python

```bash
pip install flash-rerank
```

```python
import flash_rerank

# Load model (auto-downloads from HuggingFace Hub on first use)
reranker = flash_rerank.load("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Rerank documents by semantic relevance
results = reranker.rerank("distributed systems at scale", [
    "Designing Data-Intensive Applications covers distributed systems",
    "A cookbook with Italian recipes",
    "Raft consensus algorithm for fault-tolerant systems",
    "Introduction to watercolor painting techniques",
    "MapReduce: simplified data processing on large clusters",
], top_k=3)

for index, score in results:
    print(f"  doc {index}: {score:.4f}")
```

### Rust

```bash
cargo add flash_rerank
```

```rust
use flash_rerank::models::ModelRegistry;
use flash_rerank::engine::ort_backend::OrtScorer;
use flash_rerank::engine::Scorer;
use flash_rerank::ModelConfig;

// Load from HuggingFace cache
let cache_dir = dirs::home_dir().unwrap().join(".cache/huggingface/hub");
let registry = ModelRegistry::new(cache_dir);
let model_dir = registry.load("cross-encoder/ms-marco-MiniLM-L-6-v2")?;

// Score documents
let scorer = OrtScorer::new(ModelConfig::default(), &model_dir)?;
let results = scorer.score("query", &docs)?;
// Results are sorted by descending score, each with index, score in [0.0, 1.0]
```

For maximum throughput, use `ParallelScorer` instead of `OrtScorer`:

```rust
use flash_rerank::engine::parallel::ParallelScorer;

let scorer = ParallelScorer::new(ModelConfig::default(), &model_dir, None)?;
// Splits batch across N worker threads — 1.7x faster than single-session
let results = scorer.score("query", &docs)?;
```

### CLI

```bash
cargo install flash-rerank-cli
```

```bash
# Download a model from HuggingFace Hub
flash-rerank download --model cross-encoder/ms-marco-MiniLM-L-6-v2

# Benchmark reranking latency
flash-rerank bench --model cross-encoder/ms-marco-MiniLM-L-6-v2 --documents 100

# Start the HTTP server
flash-rerank serve --model cross-encoder/ms-marco-MiniLM-L-6-v2 --port 8080

# Manage cached models
flash-rerank models list

# Generate shell completions
flash-rerank completions bash > ~/.bash_completion.d/flash-rerank
```

### WASM / JavaScript

```bash
npm install flash-rerank-wasm
```

```javascript
import init, { load_model, rerank } from 'flash-rerank-wasm';

await init();

// Load model bytes and tokenizer (fetch from your server or CDN)
const modelBytes = await fetch('/models/minilm-int8.onnx').then(r => r.arrayBuffer());
const tokenizerJson = await fetch('/models/tokenizer.json').then(r => r.text());
load_model(new Uint8Array(modelBytes), tokenizerJson);

// Rerank entirely client-side — documents never leave the browser
const results = rerank("machine learning", ["ML is a subset of AI", "Weather is sunny"], 5);
console.log(results); // [{index: 0, score: 0.95}, ...]
```

---

## Two-Stage Pipeline — BM25-Turbo + Flash-Rerank

The direct sequel to [**BM25-Turbo**](https://github.com/TheSauceSuite/BM25-Turbo-Rust-Python-WASM-CLI). Together they form the fastest end-to-end retrieval + reranking pipeline in the world.

```
┌──────────────────────────────────────────────────────────────────┐
│  User Query: "distributed systems at scale"                      │
│                                                                  │
│  Stage 1: BM25-Turbo (8.6ms)                                    │
│  ├── Searches 8.8 MILLION documents via precomputed BM25         │
│  ├── Returns top 100 keyword matches                             │
│  └── No math at query time — sparse vector lookup                │
│                                                                  │
│  Stage 2: Flash-Rerank (72ms)                                    │
│  ├── Cross-encoder scores each (query, document) pair            │
│  ├── INT8 quantized inference on CPU                             │
│  ├── Parallel sub-batch across 8 worker threads                  │
│  └── Returns top 5 semantically ranked results                   │
│                                                                  │
│  Total: 80ms — Search millions, get the best 5.                  │
└──────────────────────────────────────────────────────────────────┘
```

### Installation

```bash
pip install flash-rerank bm25-turbo
```

### Usage

```python
import flash_rerank
import bm25_turbo

# Build BM25 index (one-time, ~60s for 8.8M docs)
index = bm25_turbo.build_index(corpus, method="lucene")

# Load reranker
reranker = flash_rerank.load("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Two-stage pipeline: keyword retrieval → semantic reranking
pipeline = flash_rerank.Pipeline(index, reranker)
results = pipeline.search("distributed systems at scale", top_k=5, retrieve=100)
# 80ms end-to-end: BM25 retrieval (8.6ms) + neural rerank (72ms)
```

### Hybrid RRF Fusion

Combine BM25 keyword scores with vector similarity and neural reranking via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf):

```python
results = reranker.rerank_hybrid(
    query="distributed systems at scale",
    documents=candidates,
    bm25_scores=bm25_results,
    alpha=0.6,  # weight toward neural scores (default 0.5)
)
```

### Why This Pipeline Wins

| Approach | Latency | Accuracy | Cost |
|----------|---------|----------|------|
| **BM25-Turbo + Flash-Rerank** | **80ms** | High (neural reranking) | **Free** (local) |
| Elasticsearch + Cohere Rerank | ~700ms | High | $1/1000 queries |
| Vector DB + Jina Rerank | ~300ms | High | $0.80/1000 queries |
| Pure vector search (Qdrant/Pinecone) | ~50ms | Medium (no reranking) | $0.05+/query |
| BM25 only (no reranking) | 8.6ms | Low-medium | Free |

Flash-Rerank + BM25-Turbo gives you neural-level accuracy at BM25-level latency and zero marginal cost.

---

## Requirements

- **Rust 1.85+** (edition 2024) — for building from source or `cargo install`
- **Python 3.9+** — for `pip install flash-rerank`
- **NVIDIA GPU** (optional) — enable with `cargo build --features cuda,tensorrt`

No GPU drivers, CUDA toolkit, or cloud API keys required for CPU inference.

## Installation

| Platform | Command | Package |
|----------|---------|---------|
| **Rust library** | `cargo add flash_rerank` | [crates.io/crates/flash_rerank](https://crates.io/crates/flash_rerank) |
| **CLI binary** | `cargo install flash-rerank-cli` | [crates.io/crates/flash-rerank-cli](https://crates.io/crates/flash-rerank-cli) |
| **Python** | `pip install flash-rerank` | [pypi.org/project/flash-rerank](https://pypi.org/project/flash-rerank/) |
| **WASM / npm** | `npm install flash-rerank-wasm` | [npmjs.com/package/flash-rerank-wasm](https://www.npmjs.com/package/flash-rerank-wasm) |
| **Server** | `cargo add flash-rerank-server` | [crates.io/crates/flash-rerank-server](https://crates.io/crates/flash-rerank-server) |

---

## Features

### Inference Engine
- **Parallel sub-batch inference** — N workers x M threads saturate all CPU cores
- **Auto INT8 model selection** — quantized ONNX models auto-selected for CPU (2x speedup)
- **GPU acceleration** — CUDA and TensorRT execution providers (optional feature)
- **Dynamic model input detection** — supports both 2-input (BGE) and 3-input (MiniLM) models
- **Memory-mapped model loading** — cold start in microseconds via `memmap2`

### Scoring & Fusion
- **Score calibration** — sigmoid normalization + custom Platt scaling (0.0-1.0 scores)
- **Hybrid RRF fusion** — combine BM25, vector, and neural scores with configurable weights
- **Cascade reranking** — fast model filters, accurate model refines uncertain results
- **ColBERT late interaction** — MaxSim scoring with LRU embedding cache

### Server & Operations
- **Dynamic batching** — SLA-aware request grouping for GPU throughput
- **Multi-GPU scaling** — least-loaded routing across available GPUs
- **A/B model comparison** — deterministic traffic splitting with per-variant metrics
- **Canary deployment** — gradual rollout (1% → 5% → 25% → 100%) with auto-rollback
- **Score drift detection** — KL divergence monitoring for distribution shifts
- **OpenTelemetry tracing** — behind `telemetry` feature flag, zero-cost when disabled

### Developer Experience
- **Async Python** — `arerank()` releases the GIL and runs on asyncio executors
- **WASM browser reranking** — client-side inference via tract-onnx (documents never leave the device)
- **BEIR benchmarking suite** — accuracy and latency benchmarks against standard IR datasets
- **235+ tests** — property-based (proptest), snapshot (insta), integration, stress, CLI

---

## Model Zoo

| Model | Params | Latency (100 docs) | Best For |
|-------|--------|-------------------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | **72ms** (CPU INT8) | Speed — fastest inference |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | **165ms** (CPU INT8) | Speed + accuracy balance |
| `BAAI/bge-reranker-v2-m3` | 568M | ~7s (CPU) | Multilingual (100+ languages) |
| `Qwen3-Reranker-0.6B` | 600M | — | Most accurate small model |

Any ONNX cross-encoder model from HuggingFace Hub works with Flash-Rerank. The engine auto-detects model inputs and selects the best available quantized variant.

```bash
# Download any model
flash-rerank download --model cross-encoder/ms-marco-MiniLM-L-6-v2

# List cached models
flash-rerank models list

# Benchmark a model
flash-rerank bench --model cross-encoder/ms-marco-MiniLM-L-6-v2 --documents 100
```

---

## HTTP Server

### Quick Start

```bash
# Install and start
cargo install flash-rerank-cli
flash-rerank download --model cross-encoder/ms-marco-MiniLM-L-6-v2
flash-rerank serve --model cross-encoder/ms-marco-MiniLM-L-6-v2 --port 8080
```

### API

```bash
# Rerank documents
curl -X POST http://localhost:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "documents": ["ML is a subset of AI", "The weather is sunny", "Neural networks learn patterns"],
    "top_k": 2
  }'

# Health check
curl http://localhost:8080/health

# Prometheus metrics
curl http://localhost:8080/metrics
```

### Advanced Server Options

```bash
# Dynamic batching (group requests within 5ms window, max batch 256)
flash-rerank serve --model ms-marco-MiniLM-L-6-v2 --max-batch 256 --max-wait-ms 5

# Multi-GPU (distribute across GPUs 0, 1, 2)
flash-rerank serve --model ms-marco-MiniLM-L-6-v2 --gpus 0,1,2

# With OpenTelemetry tracing
flash-rerank serve --model ms-marco-MiniLM-L-6-v2 --otlp-endpoint http://localhost:4317
```

### Management Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rerank` | POST | Score and rank documents |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/admin/drift/status` | GET | Score drift detection status |
| `/admin/drift/reset-baseline` | POST | Reset drift baseline |
| `/admin/ab` | POST/PUT/DELETE | A/B test management |
| `/admin/ab/metrics` | GET | Per-variant comparison metrics |
| `/admin/canary/start` | POST | Start canary deployment |
| `/admin/canary/status` | GET | Canary rollout status |
| `/admin/canary/abort` | POST | Abort and rollback |

---

## Architecture

```
Flash-Rerank Workspace
├── flash_rerank/              # Core library (Scorer trait + backends)
│   ├── engine/
│   │   ├── ort_backend.rs     #   ONNX Runtime: CPU, CUDA, TensorRT
│   │   ├── parallel.rs        #   Parallel sub-batch scorer (N workers)
│   │   ├── colbert.rs         #   ColBERT MaxSim late interaction
│   │   └── tensorrt.rs        #   TensorRT engine compiler
│   ├── tokenize/              #   HuggingFace tokenizers wrapper
│   ├── calibrate/             #   Sigmoid + Platt score calibration
│   ├── fusion/                #   Reciprocal Rank Fusion (RRF)
│   ├── cascade/               #   Fast model → accurate model pipeline
│   ├── batch/                 #   Dynamic request batching
│   ├── multi_gpu/             #   Least-loaded GPU routing
│   └── models/                #   HuggingFace Hub download + cache
├── flash-rerank-server/       # HTTP server (axum)
│   ├── routes.rs              #   /rerank, /health, /metrics, /admin/*
│   ├── drift.rs               #   KL divergence score monitoring
│   ├── ab_test.rs             #   A/B traffic splitting
│   ├── canary.rs              #   Progressive rollout
│   └── telemetry.rs           #   OpenTelemetry OTLP exporter
├── flash-rerank-cli/          # CLI binary (clap)
│   ├── download.rs            #   Model download from HuggingFace
│   ├── compile.rs             #   ONNX → TensorRT compilation
│   ├── bench.rs               #   Latency + accuracy benchmarks
│   ├── serve.rs               #   Start HTTP server
│   └── models.rs              #   Cache management
├── flash-rerank-python/       # Python bindings (PyO3 + maturin)
├── flash-rerank-wasm/         # Browser WASM (tract-onnx)
├── benchmarks/                # Criterion + divan + BEIR + MSMARCO
└── examples/                  # Runnable examples
```

---

## Why Flash-Rerank is Fast

Every existing reranker (Cohere, Voyage, Jina, BGE) runs a **Python server on top of PyTorch or ONNX Runtime**. Flash-Rerank eliminates every layer of overhead:

```
┌─────────────────────────────────────────────────────────────────┐
│  Competitor Stack              │  Flash-Rerank Stack            │
│                                │                                │
│  Python HTTP server            │  Rust HTTP server (axum)       │
│  ├── Python GIL                │  ├── No GIL, no interpreter   │
│  ├── NumPy array conversion    │  ├── Zero-copy tensors        │
│  ├── PyTorch / ONNX Runtime    │  ├── ONNX Runtime (direct)    │
│  ├── FP32 inference            │  ├── INT8 auto-quantized      │
│  ├── Single session            │  ├── N parallel workers       │
│  └── GPU (required)            │  └── CPU-only (GPU optional)  │
│                                │                                │
│  Result: 188-603ms             │  Result: 72ms                  │
└─────────────────────────────────────────────────────────────────┘
```

1. **Pure Rust** — No Python interpreter or GIL in the inference hot path
2. **INT8 quantized models** — Auto-selected on CPU via AVX-512 instructions (2x speedup)
3. **Parallel sub-batch scoring** — 8 worker threads each running an independent ORT session
4. **Zero-copy tokenization** — HuggingFace `tokenizers` encodes directly into model input tensors
5. **Memory-mapped model loading** — Models load via `memmap2` in microseconds
6. **Dynamic model input detection** — Auto-detects 2-input vs 3-input models at load time

---

## Benchmarks

### Reproducing Our Numbers

```bash
git clone https://github.com/TheSauceSuite/Flash-Rerank-Rust-Python-WASM-CLI-.git
cd Flash-Rerank-Rust-Python-WASM-CLI-

# Download model
cargo run -p flash-rerank-cli --release -- download --model cross-encoder/ms-marco-MiniLM-L-6-v2

# Run competitive benchmark
cargo bench --bench competitive

# Run latency profiling
cargo run --example micro_profile --release

# Run pipeline benchmark (BM25-Turbo + Flash-Rerank)
cargo run --example pipeline_benchmark --release

# Run full BEIR evaluation
MSMARCO_PATH=/path/to/msmarco cargo bench --bench msmarco_accuracy
```

### Our Benchmark Hardware

| Component | Spec |
|-----------|------|
| CPU | Intel i9-12900H (14 cores, 20 threads) |
| GPU | NVIDIA RTX 3080 Ti Laptop (16GB VRAM) |
| RAM | 32GB DDR5 |
| OS | Windows 11 |
| Rust | 1.93.1 |
| ONNX Runtime | 2.0.0-rc.12 |

On a **desktop RTX 3090 Ti** with TensorRT INT8, the target is **<20ms for 100 documents** — 10x faster than Jina's published benchmark.

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Build everything
cargo build --workspace

# Run tests (235+ tests)
cargo test --workspace --exclude flash-rerank-python --exclude flash-rerank-wasm

# Lint
cargo clippy --workspace --exclude flash-rerank-python --exclude flash-rerank-wasm -- -D warnings

# Format
cargo fmt --all

# Build Python wheel
cd flash-rerank-python && maturin develop --release

# Build WASM
cd flash-rerank-wasm && wasm-pack build --target web
```

---

## Roadmap

- [ ] TensorRT INT8 engine compilation with auto-calibration
- [ ] Pre-compiled engine distribution via HuggingFace Hub
- [ ] WebGPU acceleration for browser inference
- [ ] Early-exit transformer inference for adaptive compute
- [ ] Multi-modal reranking (text + images)
- [ ] Listwise reranking (RankGPT-style)

---

## License

**v0.2.0+** is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

- **Individuals and open-source projects**: Free to use under AGPL-3.0 terms.
- **Enterprises and commercial use**: A commercial license is available at **[alessandrobenigni.com](https://alessandrobenigni.com)** — use Flash-Rerank in proprietary software without the AGPL copyleft requirement. See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for details.

> **Legacy**: v0.1.x was released under MIT/Apache-2.0. Those versions remain under their original terms but are no longer maintained.

---

<p align="center">
  <strong>Built by <a href="https://github.com/TheSauceSuite">The Sauce Suite</a></strong><br>
  <sub>Part of the BM25-Turbo ecosystem — the fastest search stack in the world.</sub>
</p>
