# Contributing to Flash-Rerank

Thank you for your interest in contributing to Flash-Rerank! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/Flash-Rerank-Rust-Python-WASM-CLI.git
   cd Flash-Rerank-Rust-Python-WASM-CLI
   ```
3. **Create a branch** for your change:
   ```bash
   git checkout -b feature/my-change
   ```

## Development Setup

You need Rust 1.85+ (edition 2024). Install via [rustup](https://rustup.rs/).

```bash
# Build the workspace (excludes Python and WASM bindings)
cargo build --workspace --exclude flash-rerank-python --exclude flash-rerank-wasm

# Run all tests
cargo test --workspace --exclude flash-rerank-python --exclude flash-rerank-wasm

# Run clippy lints (must pass with zero warnings)
cargo clippy --workspace --exclude flash-rerank-python --exclude flash-rerank-wasm -- -D warnings

# Check formatting
cargo fmt --all -- --check
```

### Python Bindings

Python bindings require [maturin](https://www.maturin.rs/) and Python 3.9+:

```bash
cd flash-rerank-python
pip install maturin
maturin develop
```

### WASM Bindings

WASM bindings require [wasm-pack](https://rustwasm.github.io/wasm-pack/):

```bash
cd flash-rerank-wasm
wasm-pack build --target web
```

## Making Changes

1. Write your code following the existing patterns in the codebase.
2. Add tests for new functionality.
3. Ensure all checks pass:
   - `cargo test` -- all tests green
   - `cargo clippy -- -D warnings` -- no lint warnings
   - `cargo fmt --all -- --check` -- formatting consistent
4. Write a clear commit message describing *why* the change was made.

## Submitting a Pull Request

1. **Push** your branch to your fork.
2. Open a **Pull Request** against the `main` branch.
3. Describe your changes and link any related issues.
4. Ensure CI passes on your PR.

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests.
- Include steps to reproduce for bugs, with OS, Rust version, and hardware (CPU/GPU) details.

## License

By contributing, you agree that your contributions will be licensed under the MIT OR Apache-2.0 dual license.
