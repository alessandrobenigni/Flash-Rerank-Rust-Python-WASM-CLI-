# Contributing to Flash-Rerank

Thank you for your interest in contributing to Flash-Rerank! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/Flash-Rerank-Rust-Python-WASM-CLI-.git
   cd Flash-Rerank-Rust-Python-WASM-CLI-
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
3. **Sign off your commits (DCO)** — Every commit must include a `Signed-off-by` line certifying you wrote the code and have the right to submit it. Use `git commit -s` to add it automatically:
   ```bash
   git commit -s -m "Add feature X"
   # Produces: Signed-off-by: Your Name <your@email.com>
   ```
   The DCO bot will block your PR if any commit is missing the sign-off.
4. **Sign the CLA** — A bot will comment on your PR asking you to sign the [Contributor License Agreement](CLA.md). Comment with the exact phrase: `I have read the CLA Document and I hereby sign the CLA`. The PR cannot be merged until the CLA is signed.
5. Describe your changes and link any related issues.
6. Ensure CI passes on your PR.

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests.
- Include steps to reproduce for bugs, with OS, Rust version, and hardware (CPU/GPU) details.

## Contributor License Agreement (CLA)

All contributors must sign the [CLA](CLA.md) before their first Pull Request can be merged. This is required because Flash-Rerank uses a dual-licensing model (AGPL-3.0 for open-source, commercial license for enterprises). The CLA grants the maintainer the right to include your contributions under both licenses.

The signing process is fully automated via a GitHub bot — just comment on your PR and you're done.

## Developer Certificate of Origin (DCO)

All commits must be signed off with `git commit -s`, certifying that you wrote the code and have the right to submit it under the project's license. This is the [Developer Certificate of Origin](https://developercertificate.org/). The DCO bot automatically checks every commit in your PR.

If you forgot to sign off, you can amend your last commit:
```bash
git commit --amend -s --no-edit
git push --force-with-lease
```

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0-or-later license and that the maintainer may re-license contributions under commercial terms per the [CLA](CLA.md).
