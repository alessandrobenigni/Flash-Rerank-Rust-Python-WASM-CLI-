//! CLI integration tests for flash-rerank binary.
//!
//! Uses assert_cmd to invoke the compiled binary and verify exit codes,
//! stdout/stderr content. No model files or GPU required.

use assert_cmd::Command;
use predicates::prelude::*;

fn cmd() -> Command {
    Command::cargo_bin("flash-rerank-cli").unwrap()
}

// ===== Help output tests =====

#[test]
fn cli_help_main() {
    cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Flash-Rerank CLI"));
}

#[test]
fn cli_help_compile() {
    cmd()
        .args(["compile", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Compile"));
}

#[test]
fn cli_help_bench() {
    cmd()
        .args(["bench", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("benchmark").or(predicate::str::contains("Bench")));
}

#[test]
fn cli_help_serve() {
    cmd()
        .args(["serve", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("server").or(predicate::str::contains("Serve")));
}

#[test]
fn cli_help_download() {
    cmd()
        .args(["download", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Download").or(predicate::str::contains("download")));
}

#[test]
fn cli_help_models() {
    cmd()
        .args(["models", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("model").or(predicate::str::contains("Model")));
}

#[test]
fn cli_help_completions() {
    cmd()
        .args(["completions", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("completions").or(predicate::str::contains("shell")));
}

// ===== Completions generation tests =====

#[test]
fn cli_completions_bash() {
    cmd()
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::contains("flash-rerank"));
}

#[test]
fn cli_completions_zsh() {
    cmd()
        .args(["completions", "zsh"])
        .assert()
        .success()
        .stdout(predicate::str::contains("flash-rerank"));
}

#[test]
fn cli_completions_fish() {
    cmd()
        .args(["completions", "fish"])
        .assert()
        .success()
        .stdout(predicate::str::contains("flash-rerank"));
}

#[test]
fn cli_completions_powershell() {
    cmd()
        .args(["completions", "powershell"])
        .assert()
        .success()
        .stdout(predicate::str::contains("flash-rerank"));
}

// ===== Error handling tests =====

#[test]
fn cli_invalid_subcommand() {
    cmd()
        .arg("nonexistent")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error").or(predicate::str::contains("unrecognized")));
}

#[test]
fn cli_no_args() {
    // Running with no subcommand should fail or show help
    cmd().assert().failure();
}

#[test]
fn cli_compile_missing_required_args() {
    cmd()
        .arg("compile")
        .assert()
        .failure()
        .stderr(predicate::str::is_empty().not());
}

#[test]
fn cli_download_missing_required_args() {
    cmd()
        .arg("download")
        .assert()
        .failure()
        .stderr(predicate::str::is_empty().not());
}

#[test]
fn cli_bench_missing_required_args() {
    cmd()
        .arg("bench")
        .assert()
        .failure()
        .stderr(predicate::str::is_empty().not());
}

// ===== Flag tests =====

#[test]
fn cli_verbose_flag_accepted() {
    // --verbose is a global flag; with no subcommand it should still fail
    // but not due to unknown flag
    cmd().arg("--verbose").assert().failure().stderr(
        predicate::str::contains("verbose").not(), // should not complain about --verbose itself
    );
}

#[test]
fn cli_debug_flag_accepted() {
    cmd().arg("--debug").assert().failure().stderr(
        predicate::str::contains("debug").not(), // should not complain about --debug itself
    );
}

#[test]
fn cli_version() {
    cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("flash-rerank"));
}
