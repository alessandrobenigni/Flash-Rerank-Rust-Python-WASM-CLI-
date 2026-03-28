//! Snapshot tests for configuration defaults and error display strings.
//!
//! Uses insta for yaml snapshot testing. Run `cargo insta review` to
//! approve new snapshots after intentional changes.

use insta::assert_yaml_snapshot;

use flash_rerank::Error;
use flash_rerank::fusion::FusionConfig;
use flash_rerank::types::{ModelConfig, RerankConfig};

#[test]
fn snapshot_model_config_default() {
    assert_yaml_snapshot!("model_config_default", ModelConfig::default());
}

#[test]
fn snapshot_rerank_config_default() {
    assert_yaml_snapshot!("rerank_config_default", RerankConfig::default());
}

#[test]
fn snapshot_fusion_config_default() {
    assert_yaml_snapshot!("fusion_config_default", FusionConfig::default());
}

#[test]
fn snapshot_error_display_strings() {
    let errors = vec![
        (
            "model",
            format!("{}", Error::Model("test model error".into())),
        ),
        (
            "tokenizer",
            format!("{}", Error::Tokenizer("test tokenizer error".into())),
        ),
        (
            "inference",
            format!("{}", Error::Inference("test inference error".into())),
        ),
        (
            "download",
            format!("{}", Error::Download("test download error".into())),
        ),
        (
            "cache",
            format!("{}", Error::Cache("test cache error".into())),
        ),
        (
            "config",
            format!("{}", Error::Config("test config error".into())),
        ),
        (
            "calibration",
            format!("{}", Error::Calibration("test calibration error".into())),
        ),
    ];

    let error_map: std::collections::BTreeMap<&str, &str> =
        errors.iter().map(|(k, v)| (*k, v.as_str())).collect();

    assert_yaml_snapshot!("error_display_strings", error_map);
}
