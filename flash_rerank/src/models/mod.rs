//! Model download, caching, and lifecycle management.
//!
//! Provides [`ModelRegistry`] for downloading models from HuggingFace Hub,
//! loading them from the local cache, listing cached models, and deleting
//! cached entries. Uses the `hf-hub` crate for download and cache layout.

use std::path::PathBuf;

use crate::Result;
use crate::types::CacheMetadata;

/// Required files for a reranking model. First found wins per group.
/// Each entry is a group of alternative paths to try in order.
const REQUIRED_FILE_GROUPS: &[&[&str]] = &[
    &["model.onnx", "onnx/model.onnx"], // ONNX model
    &["tokenizer.json"],                // Tokenizer
];

/// Optional files that are downloaded if present.
const OPTIONAL_FILES: &[&str] = &[
    "config.json",
    "calibration.json",
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
];

/// Registry for downloading and loading models from HuggingFace Hub.
pub struct ModelRegistry {
    cache_dir: PathBuf,
}

impl ModelRegistry {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Return the cache directory for this registry.
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }

    /// Download a model from HuggingFace Hub and return the local path.
    ///
    /// Uses `hf_hub` which handles caching and partial downloads natively.
    /// Required files: model.onnx, tokenizer.json.
    /// Optional files: config.json, calibration.json.
    pub async fn download(&self, model_id: &str) -> Result<PathBuf> {
        let api =
            hf_hub::api::tokio::Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
        let repo = api.model(model_id.to_string());

        // Download required files — try each alternative path in order
        let mut model_dir: Option<PathBuf> = None;
        for group in REQUIRED_FILE_GROUPS {
            let mut downloaded = false;
            for filename in *group {
                match repo.get(filename).await {
                    Ok(file_path) => {
                        if model_dir.is_none() {
                            // Derive snapshot root from file path — walk up past any subdirs
                            let mut dir = file_path.parent().map(|p| p.to_path_buf());
                            // If the file was in onnx/ subdir, go up one more level
                            if filename.contains('/') {
                                dir = dir.and_then(|d| d.parent().map(|p| p.to_path_buf()));
                            }
                            model_dir = dir;
                        }
                        downloaded = true;
                        break;
                    }
                    Err(_) => continue, // Try next alternative
                }
            }
            if !downloaded {
                return Err(crate::Error::Download(format!(
                    "Failed to download required file (tried: {})",
                    group.join(", ")
                )));
            }
        }

        // Download optional files (non-fatal if missing)
        for filename in OPTIONAL_FILES {
            let _ = repo.get(filename).await;
        }

        let model_dir = model_dir.ok_or_else(|| {
            crate::Error::Download("Could not determine model directory".to_string())
        })?;

        // Write cache metadata
        let metadata = CacheMetadata {
            model_id: model_id.to_string(),
            source_url: format!("https://huggingface.co/{model_id}"),
            sha256: String::new(), // hf-hub verifies integrity internally via etags
            download_date: chrono_now_iso(),
            file_size_bytes: dir_size(&model_dir),
            precision: None,
            gpu_arch: None,
            trt_version: None,
            platt_a: None,
            platt_b: None,
        };

        let meta_path = model_dir.join("flash_rerank_cache.json");
        let meta_json = serde_json::to_string_pretty(&metadata)?;
        tokio::fs::write(&meta_path, meta_json)
            .await
            .map_err(|e| crate::Error::Cache(format!("Failed to write cache metadata: {e}")))?;

        tracing::info!(model_id, path = %model_dir.display(), "Model downloaded successfully");
        Ok(model_dir)
    }

    /// Load a model from the local HuggingFace Hub cache, returning its directory path.
    ///
    /// Resolves the model_id to the hf-hub cache layout. Returns an error with
    /// a download hint if the model is not found locally.
    pub fn load(&self, model_id: &str) -> Result<PathBuf> {
        // hf-hub stores models under: <cache_dir>/models--<org>--<name>/snapshots/<rev>/
        let sanitized = model_id.replace('/', "--");
        let model_base = self.cache_dir.join(format!("models--{sanitized}"));

        if !model_base.exists() {
            return Err(crate::Error::Model(format!(
                "Model '{model_id}' not found in cache. Run `flash-rerank download --model {model_id}` first."
            )));
        }

        // Find the latest snapshot
        let snapshots_dir = model_base.join("snapshots");
        if !snapshots_dir.exists() {
            return Err(crate::Error::Model(format!(
                "Model '{model_id}' cache is corrupted (no snapshots directory)."
            )));
        }

        let mut latest_snapshot: Option<PathBuf> = None;
        let entries = std::fs::read_dir(&snapshots_dir)?;
        for entry in entries {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                latest_snapshot = Some(entry.path());
            }
        }

        let snapshot_dir = latest_snapshot.ok_or_else(|| {
            crate::Error::Model(format!(
                "Model '{model_id}' cache has no snapshot revisions."
            ))
        })?;

        // Verify required files exist — check all alternative paths per group
        for group in REQUIRED_FILE_GROUPS {
            let found = group
                .iter()
                .any(|filename| snapshot_dir.join(filename).exists());
            if !found {
                return Err(crate::Error::Model(format!(
                    "Model '{model_id}' is missing required file (expected one of: {}). Re-download with `flash-rerank download --model {model_id}`.",
                    group.join(", ")
                )));
            }
        }

        tracing::info!(model_id, path = %snapshot_dir.display(), "Model loaded from cache");
        Ok(snapshot_dir)
    }

    /// List all models in the local cache with their metadata.
    pub fn list(&self) -> Result<Vec<CacheMetadata>> {
        let mut results = Vec::new();

        if !self.cache_dir.exists() {
            return Ok(results);
        }

        let entries = std::fs::read_dir(&self.cache_dir)?;
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.starts_with("models--") || !entry.file_type()?.is_dir() {
                continue;
            }

            // Look for cache metadata in snapshots
            let snapshots_dir = entry.path().join("snapshots");
            if !snapshots_dir.exists() {
                continue;
            }

            if let Ok(snap_entries) = std::fs::read_dir(&snapshots_dir) {
                for snap_entry in snap_entries.flatten() {
                    let meta_path = snap_entry.path().join("flash_rerank_cache.json");
                    if meta_path.exists() {
                        if let Ok(content) = std::fs::read_to_string(&meta_path) {
                            if let Ok(metadata) = serde_json::from_str::<CacheMetadata>(&content) {
                                results.push(metadata);
                            }
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Delete all cached files for a given model.
    pub fn delete(&self, model_id: &str) -> Result<()> {
        let sanitized = model_id.replace('/', "--");
        let model_base = self.cache_dir.join(format!("models--{sanitized}"));

        if !model_base.exists() {
            return Err(crate::Error::Cache(format!(
                "Model '{model_id}' not found in cache."
            )));
        }

        std::fs::remove_dir_all(&model_base)?;
        tracing::info!(model_id, "Model deleted from cache");
        Ok(())
    }
}

/// Get an ISO-8601 timestamp string without a chrono dependency.
fn chrono_now_iso() -> String {
    // Use std::time for a basic timestamp
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s since epoch", duration.as_secs())
}

/// Compute total size of a directory.
fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_nonexistent_returns_error() {
        let tmp = std::env::temp_dir().join("flash_rerank_test_empty_cache");
        let _ = std::fs::create_dir_all(&tmp);
        let registry = ModelRegistry::new(tmp.clone());
        let result = registry.load("nonexistent/model");
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::Error::Model(msg) => {
                assert!(msg.contains("not found in cache"));
            }
            other => panic!("Expected Error::Model, got: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_list_empty_cache() {
        let tmp = std::env::temp_dir().join("flash_rerank_test_list_empty");
        let _ = std::fs::create_dir_all(&tmp);
        let registry = ModelRegistry::new(tmp.clone());
        let models = registry.list().unwrap();
        assert!(models.is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_delete_nonexistent_returns_error() {
        let tmp = std::env::temp_dir().join("flash_rerank_test_delete_nonexistent");
        let _ = std::fs::create_dir_all(&tmp);
        let registry = ModelRegistry::new(tmp.clone());
        let result = registry.delete("nonexistent/model");
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // --- Additional model registry tests ---

    #[test]
    fn test_load_corrupted_cache_no_snapshots_dir() {
        let tmp = std::env::temp_dir().join("flash_rerank_test_corrupted_cache");
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);

        // Create model directory but without snapshots subdirectory
        let model_base = tmp.join("models--test--model");
        std::fs::create_dir_all(&model_base).unwrap();

        let registry = ModelRegistry::new(tmp.clone());
        let result = registry.load("test/model");
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::Error::Model(msg) => {
                assert!(
                    msg.contains("corrupted"),
                    "Error should mention corruption: {msg}"
                );
            }
            other => panic!("Expected Model error, got: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_load_empty_snapshots_dir() {
        let tmp = std::env::temp_dir().join("flash_rerank_test_empty_snapshots");
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);

        let snapshots = tmp.join("models--test--model").join("snapshots");
        std::fs::create_dir_all(&snapshots).unwrap();

        let registry = ModelRegistry::new(tmp.clone());
        let result = registry.load("test/model");
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::Error::Model(msg) => {
                assert!(
                    msg.contains("no snapshot"),
                    "Error should mention missing snapshots: {msg}"
                );
            }
            other => panic!("Expected Model error, got: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_load_missing_required_file() {
        let tmp = std::env::temp_dir().join("flash_rerank_test_missing_required");
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);

        let snap_dir = tmp
            .join("models--test--model")
            .join("snapshots")
            .join("abc123");
        std::fs::create_dir_all(&snap_dir).unwrap();
        // Only create tokenizer.json, not model.onnx
        std::fs::write(snap_dir.join("tokenizer.json"), "{}").unwrap();

        let registry = ModelRegistry::new(tmp.clone());
        let result = registry.load("test/model");
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::Error::Model(msg) => {
                assert!(
                    msg.contains("model.onnx"),
                    "Error should mention missing file: {msg}"
                );
            }
            other => panic!("Expected Model error, got: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_list_with_valid_metadata() {
        let tmp = std::env::temp_dir().join("flash_rerank_test_list_metadata");
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);

        let snap_dir = tmp
            .join("models--org--mymodel")
            .join("snapshots")
            .join("rev1");
        std::fs::create_dir_all(&snap_dir).unwrap();

        let meta = CacheMetadata {
            model_id: "org/mymodel".to_string(),
            source_url: "https://huggingface.co/org/mymodel".to_string(),
            sha256: "abc".to_string(),
            download_date: "2026-01-01".to_string(),
            file_size_bytes: 1000,
            precision: None,
            gpu_arch: None,
            trt_version: None,
            platt_a: None,
            platt_b: None,
        };
        let meta_json = serde_json::to_string_pretty(&meta).unwrap();
        std::fs::write(snap_dir.join("flash_rerank_cache.json"), meta_json).unwrap();

        let registry = ModelRegistry::new(tmp.clone());
        let models = registry.list().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].model_id, "org/mymodel");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_list_ignores_non_model_dirs() {
        let tmp = std::env::temp_dir().join("flash_rerank_test_ignore_nonmodel");
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);

        // Create a non-model directory and a regular file
        std::fs::create_dir_all(tmp.join("random_dir")).unwrap();
        std::fs::write(tmp.join("somefile.txt"), "hello").unwrap();

        let registry = ModelRegistry::new(tmp.clone());
        let models = registry.list().unwrap();
        assert!(models.is_empty(), "Non-model dirs should be ignored");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_delete_existing_model() {
        let tmp = std::env::temp_dir().join("flash_rerank_test_delete_existing");
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);

        let model_base = tmp.join("models--org--deleteme");
        std::fs::create_dir_all(model_base.join("snapshots").join("rev1")).unwrap();

        let registry = ModelRegistry::new(tmp.clone());
        let result = registry.delete("org/deleteme");
        assert!(result.is_ok(), "Deleting existing model should succeed");
        assert!(!model_base.exists(), "Model directory should be removed");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_cache_dir_getter() {
        let path = std::path::PathBuf::from("/some/cache/dir");
        let registry = ModelRegistry::new(path.clone());
        assert_eq!(registry.cache_dir(), &path);
    }

    #[test]
    fn test_dir_size_empty_directory() {
        let tmp = std::env::temp_dir().join("flash_rerank_test_dir_size_empty");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        assert_eq!(dir_size(&tmp), 0);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_chrono_format() {
        let ts = chrono_now_iso();
        assert!(
            ts.contains("since epoch"),
            "Timestamp should contain 'since epoch': {ts}"
        );
        // Should parse the seconds portion
        let parts: Vec<&str> = ts.split('s').collect();
        assert!(!parts.is_empty());
        let secs: u64 = parts[0].trim().parse().expect("Should parse as u64");
        assert!(secs > 0, "Seconds since epoch should be positive");
    }
}
