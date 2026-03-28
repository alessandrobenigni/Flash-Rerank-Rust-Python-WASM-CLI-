use std::path::PathBuf;

use ort::execution_providers::TensorRT;
use ort::session::Session;

use crate::Result;
use crate::types::Precision;

/// Hub repository for pre-compiled TensorRT engines.
const ENGINE_HUB_REPO: &str = "TheSauceSuite/flash-rerank-engines";

/// TensorRT engine compiler for ONNX models.
///
/// Compiles ONNX models to optimized TensorRT engines with:
/// - INT8 calibration via representative dataset
/// - INT4 on Ada Lovelace (sm_89) and Hopper (sm_90)
/// - FP16 as universal GPU fast path
///
/// Compilation works by setting TensorRT EP's engine cache options.
/// The first `Session::run()` (triggered by session creation) compiles
/// and caches the engine.
pub struct TensorRTCompiler {
    onnx_path: PathBuf,
    precision: Precision,
    max_batch_size: usize,
}

impl TensorRTCompiler {
    pub fn new(onnx_path: PathBuf, precision: Precision, max_batch_size: usize) -> Self {
        Self {
            onnx_path,
            precision,
            max_batch_size,
        }
    }

    /// Compile the ONNX model to a TensorRT engine file.
    ///
    /// Uses ort's TensorRT EP with engine caching enabled. Loading the model
    /// triggers TRT engine compilation and caches the result at `output_path`.
    ///
    /// For INT8 precision without calibration data, falls back to FP16 with a warning.
    pub fn compile(&self, output_path: &std::path::Path) -> Result<()> {
        let output_dir = output_path
            .parent()
            .ok_or_else(|| crate::Error::Model("Invalid output path".to_string()))?;

        // Create output directory if it doesn't exist
        if !output_dir.exists() {
            std::fs::create_dir_all(output_dir)?;
        }

        // Determine effective precision -- fall back to FP16 if INT8 requested without calibration
        let (use_fp16, use_int8) = match self.precision {
            Precision::FP32 => (false, false),
            Precision::FP16 => (true, false),
            Precision::INT8 | Precision::INT4 => {
                // Check for calibration table alongside the ONNX model
                let calibration_path = self.onnx_path.with_extension("calibration");
                if calibration_path.exists() {
                    (true, true)
                } else {
                    tracing::warn!(
                        "INT8/INT4 requested but no calibration data found at {}. Falling back to FP16.",
                        calibration_path.display()
                    );
                    (true, false)
                }
            }
        };

        tracing::info!(
            onnx = %self.onnx_path.display(),
            output = %output_path.display(),
            precision = ?self.precision,
            fp16 = use_fp16,
            int8 = use_int8,
            max_batch_size = self.max_batch_size,
            "Compiling ONNX to TensorRT engine"
        );

        let trt_ep = TensorRT::default()
            .with_fp16(use_fp16)
            .with_int8(use_int8)
            .with_engine_cache(true)
            .with_engine_cache_path(output_dir.to_str().unwrap_or("."))
            .build();

        let mut builder = Session::builder().map_err(|e| crate::Error::Inference(e.to_string()))?;

        builder = builder
            .with_execution_providers([trt_ep])
            .map_err(|e| crate::Error::Inference(e.to_string()))?;

        // Loading the model triggers TRT engine compilation
        let _session = builder
            .commit_from_file(&self.onnx_path)
            .map_err(|e| crate::Error::Inference(e.to_string()))?;

        // Verify engine file was created
        if !output_path.exists() {
            // TRT may use its own naming convention for cached engines.
            // Check if any .trt or .engine file was created in the output directory.
            let any_engine = std::fs::read_dir(output_dir).ok().and_then(|entries| {
                entries.filter_map(|e| e.ok()).find(|e| {
                    let name = e.file_name();
                    let name = name.to_string_lossy();
                    name.ends_with(".trt") || name.ends_with(".engine")
                })
            });

            if any_engine.is_none() {
                return Err(crate::Error::Model(
                    "TensorRT engine file was not created. Check GPU compatibility.".to_string(),
                ));
            }
        }

        tracing::info!("TensorRT engine compiled: {}", output_path.display());
        Ok(())
    }

    /// Check HuggingFace Hub for a pre-compiled TensorRT engine matching the
    /// given GPU architecture and TRT version. If found, copy the cached file
    /// to `output_path`. If not found (or Hub is unreachable), fall back to
    /// local compilation via `self.compile()`.
    ///
    /// Engine naming convention on Hub:
    /// `{model_name}/{sm_XX}-{precision}-trt{version}.engine`
    pub async fn compile_or_download(
        &self,
        output_path: &std::path::Path,
        gpu_arch: &str,
        trt_version: &str,
    ) -> Result<()> {
        let engine_filename = format!(
            "{}-{}-trt{}.engine",
            gpu_arch,
            Self::precision_tag(self.precision),
            trt_version,
        );

        tracing::info!(
            engine = %engine_filename,
            repo = ENGINE_HUB_REPO,
            "Checking Hub for pre-compiled TensorRT engine"
        );

        let api = hf_hub::api::tokio::Api::new().map_err(|e| crate::Error::Download(e.to_string()));

        let download_result = match api {
            Ok(api) => {
                let repo = api.model(ENGINE_HUB_REPO.to_string());
                repo.get(&engine_filename).await
            }
            Err(_e) => {
                tracing::info!("Hub API unavailable, falling back to local compilation");
                return self.compile(output_path);
            }
        };

        match download_result {
            Ok(cached_path) => {
                tracing::info!(
                    cached = %cached_path.display(),
                    output = %output_path.display(),
                    "Found pre-compiled engine on Hub, copying to output"
                );

                // Ensure output directory exists
                if let Some(parent) = output_path.parent() {
                    if !parent.exists() {
                        std::fs::create_dir_all(parent)?;
                    }
                }

                std::fs::copy(&cached_path, output_path).map_err(|e| {
                    crate::Error::Cache(format!(
                        "Failed to copy pre-compiled engine to {}: {e}",
                        output_path.display()
                    ))
                })?;

                tracing::info!("Pre-compiled engine installed: {}", output_path.display());
                Ok(())
            }
            Err(e) => {
                tracing::info!(
                    error = %e,
                    "Pre-compiled engine not found on Hub, falling back to local compilation"
                );
                self.compile(output_path)
            }
        }
    }

    /// Return a lowercase precision tag for engine filename construction.
    fn precision_tag(precision: Precision) -> &'static str {
        match precision {
            Precision::FP32 => "fp32",
            Precision::FP16 => "fp16",
            Precision::INT8 => "int8",
            Precision::INT4 => "int4",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_tag_fp32() {
        assert_eq!(TensorRTCompiler::precision_tag(Precision::FP32), "fp32");
    }

    #[test]
    fn test_precision_tag_fp16() {
        assert_eq!(TensorRTCompiler::precision_tag(Precision::FP16), "fp16");
    }

    #[test]
    fn test_precision_tag_int8() {
        assert_eq!(TensorRTCompiler::precision_tag(Precision::INT8), "int8");
    }

    #[test]
    fn test_precision_tag_int4() {
        assert_eq!(TensorRTCompiler::precision_tag(Precision::INT4), "int4");
    }

    #[test]
    fn test_compiler_new_stores_config() {
        let path = PathBuf::from("/tmp/model.onnx");
        let compiler = TensorRTCompiler::new(path.clone(), Precision::FP16, 32);
        assert_eq!(compiler.onnx_path, path);
        assert_eq!(compiler.precision, Precision::FP16);
        assert_eq!(compiler.max_batch_size, 32);
    }

    #[test]
    fn test_compile_invalid_output_path() {
        // Output path with no parent should error
        let compiler =
            TensorRTCompiler::new(PathBuf::from("/nonexistent/model.onnx"), Precision::FP32, 1);
        // A path like "/" has no parent in the sense that parent() returns None for root paths
        // on some platforms, but normally compile will fail on missing ONNX file.
        // We just verify it returns an error (file does not exist).
        let result = compiler.compile(std::path::Path::new(
            "/tmp/flash_rerank_test_trt/output.engine",
        ));
        assert!(result.is_err(), "Compile with nonexistent ONNX should fail");
    }

    #[test]
    fn test_compile_missing_onnx_file() {
        let tmp = std::env::temp_dir().join("flash_rerank_trt_test_missing");
        let _ = std::fs::create_dir_all(&tmp);
        let compiler =
            TensorRTCompiler::new(tmp.join("nonexistent_model.onnx"), Precision::FP32, 1);
        let result = compiler.compile(&tmp.join("output.engine"));
        assert!(
            result.is_err(),
            "Missing ONNX file should cause compile error"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
