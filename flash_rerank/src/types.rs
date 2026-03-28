use serde::{Deserialize, Serialize};

/// Precision level for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    FP32,
    FP16,
    INT8,
    INT4,
}

/// Device target for inference execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(usize),
    TensorRT(usize),
}

/// Scoring method for reranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScorerType {
    CrossEncoder,
    ColBERT,
}

/// Configuration for loading and running a reranking model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_id: String,
    pub precision: Precision,
    pub device: Device,
    pub scorer_type: ScorerType,
    pub max_length: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_id: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
            precision: Precision::FP32,
            device: Device::Cpu,
            scorer_type: ScorerType::CrossEncoder,
            max_length: 128,
        }
    }
}

/// A single reranking request: query + documents to score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankRequest {
    pub query: String,
    pub documents: Vec<String>,
    pub top_k: Option<usize>,
    pub return_documents: bool,
}

/// A single scored document in reranking results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

/// Configuration for reranking behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankConfig {
    pub batch_size: usize,
    pub max_length: usize,
    pub normalize_scores: bool,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            batch_size: 64,
            max_length: 512,
            normalize_scores: true,
        }
    }
}

/// Metadata stored alongside cached model files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub model_id: String,
    pub source_url: String,
    pub sha256: String,
    pub download_date: String,
    pub file_size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision: Option<Precision>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_arch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trt_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub platt_a: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub platt_b: Option<f64>,
}

/// Manifest of files to download for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub model_id: String,
    pub files: Vec<ModelFile>,
    pub total_size_bytes: u64,
}

/// A single file within a model's manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Default impls ---

    #[test]
    fn test_model_config_default() {
        let cfg = ModelConfig::default();
        assert_eq!(cfg.model_id, "cross-encoder/ms-marco-MiniLM-L-6-v2");
        assert_eq!(cfg.precision, Precision::FP32);
        assert_eq!(cfg.device, Device::Cpu);
        assert_eq!(cfg.scorer_type, ScorerType::CrossEncoder);
        assert_eq!(cfg.max_length, 128);
    }

    #[test]
    fn test_rerank_config_default() {
        let cfg = RerankConfig::default();
        assert_eq!(cfg.batch_size, 64);
        assert_eq!(cfg.max_length, 512);
        assert!(cfg.normalize_scores);
    }

    // --- Serde roundtrips ---

    #[test]
    fn test_precision_serde_roundtrip() {
        for variant in [
            Precision::FP32,
            Precision::FP16,
            Precision::INT8,
            Precision::INT4,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: Precision = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn test_device_serde_roundtrip() {
        for variant in [
            Device::Cpu,
            Device::Cuda(0),
            Device::Cuda(3),
            Device::TensorRT(1),
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: Device = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn test_scorer_type_serde_roundtrip() {
        for variant in [ScorerType::CrossEncoder, ScorerType::ColBERT] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: ScorerType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn test_model_config_serde_roundtrip() {
        let cfg = ModelConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_id, cfg.model_id);
        assert_eq!(back.precision, cfg.precision);
        assert_eq!(back.device, cfg.device);
        assert_eq!(back.scorer_type, cfg.scorer_type);
        assert_eq!(back.max_length, cfg.max_length);
    }

    #[test]
    fn test_rerank_request_serde_roundtrip() {
        let req = RerankRequest {
            query: "what is ML".to_string(),
            documents: vec!["doc1".to_string(), "doc2".to_string()],
            top_k: Some(5),
            return_documents: true,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: RerankRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.query, req.query);
        assert_eq!(back.documents, req.documents);
        assert_eq!(back.top_k, Some(5));
        assert!(back.return_documents);
    }

    #[test]
    fn test_rerank_result_serde_roundtrip() {
        let res = RerankResult {
            index: 3,
            score: 0.95,
            document: Some("hello".to_string()),
        };
        let json = serde_json::to_string(&res).unwrap();
        let back: RerankResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.index, 3);
        assert!((back.score - 0.95).abs() < 1e-6);
        assert_eq!(back.document, Some("hello".to_string()));
    }

    #[test]
    fn test_cache_metadata_serde_roundtrip() {
        let meta = CacheMetadata {
            model_id: "test/model".to_string(),
            source_url: "https://example.com".to_string(),
            sha256: "abc123".to_string(),
            download_date: "2026-01-01".to_string(),
            file_size_bytes: 1024,
            precision: Some(Precision::FP16),
            gpu_arch: Some("sm_89".to_string()),
            trt_version: Some("10.0".to_string()),
            platt_a: Some(-1.5),
            platt_b: Some(0.3),
        };
        let json = serde_json::to_string(&meta).unwrap();
        let back: CacheMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_id, meta.model_id);
        assert_eq!(back.precision, Some(Precision::FP16));
        assert_eq!(back.platt_a, Some(-1.5));
    }

    #[test]
    fn test_model_manifest_serde_roundtrip() {
        let manifest = ModelManifest {
            model_id: "org/model".to_string(),
            files: vec![ModelFile {
                filename: "model.onnx".to_string(),
                size_bytes: 50_000,
                sha256: "abc".to_string(),
            }],
            total_size_bytes: 50_000,
        };
        let json = serde_json::to_string(&manifest).unwrap();
        let back: ModelManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_id, manifest.model_id);
        assert_eq!(back.files.len(), 1);
        assert_eq!(back.total_size_bytes, 50_000);
    }

    // --- skip_serializing_if ---

    #[test]
    fn test_rerank_result_skip_none_document() {
        let res = RerankResult {
            index: 0,
            score: 0.5,
            document: None,
        };
        let json = serde_json::to_string(&res).unwrap();
        assert!(
            !json.contains("document"),
            "None document should be skipped"
        );
    }

    #[test]
    fn test_rerank_result_includes_some_document() {
        let res = RerankResult {
            index: 0,
            score: 0.5,
            document: Some("text".to_string()),
        };
        let json = serde_json::to_string(&res).unwrap();
        assert!(
            json.contains("document"),
            "Some document should be included"
        );
    }

    #[test]
    fn test_cache_metadata_skip_none_optional_fields() {
        let meta = CacheMetadata {
            model_id: "m".to_string(),
            source_url: "u".to_string(),
            sha256: "s".to_string(),
            download_date: "d".to_string(),
            file_size_bytes: 0,
            precision: None,
            gpu_arch: None,
            trt_version: None,
            platt_a: None,
            platt_b: None,
        };
        let json = serde_json::to_string(&meta).unwrap();
        assert!(!json.contains("precision"));
        assert!(!json.contains("gpu_arch"));
        assert!(!json.contains("trt_version"));
        assert!(!json.contains("platt_a"));
        assert!(!json.contains("platt_b"));
    }

    // --- Clone / Eq ---

    #[test]
    fn test_precision_clone_eq() {
        let a = Precision::INT8;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_device_clone_eq() {
        let a = Device::Cuda(2);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_scorer_type_clone_eq() {
        let a = ScorerType::ColBERT;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // --- Enum variant coverage ---

    #[test]
    fn test_precision_all_variants_debug() {
        let variants = [
            Precision::FP32,
            Precision::FP16,
            Precision::INT8,
            Precision::INT4,
        ];
        for v in variants {
            let dbg = format!("{:?}", v);
            assert!(!dbg.is_empty());
        }
    }

    #[test]
    fn test_device_all_variants_debug() {
        let variants = [Device::Cpu, Device::Cuda(0), Device::TensorRT(0)];
        for v in variants {
            let dbg = format!("{:?}", v);
            assert!(!dbg.is_empty());
        }
    }
}
