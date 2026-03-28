//! WASM module for Flash-Rerank -- browser and edge inference via tract.
//!
//! This crate is independent from `flash_rerank` -- it uses `tract-onnx` instead
//! of `ort` for ONNX inference, as `ort` depends on native shared libraries
//! that cannot compile to WebAssembly. Types are duplicated (small surface area).

use serde::{Deserialize, Serialize};
use tract_onnx::prelude::*;
use wasm_bindgen::prelude::*;

/// A single scored document in WASM reranking results.
#[derive(Debug, Serialize, Deserialize)]
pub struct WasmRerankResult {
    pub index: usize,
    pub score: f32,
}

/// Global model state -- loaded once via `load_model()`, used by `rerank()`.
static MODEL: std::sync::OnceLock<
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
> = std::sync::OnceLock::new();

/// Global tokenizer state -- loaded once via `load_model()`.
static TOKENIZER: std::sync::OnceLock<tokenizers::Tokenizer> = std::sync::OnceLock::new();

/// Helper to convert anyhow::Error to JsError (tract returns anyhow errors).
fn to_js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// Load a quantized ONNX model and tokenizer for WASM inference.
///
/// Call this once before calling `rerank()`. The model and tokenizer are stored
/// in global `OnceLock` state. Calling again after successful load returns an error.
///
/// # Arguments
/// * `model_bytes` - Raw bytes of the ONNX model file
/// * `tokenizer_json` - JSON string of the HuggingFace tokenizer config
#[wasm_bindgen]
pub fn load_model(model_bytes: &[u8], tokenizer_json: &str) -> Result<(), JsError> {
    let model = tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(model_bytes))
        .map_err(to_js_err)?
        .into_optimized()
        .map_err(to_js_err)?
        .into_runnable()
        .map_err(to_js_err)?;
    MODEL
        .set(model)
        .map_err(|_| JsError::new("Model already loaded"))?;

    let tokenizer =
        tokenizers::Tokenizer::from_bytes(tokenizer_json.as_bytes()).map_err(to_js_err)?;
    TOKENIZER
        .set(tokenizer)
        .map_err(|_| JsError::new("Tokenizer already loaded"))?;

    Ok(())
}

/// Rerank documents for a query using tract ONNX inference.
///
/// Documents are processed one at a time (no batching) to avoid OOM in WASM.
/// For large candidate lists, the caller should split into sub-batches on the JS side.
///
/// # Arguments
/// * `query` - The search query string
/// * `documents` - A JS array of document strings (passed as JsValue)
///
/// # Returns
/// A JS array of `WasmRerankResult` objects sorted by score descending.
#[wasm_bindgen]
pub fn rerank(query: &str, documents: JsValue) -> Result<JsValue, JsError> {
    let documents: Vec<String> = serde_wasm_bindgen::from_value(documents).map_err(to_js_err)?;
    let model = MODEL
        .get()
        .ok_or_else(|| JsError::new("No model loaded. Call load_model() first."))?;
    let tokenizer = TOKENIZER
        .get()
        .ok_or_else(|| JsError::new("No tokenizer loaded. Call load_model() first."))?;

    let mut results = Vec::with_capacity(documents.len());

    for (i, doc) in documents.iter().enumerate() {
        // Tokenize the query-document pair
        let encoding = tokenizer
            .encode((query, doc.as_str()), true)
            .map_err(to_js_err)?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

        let seq_len = input_ids.len();

        // Build tract input tensors (batch_size=1, seq_len)
        let input_ids =
            tract_ndarray::Array2::from_shape_vec((1, seq_len), input_ids).map_err(to_js_err)?;
        let attention_mask = tract_ndarray::Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(to_js_err)?;
        let token_type_ids = tract_ndarray::Array2::from_shape_vec((1, seq_len), token_type_ids)
            .map_err(to_js_err)?;

        // Run tract inference
        let outputs = model
            .run(tvec![
                input_ids.into_tvalue(),
                attention_mask.into_tvalue(),
                token_type_ids.into_tvalue(),
            ])
            .map_err(to_js_err)?;

        // Extract logit and apply sigmoid calibration
        let logit: f32 = *outputs[0].to_scalar::<f32>().map_err(to_js_err)?;
        let score = 1.0 / (1.0 + (-logit).exp());

        results.push(WasmRerankResult { index: i, score });
    }

    // Sort by score descending
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    serde_wasm_bindgen::to_value(&results).map_err(to_js_err)
}
