use crate::Result;

/// Wrapper around HuggingFace tokenizers for query-document pair encoding.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Load a tokenizer from a HuggingFace model directory or file.
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| crate::Error::Tokenizer(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get a mutable reference to the inner HuggingFace tokenizer.
    ///
    /// Used by ColBERT scorer for single-text encoding with custom truncation.
    pub fn inner_mut(&mut self) -> &mut tokenizers::Tokenizer {
        &mut self.inner
    }

    /// Get a reference to the inner HuggingFace tokenizer.
    ///
    /// Used by ColBERT scorer for single-text encoding.
    pub fn inner_ref(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }

    /// Tokenize a batch of (query, document) pairs for cross-encoder input.
    ///
    /// Each (query, document) pair is encoded as a dual input for cross-encoder
    /// models, with truncation to `max_length` and padding to the longest
    /// sequence in the batch.
    #[tracing::instrument(skip(self, query, documents), fields(num_documents = documents.len(), max_length))]
    pub fn tokenize_pairs(
        &mut self,
        query: &str,
        documents: &[String],
        max_length: usize,
    ) -> Result<Vec<tokenizers::Encoding>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Configure truncation and padding for cross-encoder input
        let truncation = tokenizers::TruncationParams {
            max_length,
            ..Default::default()
        };
        self.inner
            .with_truncation(Some(truncation))
            .map_err(|e| crate::Error::Tokenizer(e.to_string()))?;

        let padding = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        self.inner.with_padding(Some(padding));

        // Build dual (query, document) encode inputs
        let inputs: Vec<tokenizers::EncodeInput> = documents
            .iter()
            .map(|doc| tokenizers::EncodeInput::Dual(query.into(), doc.as_str().into()))
            .collect();

        self.inner
            .encode_batch(inputs, true)
            .map_err(|e| crate::Error::Tokenizer(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_file_missing_file() {
        let result = Tokenizer::from_file(std::path::Path::new("/nonexistent/tokenizer.json"));
        assert!(result.is_err());
    }

    /// Helper: try to locate a tokenizer.json in HF cache for tests.
    /// Returns None if no cached tokenizer is available.
    fn find_cached_tokenizer() -> Option<std::path::PathBuf> {
        // Check common HF cache locations
        let home = std::env::var("HF_HOME")
            .or_else(|_| std::env::var("HOME").map(|h| format!("{h}/.cache/huggingface/hub")))
            .or_else(|_| {
                std::env::var("USERPROFILE").map(|h| format!("{h}/.cache/huggingface/hub"))
            })
            .ok()?;
        let cache = std::path::Path::new(&home);
        if !cache.exists() {
            return None;
        }
        // Look for any tokenizer.json in model snapshots
        for entry in std::fs::read_dir(cache).ok()?.flatten() {
            let snapshots = entry.path().join("snapshots");
            if snapshots.exists() {
                for snap in std::fs::read_dir(&snapshots).ok()?.flatten() {
                    let tok = snap.path().join("tokenizer.json");
                    if tok.exists() {
                        return Some(tok);
                    }
                }
            }
        }
        None
    }

    #[test]
    #[ignore = "requires a cached tokenizer model"]
    fn test_tokenize_empty_documents() {
        let tok_path = find_cached_tokenizer().expect("No cached tokenizer found");
        let mut tok = Tokenizer::from_file(&tok_path).unwrap();
        let results = tok.tokenize_pairs("query", &[], 128).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    #[ignore = "requires a cached tokenizer model"]
    fn test_tokenize_single_document() {
        let tok_path = find_cached_tokenizer().expect("No cached tokenizer found");
        let mut tok = Tokenizer::from_file(&tok_path).unwrap();
        let docs = vec!["hello world".to_string()];
        let results = tok.tokenize_pairs("test query", &docs, 128).unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].get_ids().is_empty());
    }

    #[test]
    #[ignore = "requires a cached tokenizer model"]
    fn test_tokenize_batch_consistency() {
        let tok_path = find_cached_tokenizer().expect("No cached tokenizer found");
        let mut tok = Tokenizer::from_file(&tok_path).unwrap();
        let docs = vec![
            "doc one".to_string(),
            "doc two".to_string(),
            "doc three longer text".to_string(),
        ];
        let results = tok.tokenize_pairs("query", &docs, 128).unwrap();
        assert_eq!(results.len(), 3);
        // All encodings should have the same padded length (BatchLongest)
        let len0 = results[0].get_ids().len();
        for enc in &results {
            assert_eq!(
                enc.get_ids().len(),
                len0,
                "Batch padding should produce equal lengths"
            );
        }
    }

    #[test]
    #[ignore = "requires a cached tokenizer model"]
    fn test_tokenize_truncation() {
        let tok_path = find_cached_tokenizer().expect("No cached tokenizer found");
        let mut tok = Tokenizer::from_file(&tok_path).unwrap();
        let long_doc = "word ".repeat(1000);
        let docs = vec![long_doc];
        let max_len = 32;
        let results = tok.tokenize_pairs("q", &docs, max_len).unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            results[0].get_ids().len() <= max_len,
            "Truncation should cap length"
        );
    }

    #[test]
    #[ignore = "requires a cached tokenizer model"]
    fn test_tokenize_unicode() {
        let tok_path = find_cached_tokenizer().expect("No cached tokenizer found");
        let mut tok = Tokenizer::from_file(&tok_path).unwrap();
        let docs = vec!["Hola mundo -- 你好世界 🌍".to_string()];
        let results = tok
            .tokenize_pairs("Unicode query: こんにちは", &docs, 128)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].get_ids().is_empty());
    }

    #[test]
    #[ignore = "requires a cached tokenizer model"]
    fn test_tokenize_empty_query_and_doc() {
        let tok_path = find_cached_tokenizer().expect("No cached tokenizer found");
        let mut tok = Tokenizer::from_file(&tok_path).unwrap();
        let docs = vec!["".to_string()];
        let results = tok.tokenize_pairs("", &docs, 128).unwrap();
        assert_eq!(results.len(), 1);
        // Even empty strings produce special tokens
        assert!(!results[0].get_ids().is_empty());
    }

    #[test]
    #[ignore = "requires a cached tokenizer model"]
    fn test_tokenize_very_long_input() {
        let tok_path = find_cached_tokenizer().expect("No cached tokenizer found");
        let mut tok = Tokenizer::from_file(&tok_path).unwrap();
        let long_doc = "a ".repeat(10_000);
        let docs = vec![long_doc];
        let results = tok.tokenize_pairs("q", &docs, 512).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].get_ids().len() <= 512);
    }

    #[test]
    #[ignore = "requires a cached tokenizer model"]
    fn test_inner_mut_returns_mutable_ref() {
        let tok_path = find_cached_tokenizer().expect("No cached tokenizer found");
        let mut tok = Tokenizer::from_file(&tok_path).unwrap();
        let inner = tok.inner_mut();
        // Just verify we can set truncation on the inner tokenizer directly
        let _ = inner.with_truncation(None);
    }

    #[test]
    #[ignore = "requires a cached tokenizer model"]
    fn test_inner_ref_returns_ref() {
        let tok_path = find_cached_tokenizer().expect("No cached tokenizer found");
        let tok = Tokenizer::from_file(&tok_path).unwrap();
        let inner = tok.inner_ref();
        // Verify the inner tokenizer has a vocab
        assert!(inner.get_vocab_size(true) > 0);
    }
}
