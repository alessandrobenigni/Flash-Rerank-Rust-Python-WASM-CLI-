use std::io::BufRead;
use std::path::Path;

/// BEIR dataset loader for evaluation benchmarks.
///
/// BEIR datasets follow a standard layout:
/// - `queries.jsonl`:      `{"_id": "q1", "text": "query text"}`
/// - `corpus.jsonl`:       `{"_id": "d1", "title": "...", "text": "doc text"}`
/// - `qrels/test.tsv`:     `query-id\tcorpus-id\tscore` (tab-separated, header row)
pub struct BeirDataset {
    /// Dataset name (e.g., "scifact", "msmarco").
    pub name: String,
    /// Query texts, indexed by position.
    pub queries: Vec<String>,
    /// Document texts (title + text concatenated), indexed by position.
    pub documents: Vec<String>,
    /// Relevance judgments as (query_index, document_index, relevance_label).
    pub relevance: Vec<(usize, usize, u8)>,
}

impl BeirDataset {
    /// Load a BEIR dataset from a local directory.
    ///
    /// Expects the directory structure:
    /// ```text
    /// {path}/{dataset_name}/
    ///   queries.jsonl
    ///   corpus.jsonl
    ///   qrels/test.tsv
    /// ```
    pub fn load(path: &Path, dataset_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let base = path.join(dataset_name);

        // Parse queries -- build id->index mapping
        let (queries, query_id_map) = Self::parse_queries_jsonl(&base.join("queries.jsonl"))?;

        // Parse corpus -- build id->index mapping
        let (documents, doc_id_map) = Self::parse_corpus_jsonl(&base.join("corpus.jsonl"))?;

        // Parse qrels using the id mappings
        let relevance = Self::parse_qrels_tsv(
            &base.join("qrels").join("test.tsv"),
            &query_id_map,
            &doc_id_map,
        )?;

        Ok(Self {
            name: dataset_name.to_string(),
            queries,
            documents,
            relevance,
        })
    }

    /// Parse `queries.jsonl` into (texts, id->index map).
    fn parse_queries_jsonl(
        path: &Path,
    ) -> Result<(Vec<String>, std::collections::HashMap<String, usize>), Box<dyn std::error::Error>>
    {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut texts = Vec::new();
        let mut id_map = std::collections::HashMap::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let val: serde_json::Value = serde_json::from_str(&line)?;
            let id = val["_id"]
                .as_str()
                .ok_or("missing _id in queries.jsonl")?
                .to_string();
            let text = val["text"]
                .as_str()
                .ok_or("missing text in queries.jsonl")?
                .to_string();
            let idx = texts.len();
            id_map.insert(id, idx);
            texts.push(text);
        }

        Ok((texts, id_map))
    }

    /// Parse `corpus.jsonl` into (texts, id->index map).
    /// Concatenates title and text with a space separator.
    fn parse_corpus_jsonl(
        path: &Path,
    ) -> Result<(Vec<String>, std::collections::HashMap<String, usize>), Box<dyn std::error::Error>>
    {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut texts = Vec::new();
        let mut id_map = std::collections::HashMap::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let val: serde_json::Value = serde_json::from_str(&line)?;
            let id = val["_id"]
                .as_str()
                .ok_or("missing _id in corpus.jsonl")?
                .to_string();
            let title = val["title"].as_str().unwrap_or("");
            let text = val["text"].as_str().ok_or("missing text in corpus.jsonl")?;

            let doc_text = if title.is_empty() {
                text.to_string()
            } else {
                format!("{title} {text}")
            };

            let idx = texts.len();
            id_map.insert(id, idx);
            texts.push(doc_text);
        }

        Ok((texts, id_map))
    }

    /// Parse `qrels/test.tsv` into relevance triples.
    /// Format: `query-id\tcorpus-id\tscore` with a header row.
    fn parse_qrels_tsv(
        path: &Path,
        query_id_map: &std::collections::HashMap<String, usize>,
        doc_id_map: &std::collections::HashMap<String, usize>,
    ) -> Result<Vec<(usize, usize, u8)>, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut relevance = Vec::new();

        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            // Skip header row
            if i == 0 {
                continue;
            }
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 3 {
                continue;
            }

            let query_id = parts[0];
            let doc_id = parts[1];
            let score: u8 = parts[2].parse().unwrap_or(0);

            if let (Some(&q_idx), Some(&d_idx)) =
                (query_id_map.get(query_id), doc_id_map.get(doc_id))
            {
                relevance.push((q_idx, d_idx, score));
            }
        }

        Ok(relevance)
    }

    /// Get all document indices relevant to a given query index.
    pub fn relevant_docs(&self, query_idx: usize) -> Vec<usize> {
        self.relevance
            .iter()
            .filter(|(q, _, rel)| *q == query_idx && *rel > 0)
            .map(|(_, d, _)| *d)
            .collect()
    }

    /// Get relevance labels for all documents given a query index.
    /// Returns a Vec of length `self.documents.len()` with relevance scores.
    pub fn relevance_vector(&self, query_idx: usize) -> Vec<u8> {
        let mut rels = vec![0u8; self.documents.len()];
        for &(q, d, rel) in &self.relevance {
            if q == query_idx && d < rels.len() {
                rels[d] = rel;
            }
        }
        rels
    }

    /// Number of queries in the dataset.
    pub fn num_queries(&self) -> usize {
        self.queries.len()
    }

    /// Number of documents in the corpus.
    pub fn num_documents(&self) -> usize {
        self.documents.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_dataset(dir: &Path) {
        std::fs::create_dir_all(dir.join("qrels")).unwrap();

        // queries.jsonl
        let mut f = std::fs::File::create(dir.join("queries.jsonl")).unwrap();
        writeln!(f, r#"{{"_id":"q1","text":"What is gravity?"}}"#).unwrap();
        writeln!(
            f,
            r#"{{"_id":"q2","text":"How does photosynthesis work?"}}"#
        )
        .unwrap();

        // corpus.jsonl
        let mut f = std::fs::File::create(dir.join("corpus.jsonl")).unwrap();
        writeln!(
            f,
            r#"{{"_id":"d1","title":"Gravity","text":"Gravity is a fundamental force."}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"_id":"d2","title":"Plants","text":"Plants convert sunlight to energy."}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"_id":"d3","title":"","text":"Unrelated document about cooking."}}"#
        )
        .unwrap();

        // qrels/test.tsv
        let mut f = std::fs::File::create(dir.join("qrels").join("test.tsv")).unwrap();
        writeln!(f, "query-id\tcorpus-id\tscore").unwrap();
        writeln!(f, "q1\td1\t2").unwrap();
        writeln!(f, "q2\td2\t1").unwrap();
    }

    #[test]
    fn load_test_fixture() {
        let tmp = std::env::temp_dir().join("flash_rerank_beir_test");
        let dataset_dir = tmp.join("test_dataset");
        let _ = std::fs::remove_dir_all(&tmp);
        create_test_dataset(&dataset_dir);

        let dataset = BeirDataset::load(&tmp, "test_dataset").unwrap();

        assert_eq!(dataset.name, "test_dataset");
        assert_eq!(dataset.num_queries(), 2);
        assert_eq!(dataset.num_documents(), 3);
        assert_eq!(dataset.queries[0], "What is gravity?");
        assert_eq!(
            dataset.documents[0],
            "Gravity Gravity is a fundamental force."
        );
        assert_eq!(dataset.documents[2], "Unrelated document about cooking.");

        // Check relevance
        assert_eq!(dataset.relevance.len(), 2);
        let rel_q0 = dataset.relevant_docs(0);
        assert_eq!(rel_q0, vec![0]); // q1 -> d1

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
