use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use rand::SeedableRng;
use rand::seq::SliceRandom;

use crate::accuracy;
use flash_rerank::engine::Scorer;

/// MSMARCO evaluator for reranking quality benchmarks.
///
/// Loads the MSMARCO dataset (corpus, queries, qrels) and provides
/// evaluation methods for NDCG@k and MRR metrics.
///
/// The dataset layout expected:
/// ```text
/// {msmarco_dir}/
///   corpus.jsonl       -> {"_id": "0", "title": "", "text": "...", "metadata": {}}
///   queries.jsonl      -> {"_id": "1185869", "text": "...", "metadata": {}}
///   qrels/dev.tsv      -> query_id\tcorpus_id\tscore (tab-separated, header line)
/// ```
pub struct MsmarcoEvaluator {
    /// (id, text) pairs for queries.
    pub queries: Vec<(String, String)>,
    /// id -> text mapping for corpus documents.
    pub corpus: HashMap<String, String>,
    /// query_id -> [(doc_id, relevance)] mapping from qrels.
    pub qrels: HashMap<String, Vec<(String, u8)>>,
}

impl MsmarcoEvaluator {
    /// Load the MSMARCO dataset from `msmarco_dir`.
    ///
    /// If `max_queries` is Some(n), only loads the first n queries that
    /// have relevance judgments (to keep evaluation tractable).
    pub fn load(
        msmarco_dir: &Path,
        max_queries: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // 1. Parse qrels first so we know which queries have judgments
        let qrels = Self::parse_qrels(msmarco_dir)?;

        // 2. Parse queries, filtering to those with qrels
        let queries = Self::parse_queries(msmarco_dir, &qrels, max_queries)?;

        // 3. Collect all doc_ids referenced in qrels for queries we loaded
        let query_ids: std::collections::HashSet<&str> =
            queries.iter().map(|(id, _)| id.as_str()).collect();
        let needed_doc_ids: std::collections::HashSet<String> = qrels
            .iter()
            .filter(|(qid, _)| query_ids.contains(qid.as_str()))
            .flat_map(|(_, docs)| docs.iter().map(|(did, _)| did.clone()))
            .collect();

        // 4. Parse corpus (only docs referenced in qrels + some extras for negatives)
        let corpus = Self::parse_corpus(msmarco_dir, Some(&needed_doc_ids))?;

        Ok(Self {
            queries,
            corpus,
            qrels,
        })
    }

    /// Evaluate NDCG@k over the loaded queries.
    ///
    /// For each query:
    /// 1. Collect relevant doc_ids from qrels
    /// 2. Sample non-relevant docs to fill up to `top_candidates` total
    /// 3. Score all candidates with the scorer
    /// 4. Compute NDCG@k from the ranked results
    ///
    /// Returns the mean NDCG@k across all evaluated queries.
    pub fn evaluate_ndcg(&self, scorer: &dyn Scorer, k: usize, top_candidates: usize) -> f64 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let all_doc_ids: Vec<&String> = self.corpus.keys().collect();
        let mut total_ndcg = 0.0;
        let mut evaluated = 0usize;

        for (qid, query_text) in &self.queries {
            let Some(rels) = self.qrels.get(qid) else {
                continue;
            };

            // Collect relevant doc texts and their relevance labels
            let relevant_ids: Vec<&str> = rels.iter().map(|(did, _)| did.as_str()).collect();
            let relevant_set: std::collections::HashSet<&str> =
                relevant_ids.iter().copied().collect();

            // Build candidate pool: relevant docs + random negatives
            let mut candidates: Vec<(String, u8)> = Vec::new(); // (doc_text, relevance)
            for (did, rel) in rels {
                if let Some(text) = self.corpus.get(did) {
                    candidates.push((text.clone(), *rel));
                }
            }

            // Fill with random non-relevant docs
            let needed_negatives = top_candidates.saturating_sub(candidates.len());
            if needed_negatives > 0 {
                let mut neg_pool: Vec<&&String> = all_doc_ids
                    .iter()
                    .filter(|did| !relevant_set.contains(did.as_str()))
                    .collect();
                neg_pool.shuffle(&mut rng);
                for did in neg_pool.into_iter().take(needed_negatives) {
                    if let Some(text) = self.corpus.get(did.as_str()) {
                        candidates.push((text.clone(), 0));
                    }
                }
            }

            if candidates.is_empty() {
                continue;
            }

            // Score with the model
            let doc_texts: Vec<String> = candidates.iter().map(|(t, _)| t.clone()).collect();
            let relevance_labels: Vec<u8> = candidates.iter().map(|(_, r)| *r).collect();

            match scorer.score(query_text, &doc_texts) {
                Ok(results) => {
                    // results are sorted by score descending
                    let ranked: Vec<usize> = results.iter().map(|r| r.index).collect();
                    let ndcg = accuracy::ndcg_at_k(&ranked, &relevance_labels, k);
                    total_ndcg += ndcg;
                    evaluated += 1;
                }
                Err(e) => {
                    eprintln!("Scorer error for query {qid}: {e}");
                }
            }
        }

        if evaluated == 0 {
            return 0.0;
        }
        total_ndcg / evaluated as f64
    }

    /// Evaluate MRR (Mean Reciprocal Rank) over the loaded queries.
    ///
    /// For each query:
    /// 1. Collect relevant doc_ids from qrels
    /// 2. Sample non-relevant docs to fill up to `top_candidates` total
    /// 3. Score all candidates with the scorer
    /// 4. Compute RR (reciprocal rank of first relevant doc)
    ///
    /// Returns the mean MRR across all evaluated queries.
    pub fn evaluate_mrr(&self, scorer: &dyn Scorer, top_candidates: usize) -> f64 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let all_doc_ids: Vec<&String> = self.corpus.keys().collect();
        let mut total_rr = 0.0;
        let mut evaluated = 0usize;

        for (qid, query_text) in &self.queries {
            let Some(rels) = self.qrels.get(qid) else {
                continue;
            };

            let relevant_ids: Vec<&str> = rels.iter().map(|(did, _)| did.as_str()).collect();
            let relevant_set: std::collections::HashSet<&str> =
                relevant_ids.iter().copied().collect();

            // Build candidate pool
            let mut candidates: Vec<(String, bool)> = Vec::new(); // (doc_text, is_relevant)
            for (did, rel) in rels {
                if *rel > 0 {
                    if let Some(text) = self.corpus.get(did) {
                        candidates.push((text.clone(), true));
                    }
                }
            }

            let relevant_count = candidates.len();

            // Fill with negatives
            let needed_negatives = top_candidates.saturating_sub(candidates.len());
            if needed_negatives > 0 {
                let mut neg_pool: Vec<&&String> = all_doc_ids
                    .iter()
                    .filter(|did| !relevant_set.contains(did.as_str()))
                    .collect();
                neg_pool.shuffle(&mut rng);
                for did in neg_pool.into_iter().take(needed_negatives) {
                    if let Some(text) = self.corpus.get(did.as_str()) {
                        candidates.push((text.clone(), false));
                    }
                }
            }

            if candidates.is_empty() {
                continue;
            }

            let doc_texts: Vec<String> = candidates.iter().map(|(t, _)| t.clone()).collect();
            // Relevant indices are the first `relevant_count` entries
            let relevant_indices: Vec<usize> = (0..relevant_count).collect();

            match scorer.score(query_text, &doc_texts) {
                Ok(results) => {
                    let ranked: Vec<usize> = results.iter().map(|r| r.index).collect();
                    let rr = accuracy::mrr(&ranked, &relevant_indices);
                    total_rr += rr;
                    evaluated += 1;
                }
                Err(e) => {
                    eprintln!("Scorer error for query {qid}: {e}");
                }
            }
        }

        if evaluated == 0 {
            return 0.0;
        }
        total_rr / evaluated as f64
    }

    /// Parse qrels/dev.tsv into query_id -> [(doc_id, relevance)].
    fn parse_qrels(
        msmarco_dir: &Path,
    ) -> Result<HashMap<String, Vec<(String, u8)>>, Box<dyn std::error::Error>> {
        let qrels_path = msmarco_dir.join("qrels").join("dev.tsv");
        let file = std::fs::File::open(&qrels_path)?;
        let reader = std::io::BufReader::new(file);
        let mut qrels: HashMap<String, Vec<(String, u8)>> = HashMap::new();

        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            // Skip header row
            if i == 0 && (line.starts_with("query") || line.starts_with("query-id")) {
                continue;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let parts: Vec<&str> = trimmed.split('\t').collect();
            if parts.len() < 3 {
                continue;
            }

            let query_id = parts[0].to_string();
            let doc_id = parts[1].to_string();
            let score: u8 = parts[2].parse().unwrap_or(0);

            qrels
                .entry(query_id)
                .or_insert_with(Vec::new)
                .push((doc_id, score));
        }

        Ok(qrels)
    }

    /// Parse queries.jsonl, filtering to those present in qrels.
    fn parse_queries(
        msmarco_dir: &Path,
        qrels: &HashMap<String, Vec<(String, u8)>>,
        max_queries: Option<usize>,
    ) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
        let queries_path = msmarco_dir.join("queries.jsonl");
        let file = std::fs::File::open(&queries_path)?;
        let reader = std::io::BufReader::new(file);
        let mut queries = Vec::new();
        let limit = max_queries.unwrap_or(usize::MAX);

        for line in reader.lines() {
            if queries.len() >= limit {
                break;
            }
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let val: serde_json::Value = serde_json::from_str(&line)?;
            let id = val["_id"]
                .as_str()
                .ok_or("missing _id in queries.jsonl")?
                .to_string();

            // Only include queries that have relevance judgments
            if !qrels.contains_key(&id) {
                continue;
            }

            let text = val["text"]
                .as_str()
                .ok_or("missing text in queries.jsonl")?
                .to_string();
            queries.push((id, text));
        }

        Ok(queries)
    }

    /// Parse corpus.jsonl. If `needed_ids` is Some, also loads extra docs for negative sampling.
    fn parse_corpus(
        msmarco_dir: &Path,
        needed_ids: Option<&std::collections::HashSet<String>>,
    ) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        let corpus_path = msmarco_dir.join("corpus.jsonl");
        let file = std::fs::File::open(&corpus_path)?;
        let reader = std::io::BufReader::new(file);
        let mut corpus = HashMap::new();
        let mut extra_count = 0usize;
        // Load some extra docs beyond the needed ones for negative sampling
        let max_extra = 10_000;

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

            let is_needed = needed_ids.map(|ids| ids.contains(&id)).unwrap_or(true);

            if is_needed {
                corpus.insert(id, doc_text);
            } else if extra_count < max_extra {
                corpus.insert(id, doc_text);
                extra_count += 1;
            }
            // Once we have enough extras and all needed, we can stop early
            // but for simplicity we read the whole file (MSMARCO is ~8M docs,
            // we cap extras at 10k)
            if extra_count >= max_extra && !is_needed {
                // Continue to find remaining needed docs
                if let Some(ids) = needed_ids {
                    if ids.iter().all(|id| corpus.contains_key(id)) {
                        break;
                    }
                }
            }
        }

        Ok(corpus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_msmarco(dir: &Path) {
        std::fs::create_dir_all(dir.join("qrels")).unwrap();

        // queries.jsonl
        let mut f = std::fs::File::create(dir.join("queries.jsonl")).unwrap();
        writeln!(
            f,
            r#"{{"_id":"q1","text":"what is gravity","metadata":{{}}}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"_id":"q2","text":"how does photosynthesis work","metadata":{{}}}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"_id":"q3","text":"no judgments for this","metadata":{{}}}}"#
        )
        .unwrap();

        // corpus.jsonl
        let mut f = std::fs::File::create(dir.join("corpus.jsonl")).unwrap();
        writeln!(
            f,
            r#"{{"_id":"d1","title":"","text":"Gravity is a fundamental force of nature.","metadata":{{}}}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"_id":"d2","title":"","text":"Plants use sunlight to produce energy.","metadata":{{}}}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"_id":"d3","title":"","text":"Cooking is a useful life skill.","metadata":{{}}}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"_id":"d4","title":"","text":"The earth orbits the sun.","metadata":{{}}}}"#
        )
        .unwrap();

        // qrels/dev.tsv
        let mut f = std::fs::File::create(dir.join("qrels").join("dev.tsv")).unwrap();
        writeln!(f, "query-id\tcorpus-id\tscore").unwrap();
        writeln!(f, "q1\td1\t1").unwrap();
        writeln!(f, "q2\td2\t1").unwrap();
    }

    #[test]
    fn load_test_msmarco() {
        let tmp = std::env::temp_dir().join("flash_rerank_msmarco_test");
        let _ = std::fs::remove_dir_all(&tmp);
        create_test_msmarco(&tmp);

        let eval = MsmarcoEvaluator::load(&tmp, None).unwrap();

        // q3 has no qrels so should be filtered out
        assert_eq!(eval.queries.len(), 2);
        assert!(eval.qrels.contains_key("q1"));
        assert!(eval.qrels.contains_key("q2"));
        assert!(!eval.qrels.contains_key("q3"));
        assert!(eval.corpus.contains_key("d1"));
        assert!(eval.corpus.contains_key("d2"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn load_with_max_queries() {
        let tmp = std::env::temp_dir().join("flash_rerank_msmarco_max_test");
        let _ = std::fs::remove_dir_all(&tmp);
        create_test_msmarco(&tmp);

        let eval = MsmarcoEvaluator::load(&tmp, Some(1)).unwrap();
        assert_eq!(eval.queries.len(), 1);

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
