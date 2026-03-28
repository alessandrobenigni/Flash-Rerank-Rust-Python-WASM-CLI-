use divan::Bencher;
use std::sync::OnceLock;

static TOKENIZER: OnceLock<std::sync::Mutex<flash_rerank::tokenize::Tokenizer>> = OnceLock::new();

fn get_tokenizer() -> &'static std::sync::Mutex<flash_rerank::tokenize::Tokenizer> {
    TOKENIZER.get_or_init(|| {
        let cache_dir = default_cache_dir();
        let snapshot_dir = cache_dir
            .join("models--cross-encoder--ms-marco-MiniLM-L-6-v2")
            .join("snapshots");

        // Find the first snapshot directory
        let snapshot = std::fs::read_dir(&snapshot_dir)
            .unwrap_or_else(|_| {
                panic!(
                    "Tokenizer snapshot not found at {}. Download the model first.",
                    snapshot_dir.display()
                )
            })
            .next()
            .expect("No snapshot directories found")
            .unwrap()
            .path();

        let tokenizer =
            flash_rerank::tokenize::Tokenizer::from_file(&snapshot.join("tokenizer.json"))
                .expect("Failed to load tokenizer");

        std::sync::Mutex::new(tokenizer)
    })
}

fn default_cache_dir() -> std::path::PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return std::path::PathBuf::from(cache);
    }
    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        return std::path::PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    std::path::PathBuf::from(".cache/huggingface/hub")
}

#[divan::bench]
fn tokenize_single_pair(b: Bencher) {
    let tokenizer = get_tokenizer();
    let query = "What is machine learning?";
    let docs = vec!["A document about ML and artificial intelligence.".to_string()];

    b.bench(|| {
        let mut tok = tokenizer.lock().unwrap();
        tok.tokenize_pairs(query, &docs, 512).unwrap()
    });
}

#[divan::bench]
fn tokenize_batch_64(b: Bencher) {
    let tokenizer = get_tokenizer();
    let query = "What is machine learning?";
    let docs: Vec<String> = (0..64)
        .map(|i| format!("Document number {i} about various topics in science and technology"))
        .collect();

    b.bench(|| {
        let mut tok = tokenizer.lock().unwrap();
        tok.tokenize_pairs(query, &docs, 512).unwrap()
    });
}

fn main() {
    divan::main();
}
