fn main() {
    use ort::execution_providers::{CPU, CUDA, TensorRT};
    use ort::session::Session;
    use ort::value::Tensor;

    let home = std::env::var("USERPROFILE").unwrap();
    let base = format!(
        "{home}/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots"
    );
    let snap = std::fs::read_dir(&base)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let model_path = snap.join("onnx/model.onnx");
    println!("Model: {}", model_path.display());

    println!("\n--- GPU Info ---");
    let _ = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ])
        .status();

    let batch = 100;
    let seq = 40;
    let ids: Vec<i64> = vec![1; batch * seq];
    let mask: Vec<i64> = vec![1; batch * seq];
    let tids: Vec<i64> = vec![0; batch * seq];
    let shape = [batch as i64, seq as i64];

    // Test CUDA
    println!("\n--- Testing CUDA EP ---");
    let cuda_result: Result<(), String> = (|| {
        let mut builder = Session::builder().map_err(|e| format!("builder: {e}"))?;
        builder = builder
            .with_execution_providers([CUDA::default().build()])
            .map_err(|e| format!("cuda ep: {e}"))?;
        let mut session = builder
            .commit_from_file(&model_path)
            .map_err(|e| format!("commit: {e}"))?;
        println!("CUDA session created!");

        // Warm up
        for _ in 0..5 {
            let ids_t = Tensor::from_array((shape, ids.clone())).map_err(|e| e.to_string())?;
            let mask_t = Tensor::from_array((shape, mask.clone())).map_err(|e| e.to_string())?;
            let tids_t = Tensor::from_array((shape, tids.clone())).map_err(|e| e.to_string())?;
            session
                .run(ort::inputs![
                    "input_ids" => ids_t, "attention_mask" => mask_t, "token_type_ids" => tids_t,
                ])
                .map_err(|e| e.to_string())?;
        }

        println!("GPU after warm-up:");
        let _ = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader",
            ])
            .status();

        let mut times = Vec::new();
        for _ in 0..30 {
            let t = std::time::Instant::now();
            let ids_t = Tensor::from_array((shape, ids.clone())).map_err(|e| e.to_string())?;
            let mask_t = Tensor::from_array((shape, mask.clone())).map_err(|e| e.to_string())?;
            let tids_t = Tensor::from_array((shape, tids.clone())).map_err(|e| e.to_string())?;
            session
                .run(ort::inputs![
                    "input_ids" => ids_t, "attention_mask" => mask_t, "token_type_ids" => tids_t,
                ])
                .map_err(|e| e.to_string())?;
            times.push(t.elapsed());
        }
        times.sort();
        println!("CUDA 100-doc @ seq={seq} (30 iters):");
        println!("  P50:  {:.2}ms", times[15].as_secs_f64() * 1000.0);
        println!("  Min:  {:.2}ms", times[0].as_secs_f64() * 1000.0);
        println!(
            "  Mean: {:.2}ms",
            times.iter().sum::<std::time::Duration>().as_secs_f64() / 30.0 * 1000.0
        );
        Ok(())
    })();
    if let Err(e) = cuda_result {
        println!("CUDA FAILED: {e}");
    }

    // Test TensorRT FP16
    println!("\n--- Testing TensorRT FP16 EP ---");
    let trt_result: Result<(), String> = (|| {
        let mut builder = Session::builder().map_err(|e| format!("builder: {e}"))?;
        builder = builder
            .with_execution_providers([TensorRT::default().with_fp16(true).build()])
            .map_err(|e| format!("trt ep: {e}"))?;
        let mut session = builder
            .commit_from_file(&model_path)
            .map_err(|e| format!("commit: {e}"))?;
        println!("TensorRT FP16 session created! (first run compiles engine, may take 30-60s)");

        for i in 0..10 {
            let t = std::time::Instant::now();
            let ids_t = Tensor::from_array((shape, ids.clone())).map_err(|e| e.to_string())?;
            let mask_t = Tensor::from_array((shape, mask.clone())).map_err(|e| e.to_string())?;
            let tids_t = Tensor::from_array((shape, tids.clone())).map_err(|e| e.to_string())?;
            session
                .run(ort::inputs![
                    "input_ids" => ids_t, "attention_mask" => mask_t, "token_type_ids" => tids_t,
                ])
                .map_err(|e| e.to_string())?;
            println!(
                "  Warm-up {}: {:.1}ms",
                i + 1,
                t.elapsed().as_secs_f64() * 1000.0
            );
        }

        let mut times = Vec::new();
        for _ in 0..30 {
            let t = std::time::Instant::now();
            let ids_t = Tensor::from_array((shape, ids.clone())).map_err(|e| e.to_string())?;
            let mask_t = Tensor::from_array((shape, mask.clone())).map_err(|e| e.to_string())?;
            let tids_t = Tensor::from_array((shape, tids.clone())).map_err(|e| e.to_string())?;
            session
                .run(ort::inputs![
                    "input_ids" => ids_t, "attention_mask" => mask_t, "token_type_ids" => tids_t,
                ])
                .map_err(|e| e.to_string())?;
            times.push(t.elapsed());
        }
        times.sort();
        println!("TensorRT FP16 100-doc @ seq={seq} (30 iters):");
        println!("  P50:  {:.2}ms", times[15].as_secs_f64() * 1000.0);
        println!("  Min:  {:.2}ms", times[0].as_secs_f64() * 1000.0);
        println!(
            "  Mean: {:.2}ms",
            times.iter().sum::<std::time::Duration>().as_secs_f64() / 30.0 * 1000.0
        );
        Ok(())
    })();
    if let Err(e) = trt_result {
        println!("TensorRT FAILED: {e}");
    }

    // CPU INT8 baseline
    println!("\n--- CPU INT8 Baseline ---");
    let cpu_result: Result<(), String> = (|| {
        let qpath = snap.join("onnx/model_qint8_avx512.onnx");
        let path = if qpath.exists() { &qpath } else { &model_path };
        println!("Using: {}", path.display());
        let mut builder = Session::builder().map_err(|e| format!("builder: {e}"))?;
        builder = builder
            .with_execution_providers([CPU::default().build()])
            .map_err(|e| format!("cpu ep: {e}"))?;
        let mut session = builder
            .commit_from_file(path)
            .map_err(|e| format!("commit: {e}"))?;

        for _ in 0..5 {
            let ids_t = Tensor::from_array((shape, ids.clone())).map_err(|e| e.to_string())?;
            let mask_t = Tensor::from_array((shape, mask.clone())).map_err(|e| e.to_string())?;
            let tids_t = Tensor::from_array((shape, tids.clone())).map_err(|e| e.to_string())?;
            session
                .run(ort::inputs![
                    "input_ids" => ids_t, "attention_mask" => mask_t, "token_type_ids" => tids_t,
                ])
                .map_err(|e| e.to_string())?;
        }

        let mut times = Vec::new();
        for _ in 0..30 {
            let t = std::time::Instant::now();
            let ids_t = Tensor::from_array((shape, ids.clone())).map_err(|e| e.to_string())?;
            let mask_t = Tensor::from_array((shape, mask.clone())).map_err(|e| e.to_string())?;
            let tids_t = Tensor::from_array((shape, tids.clone())).map_err(|e| e.to_string())?;
            session
                .run(ort::inputs![
                    "input_ids" => ids_t, "attention_mask" => mask_t, "token_type_ids" => tids_t,
                ])
                .map_err(|e| e.to_string())?;
            times.push(t.elapsed());
        }
        times.sort();
        println!("CPU INT8 100-doc @ seq={seq} (30 iters):");
        println!("  P50:  {:.2}ms", times[15].as_secs_f64() * 1000.0);
        println!("  Min:  {:.2}ms", times[0].as_secs_f64() * 1000.0);
        println!(
            "  Mean: {:.2}ms",
            times.iter().sum::<std::time::Duration>().as_secs_f64() / 30.0 * 1000.0
        );
        Ok(())
    })();
    if let Err(e) = cpu_result {
        println!("CPU FAILED: {e}");
    }
}
