use std::path::PathBuf;

use clap::Args;
use flash_rerank::Precision;
use flash_rerank::engine::tensorrt::TensorRTCompiler;

#[derive(Args)]
pub struct CompileArgs {
    /// Path to the ONNX model file.
    #[arg(short, long)]
    pub model: String,

    /// Output path for the compiled TensorRT engine.
    #[arg(short, long)]
    pub output: Option<String>,

    /// Precision level (fp32, fp16, int8, int4).
    #[arg(short, long, default_value = "fp16")]
    pub precision: String,

    /// Maximum batch size for the engine.
    #[arg(long, default_value = "64")]
    pub max_batch_size: usize,

    /// GPU architecture (e.g., sm_89, sm_90). When set, checks HuggingFace Hub
    /// for pre-compiled engines before local compilation.
    #[arg(long)]
    pub gpu_arch: Option<String>,

    /// TensorRT version (e.g., 10.0). Required when --gpu-arch is set.
    #[arg(long)]
    pub trt_version: Option<String>,

    /// Skip pre-compiled engine check and always compile locally.
    #[arg(long, default_value = "false")]
    pub no_hub_check: bool,
}

pub async fn run(args: CompileArgs) -> anyhow::Result<()> {
    let precision = match args.precision.as_str() {
        "fp32" => Precision::FP32,
        "fp16" => Precision::FP16,
        "int8" => Precision::INT8,
        "int4" => Precision::INT4,
        other => anyhow::bail!("Unknown precision: {other}. Use: fp32, fp16, int8, int4"),
    };

    let onnx_path = PathBuf::from(&args.model);
    if !onnx_path.exists() {
        anyhow::bail!("ONNX model file not found: {}", onnx_path.display());
    }

    let output_path = args
        .output
        .map(PathBuf::from)
        .unwrap_or_else(|| onnx_path.with_extension("trt"));

    if output_path.exists() {
        println!(
            "Engine already exists at {}. Overwriting.",
            output_path.display()
        );
    }

    let compiler = TensorRTCompiler::new(onnx_path, precision, args.max_batch_size);

    // Check for pre-compiled engine on Hub if GPU arch is specified
    if !args.no_hub_check {
        if let (Some(gpu_arch), Some(trt_version)) = (&args.gpu_arch, &args.trt_version) {
            println!(
                "[1/2] Checking HuggingFace Hub for pre-compiled engine ({}, TRT {})...",
                gpu_arch, trt_version
            );
            compiler
                .compile_or_download(&output_path, gpu_arch, trt_version)
                .await?;
            println!("[2/2] Done. Engine: {}", output_path.display());
            return Ok(());
        }
    }

    println!(
        "[1/2] Compiling ONNX to TensorRT engine ({:?})...",
        precision
    );
    compiler.compile(&output_path)?;
    println!("[2/2] Done. Engine: {}", output_path.display());

    Ok(())
}
