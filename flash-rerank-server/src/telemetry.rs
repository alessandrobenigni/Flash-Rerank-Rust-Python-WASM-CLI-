use anyhow::Result;

/// Initialize OpenTelemetry tracing with OTLP export.
///
/// When compiled with the `telemetry` feature, this sets up a tracing-opentelemetry
/// layer that exports spans via OTLP (gRPC/tonic). When the feature is disabled,
/// this is a no-op that returns `Ok(())`.
///
/// # Arguments
/// * `otlp_endpoint` - Optional OTLP collector endpoint (e.g. "http://localhost:4317").
///   If `None`, telemetry is not initialized even when the feature is enabled.
#[cfg(feature = "telemetry")]
pub fn init_telemetry(otlp_endpoint: Option<&str>) -> Result<()> {
    use opentelemetry::trace::TracerProvider;
    use opentelemetry_otlp::WithExportConfig;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let otlp_endpoint = match otlp_endpoint {
        Some(ep) => ep,
        None => {
            tracing::info!("OTLP: not configured, skipping telemetry init");
            return Ok(());
        }
    };

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(otlp_endpoint)
        .build()?;

    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .build();

    let tracer = provider.tracer("flash-rerank");
    let telemetry_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(telemetry_layer)
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    tracing::info!(endpoint = otlp_endpoint, "OTLP telemetry initialized");
    Ok(())
}

/// No-op telemetry initialization when the `telemetry` feature is disabled.
#[cfg(not(feature = "telemetry"))]
pub fn init_telemetry(_otlp_endpoint: Option<&str>) -> Result<()> {
    Ok(())
}
