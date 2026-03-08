// mcp_server.rs — MCP server binary
//
// When Claude Code connects to this server, it discovers our audio analysis
// tools and can call them on demand. Instead of generating 22 PNGs upfront,
// Claude requests exactly the analysis it needs.
//
// NEW CONCEPT: Async Rust (briefly)
// `async fn` / `.await` lets Rust handle I/O without blocking. The MCP
// framework needs it for communication. Our CPU-bound analysis stays
// synchronous — async just manages the protocol layer.

use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_router, tool_handler,
};
// `rmcp::schemars` re-exports schemars so we use the same version rmcp expects.
// `tool` is the attribute macro for marking methods as MCP tools.
// `tool_router` is the attribute macro for the impl block containing tools.

use serde::Deserialize;

// Import our analysis library
use audio_visualizer_rs::load_audio;
use audio_visualizer_rs::analysis::{spectral, harmonic, rhythm, temporal, downsample};

// ---- Tool parameter structs ----
// `JsonSchema` generates the schema Claude sees when discovering our tools.

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct AudioInfoParams {
    /// Absolute path to the audio file (mp3, wav, flac, ogg, aac)
    path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SpectralParams {
    /// Absolute path to the audio file
    path: String,
    /// FFT window size (default: 2048). Larger = better frequency resolution.
    n_fft: Option<usize>,
    /// Hop length between windows (default: n_fft/4). Smaller = better time resolution.
    hop_length: Option<usize>,
    /// Time-series resolution: "low" (~0.5/sec), "medium" (~1/sec), "high" (~4/sec), or a number. Omit for summary only.
    resolution: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct HarmonicParams {
    /// Absolute path to the audio file
    path: String,
    /// Time-series resolution: "low" (~0.5/sec), "medium" (~1/sec), "high" (~4/sec), or a number. Omit for summary only.
    resolution: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RhythmParams {
    /// Absolute path to the audio file
    path: String,
    /// Minimum BPM to consider (default: 50)
    min_bpm: Option<f32>,
    /// Maximum BPM to consider (default: 220)
    max_bpm: Option<f32>,
    /// Time-series resolution: "low" (~0.5/sec), "medium" (~1/sec), "high" (~4/sec), or a number. Omit for summary only.
    resolution: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct FullAnalysisParams {
    /// Absolute path to the audio file
    path: String,
    /// Time-series resolution: "low" (~0.5/sec), "medium" (~1/sec), "high" (~4/sec), or a number. Omit for summary only.
    resolution: Option<String>,
}

// ---- The MCP Server ----

#[derive(Debug, Clone)]
struct AudioAnalyzerServer {
    tool_router: ToolRouter<Self>,
}

impl AudioAnalyzerServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

// Helper: load audio and compute spectrogram in one step.
fn load_and_analyse(
    path: &str,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Result<(audio_visualizer_rs::AudioData, spectral::Spectrogram), String> {
    let audio = load_audio(path)?;
    let spectrogram = spectral::compute_spectrogram(
        &audio.samples,
        audio.sample_rate,
        n_fft,
        hop_length,
    );
    Ok((audio, spectrogram))
}

// `#[tool_router]` collects all #[tool] methods and builds the routing table.
// Notice the methods are NOT async — they're plain synchronous functions.
// The rmcp framework handles the async wrapping.
#[tool_router]
impl AudioAnalyzerServer {

    #[tool(description = "Get basic information about an audio file (duration, sample rate, sample count). Use this first to understand what you're working with.")]
    fn audio_info(&self, Parameters(params): Parameters<AudioInfoParams>) -> String {
        match load_audio(&params.path) {
            Ok(audio) => {
                format!(
                    "File: {}\nSample rate: {} Hz\nSamples: {}\nDuration: {:.2} seconds",
                    params.path, audio.sample_rate, audio.len(), audio.duration,
                )
            }
            Err(e) => format!("Error: {}", e),
        }
    }

    #[tool(description = "Analyse spectral and temporal features: brightness (centroid), richness (bandwidth), energy distribution (rolloff), tonality (flatness), loudness (RMS), texture (zero crossing rate), and timbre (MFCCs). Set resolution for time-series data.")]
    fn spectral_features(&self, Parameters(params): Parameters<SpectralParams>) -> String {
        match load_and_analyse(&params.path, params.n_fft, params.hop_length) {
            Ok((audio, spectrogram)) => {
                let centroid = spectral::spectral_centroid(&spectrogram);
                let bandwidth = spectral::spectral_bandwidth(&spectrogram);
                let rolloff = spectral::spectral_rolloff(&spectrogram, None);
                let flatness = spectral::spectral_flatness(&spectrogram);
                let rms = temporal::rms_energy(&audio.samples, spectrogram.n_fft, spectrogram.hop_length);
                let zcr = temporal::zero_crossing_rate(&audio.samples, spectrogram.n_fft, spectrogram.hop_length);
                let mfccs = spectral::compute_mfccs(&spectrogram, None, None);
                let avg = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;

                // Average MFCCs across all frames for summary
                let n_mfcc = mfccs.first().map_or(0, |f| f.len());
                let mut avg_mfcc = vec![0.0_f32; n_mfcc];
                for frame in &mfccs {
                    for (i, &val) in frame.iter().enumerate() {
                        avg_mfcc[i] += val;
                    }
                }
                let mfcc_n = mfccs.len() as f32;
                for val in &mut avg_mfcc {
                    *val /= mfcc_n;
                }
                let mfcc_summary: Vec<String> = avg_mfcc.iter().enumerate()
                    .map(|(i, &v)| format!("MFCC-{}: {:.2}", i, v))
                    .collect();

                let mut result = format!(
                    "Spectral Analysis: {}\n\
                     Duration: {:.2} sec | Frames: {} | Freq bins: {}\n\n\
                     Centroid (brightness):     avg {:.0} Hz — {}\n\
                     Bandwidth (richness):      avg {:.0} Hz — {}\n\
                     Rolloff (energy focus):    avg {:.0} Hz\n\
                     Flatness (tonality):       avg {:.4} — {}\n\
                     RMS Energy (loudness):     avg {:.4}\n\
                     Zero Crossing Rate:        avg {:.4} — {}\n\
                     MFCCs (timbre):            {}\n",
                    params.path,
                    spectrogram.duration(), spectrogram.n_frames, spectrogram.n_freq_bins,
                    avg(&centroid),
                    if avg(&centroid) > 3000.0 { "bright/sharp" } else if avg(&centroid) > 1500.0 { "moderate" } else { "dark/warm" },
                    avg(&bandwidth),
                    if avg(&bandwidth) > 3000.0 { "wide/complex" } else if avg(&bandwidth) > 1500.0 { "moderate" } else { "narrow/pure" },
                    avg(&rolloff),
                    avg(&flatness),
                    if avg(&flatness) > 0.5 { "noisy" } else if avg(&flatness) > 0.1 { "mixed tonal/noisy" } else { "strongly tonal" },
                    avg(&rms),
                    avg(&zcr),
                    if avg(&zcr) > 0.1 { "percussive/noisy" } else if avg(&zcr) > 0.03 { "mixed" } else { "tonal" },
                    mfcc_summary.join(", "),
                );

                if let Some(ref res) = params.resolution {
                    match downsample::resolution_to_fps(res) {
                        Ok(target_fps) => {
                            let fps = downsample::native_fps(spectrogram.sample_rate, spectrogram.hop_length);
                            let ds_centroid = downsample::downsample_f32(&centroid, fps, target_fps);
                            let ds_bandwidth = downsample::downsample_f32(&bandwidth, fps, target_fps);
                            let ds_rolloff = downsample::downsample_f32(&rolloff, fps, target_fps);
                            let ds_flatness = downsample::downsample_f32(&flatness, fps, target_fps);
                            let ds_rms = downsample::downsample_f32(&rms, fps, target_fps);
                            let ds_zcr = downsample::downsample_f32(&zcr, fps, target_fps);
                            result.push_str(&downsample::format_f32_series(
                                "Spectral/Temporal Features Over Time",
                                &["centroid_hz", "bandwidth_hz", "rolloff_hz", "flatness", "rms", "zcr"],
                                &[&ds_centroid, &ds_bandwidth, &ds_rolloff, &ds_flatness, &ds_rms, &ds_zcr],
                            ));
                        }
                        Err(e) => result.push_str(&format!("\n\nResolution error: {}", e)),
                    }
                }

                result
            }
            Err(e) => format!("Error: {}", e),
        }
    }

    #[tool(description = "Analyse harmonic content: key detection, pitch class distribution (which notes are prominent), and tonal relationships. Essential for understanding melody, chords, and harmony. Set resolution for time-series data.")]
    fn harmonic_analysis(&self, Parameters(params): Parameters<HarmonicParams>) -> String {
        match load_and_analyse(&params.path, None, None) {
            Ok((_audio, spectrogram)) => {
                let chromagram = harmonic::compute_chromagram(&spectrogram);
                let tonnetz = harmonic::compute_tonnetz(&chromagram);
                let (key, mode, confidence) = chromagram.estimate_key();

                let mut avg_chroma = [0.0_f32; 12];
                for frame in &chromagram.chroma {
                    for (i, &val) in frame.iter().enumerate() {
                        avg_chroma[i] += val;
                    }
                }
                let n = chromagram.n_frames as f32;
                for val in &mut avg_chroma {
                    *val /= n;
                }

                let pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
                let mut ranked: Vec<(usize, f32)> = avg_chroma.iter().copied().enumerate().collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let mut result = format!(
                    "Harmonic Analysis: {}\n\nEstimated Key: {} {} (confidence: {:.3})\n\nPitch Class Distribution:\n",
                    params.path, key, mode, confidence,
                );

                for (idx, (pc, energy)) in ranked.iter().enumerate() {
                    let bar_len = (energy * 30.0) as usize;
                    let bar: String = "█".repeat(bar_len);
                    result.push_str(&format!("  {:>2}. {:<2} {:.3} {}\n", idx + 1, pitch_names[*pc], energy, bar));
                }

                let mut avg_tonnetz = [0.0_f32; 6];
                for frame in &tonnetz {
                    for (i, &val) in frame.iter().enumerate() {
                        avg_tonnetz[i] += val;
                    }
                }
                let tn = tonnetz.len() as f32;
                for val in &mut avg_tonnetz {
                    *val /= tn;
                }

                result.push_str(&format!(
                    "\nTonnetz (avg across {} frames):\n  Fifths: ({:.3}, {:.3}) | Minor 3rds: ({:.3}, {:.3}) | Major 3rds: ({:.3}, {:.3})",
                    tonnetz.len(), avg_tonnetz[0], avg_tonnetz[1], avg_tonnetz[2], avg_tonnetz[3], avg_tonnetz[4], avg_tonnetz[5],
                ));

                if let Some(ref res) = params.resolution {
                    match downsample::resolution_to_fps(res) {
                        Ok(target_fps) => {
                            let fps = downsample::native_fps(spectrogram.sample_rate, spectrogram.hop_length);
                            let ds_chroma = downsample::downsample_array(&chromagram.chroma, fps, target_fps);
                            result.push_str(&downsample::format_array_series(
                                "Chromagram Over Time",
                                &["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
                                &ds_chroma,
                            ));
                            let ds_tonnetz = downsample::downsample_array(&tonnetz, fps, target_fps);
                            result.push_str(&downsample::format_array_series(
                                "Tonnetz Over Time",
                                &["fifth_sin", "fifth_cos", "min3_sin", "min3_cos", "maj3_sin", "maj3_cos"],
                                &ds_tonnetz,
                            ));
                        }
                        Err(e) => result.push_str(&format!("\n\nResolution error: {}", e)),
                    }
                }

                result
            }
            Err(e) => format!("Error: {}", e),
        }
    }

    #[tool(description = "Analyse rhythm: tempo estimation (BPM), beat positions, tempo stability, and beat statistics. Shows whether music has a steady beat or is free-tempo. Set resolution for onset strength time-series.")]
    fn rhythm_analysis(&self, Parameters(params): Parameters<RhythmParams>) -> String {
        match load_and_analyse(&params.path, None, None) {
            Ok((_audio, spectrogram)) => {
                let analysis = rhythm::analyse_rhythm(&spectrogram, params.min_bpm, params.max_bpm);

                let mut result = format!(
                    "Rhythm Analysis: {}\n\nEstimated Tempo: {:.1} BPM (confidence: {:.3})\nDetected Beats: {}\n",
                    params.path, analysis.tempo_bpm, analysis.tempo_confidence, analysis.beat_times.len(),
                );

                if let Some(stats) = rhythm::beat_statistics(&analysis.beat_times) {
                    result.push_str(&format!(
                        "\nBeat Statistics:\n  Mean tempo: {:.1} BPM | Median: {:.1} BPM\n  Stability: {:.3} (0=erratic, 1=metronomic)\n  IBI std dev: {:.3} sec\n",
                        stats.mean_bpm, stats.median_bpm, stats.tempo_stability, stats.std_dev_ibi,
                    ));
                }

                if !analysis.beat_times.is_empty() {
                    let show = analysis.beat_times.len().min(20);
                    let beats: Vec<String> = analysis.beat_times[..show].iter().map(|t| format!("{:.2}s", t)).collect();
                    result.push_str(&format!("\nFirst {} beats: {}", show, beats.join(", ")));
                }

                if let Some(ref res) = params.resolution {
                    match downsample::resolution_to_fps(res) {
                        Ok(target_fps) => {
                            let fps = downsample::native_fps(spectrogram.sample_rate, spectrogram.hop_length);
                            let ds_onset = downsample::downsample_f32(&analysis.onset_envelope, fps, target_fps);
                            result.push_str(&downsample::format_f32_series(
                                "Onset Strength Over Time",
                                &["onset_strength"],
                                &[&ds_onset],
                            ));
                        }
                        Err(e) => result.push_str(&format!("\n\nResolution error: {}", e)),
                    }
                }

                result
            }
            Err(e) => format!("Error: {}", e),
        }
    }

    #[tool(description = "Run complete analysis: basic info, spectral/temporal features (brightness, richness, loudness, texture, timbre), harmonic content (key, notes), and rhythm (tempo, beats). Full picture in one call. Set resolution for all time-series data.")]
    fn full_analysis(&self, Parameters(params): Parameters<FullAnalysisParams>) -> String {
        let start = std::time::Instant::now();

        match load_and_analyse(&params.path, None, None) {
            Ok((audio, spectrogram)) => {
                let centroid = spectral::spectral_centroid(&spectrogram);
                let bandwidth = spectral::spectral_bandwidth(&spectrogram);
                let rolloff = spectral::spectral_rolloff(&spectrogram, None);
                let flatness = spectral::spectral_flatness(&spectrogram);
                let rms = temporal::rms_energy(&audio.samples, spectrogram.n_fft, spectrogram.hop_length);
                let zcr = temporal::zero_crossing_rate(&audio.samples, spectrogram.n_fft, spectrogram.hop_length);
                let mfccs = spectral::compute_mfccs(&spectrogram, None, None);
                let avg = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;

                // Average MFCCs for summary
                let n_mfcc = mfccs.first().map_or(0, |f| f.len());
                let mut avg_mfcc = vec![0.0_f32; n_mfcc];
                for frame in &mfccs {
                    for (i, &val) in frame.iter().enumerate() {
                        avg_mfcc[i] += val;
                    }
                }
                let mfcc_n = mfccs.len() as f32;
                for val in &mut avg_mfcc {
                    *val /= mfcc_n;
                }

                let chromagram = harmonic::compute_chromagram(&spectrogram);
                let tonnetz = harmonic::compute_tonnetz(&chromagram);
                let (key, mode, key_confidence) = chromagram.estimate_key();

                let pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
                let mut avg_chroma = [0.0_f32; 12];
                for frame in &chromagram.chroma {
                    for (i, &val) in frame.iter().enumerate() {
                        avg_chroma[i] += val;
                    }
                }
                let n = chromagram.n_frames as f32;
                for val in &mut avg_chroma {
                    *val /= n;
                }
                let mut ranked: Vec<(usize, f32)> = avg_chroma.iter().copied().enumerate().collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let rhythm_result = rhythm::analyse_rhythm(&spectrogram, None, None);
                let beat_stats = rhythm::beat_statistics(&rhythm_result.beat_times);

                let elapsed = start.elapsed();

                let mut result = format!(
                    "═══ Full Audio Analysis ═══\n\
                     File: {}\n\
                     Duration: {:.2} sec | Sample rate: {} Hz | Samples: {}\n\
                     Analysis completed in: {:.2?}\n\n\
                     ── Spectral/Temporal Features ──\n\
                     Centroid (brightness):  {:.0} Hz — {}\n\
                     Bandwidth (richness):   {:.0} Hz — {}\n\
                     Rolloff (energy focus): {:.0} Hz\n\
                     Flatness (tonality):    {:.4} — {}\n\
                     RMS Energy (loudness):  {:.4}\n\
                     Zero Crossing Rate:     {:.4} — {}\n\
                     MFCCs (timbre):         [{}]\n\n\
                     ── Harmonic Content ──\n\
                     Estimated key: {} {} (confidence: {:.3})\n\
                     Top pitch classes:\n",
                    params.path, audio.duration, audio.sample_rate, audio.len(), elapsed,
                    avg(&centroid),
                    if avg(&centroid) > 3000.0 { "bright" } else if avg(&centroid) > 1500.0 { "moderate" } else { "warm/dark" },
                    avg(&bandwidth),
                    if avg(&bandwidth) > 3000.0 { "complex" } else if avg(&bandwidth) > 1500.0 { "moderate" } else { "pure/simple" },
                    avg(&rolloff),
                    avg(&flatness),
                    if avg(&flatness) > 0.5 { "noisy" } else if avg(&flatness) > 0.1 { "mixed" } else { "strongly tonal" },
                    avg(&rms),
                    avg(&zcr),
                    if avg(&zcr) > 0.1 { "percussive/noisy" } else if avg(&zcr) > 0.03 { "mixed" } else { "tonal" },
                    avg_mfcc.iter().map(|v| format!("{:.1}", v)).collect::<Vec<_>>().join(", "),
                    key, mode, key_confidence,
                );

                for (idx, (pc, energy)) in ranked.iter().take(6).enumerate() {
                    let bar_len = (energy * 25.0) as usize;
                    let bar: String = "█".repeat(bar_len);
                    result.push_str(&format!("  {:>2}. {:<2} {:.3} {}\n", idx + 1, pitch_names[*pc], energy, bar));
                }

                result.push_str(&format!(
                    "\n── Rhythm ──\nTempo: {:.1} BPM (confidence: {:.3})\nBeats detected: {}\n",
                    rhythm_result.tempo_bpm, rhythm_result.tempo_confidence, rhythm_result.beat_times.len(),
                ));

                if let Some(stats) = beat_stats {
                    result.push_str(&format!(
                        "Mean tempo: {:.1} BPM | Median: {:.1} BPM\nStability: {:.3} (0=free, 1=locked)\n",
                        stats.mean_bpm, stats.median_bpm, stats.tempo_stability,
                    ));
                }

                // Append unified time-series table if resolution was requested
                if let Some(ref res) = params.resolution {
                    match downsample::resolution_to_fps(res) {
                        Ok(target_fps) => {
                            let fps = downsample::native_fps(spectrogram.sample_rate, spectrogram.hop_length);

                            let ds_centroid = downsample::downsample_f32(&centroid, fps, target_fps);
                            let ds_bandwidth = downsample::downsample_f32(&bandwidth, fps, target_fps);
                            let ds_rolloff = downsample::downsample_f32(&rolloff, fps, target_fps);
                            let ds_flatness = downsample::downsample_f32(&flatness, fps, target_fps);
                            let ds_rms = downsample::downsample_f32(&rms, fps, target_fps);
                            let ds_zcr = downsample::downsample_f32(&zcr, fps, target_fps);
                            let ds_onset = downsample::downsample_f32(&rhythm_result.onset_envelope, fps, target_fps);
                            let ds_chroma = downsample::downsample_array(&chromagram.chroma, fps, target_fps);
                            let ds_tonnetz = downsample::downsample_array(&tonnetz, fps, target_fps);

                            let chroma_cols: &[&str] = &["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
                            let tonnetz_cols: &[&str] = &["fifth_sin", "fifth_cos", "min3_sin", "min3_cos", "maj3_sin", "maj3_cos"];

                            result.push_str(&downsample::format_unified_timeseries(
                                ds_centroid.len(),
                                &[
                                    (&["centroid_hz"], &ds_centroid[..]),
                                    (&["bandwidth_hz"], &ds_bandwidth[..]),
                                    (&["rolloff_hz"], &ds_rolloff[..]),
                                    (&["flatness"], &ds_flatness[..]),
                                    (&["rms"], &ds_rms[..]),
                                    (&["zcr"], &ds_zcr[..]),
                                    (&["onset"], &ds_onset[..]),
                                ],
                                Some((&ds_chroma, chroma_cols)),
                                Some((&ds_tonnetz, tonnetz_cols)),
                            ));
                        }
                        Err(e) => result.push_str(&format!("\n\nResolution error: {}", e)),
                    }
                }

                result
            }
            Err(e) => format!("Error: {}", e),
        }
    }
}

// `#[tool_handler]` generates the `list_tools` and `call_tool` implementations
// on ServerHandler, wiring them to the tool_router field. Without this, the
// server connects but reports zero tools — the handler doesn't know where to
// find them. This was the missing piece!
#[tool_handler]
impl ServerHandler for AudioAnalyzerServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "Audio analysis server for examining music files. \
                 Provides tools for spectral features, harmonic analysis \
                 (key detection, pitch classes), and rhythm analysis \
                 (tempo, beats). Pass absolute file paths to the tools."
            )
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Log to stderr so it doesn't interfere with MCP's stdio JSON-RPC.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .with_writer(std::io::stderr)
        .init();

    tracing::info!("Starting Audio Analyzer MCP server...");

    let server = AudioAnalyzerServer::new();
    let service = server
        .serve(rmcp::transport::stdio())
        .await?;

    tracing::info!("Server running. Waiting for requests...");
    service.waiting().await?;
    tracing::info!("Server shutting down.");

    Ok(())
}
