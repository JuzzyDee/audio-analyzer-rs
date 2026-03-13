#![allow(clippy::type_complexity)]
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
    schemars, tool, tool_handler, tool_router,
};
// `rmcp::schemars` re-exports schemars so we use the same version rmcp expects.
// `tool` is the attribute macro for marking methods as MCP tools.
// `tool_router` is the attribute macro for the impl block containing tools.

use serde::Deserialize;

// Import our analysis library
use audio_visualizer_rs::analysis::{
    compare, downsample, harmonic, percussive, rhythm, sections, spectral, stereo, temporal,
};
use audio_visualizer_rs::{load_audio, load_audio_stereo};

// ---- Lenient numeric deserialization ----
// Models sometimes send numbers as strings (e.g. "110" instead of 110).
// These helpers accept either form so tool calls don't fail on type mismatch.

fn deserialize_lenient_f32<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<f32>, D::Error> {
    use serde::de;
    struct LenientF32;
    impl<'de> de::Visitor<'de> for LenientF32 {
        type Value = Option<f32>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "a number or numeric string")
        }
        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            Ok(Some(v as f32))
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Some(v as f32))
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
            Ok(Some(v as f32))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            v.parse::<f32>().map(Some).map_err(de::Error::custom)
        }
    }
    d.deserialize_any(LenientF32)
}

fn deserialize_lenient_usize<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<usize>, D::Error> {
    use serde::de;
    struct LenientUsize;
    impl<'de> de::Visitor<'de> for LenientUsize {
        type Value = Option<usize>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "an integer or numeric string")
        }
        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            Ok(Some(v as usize))
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Some(v as usize))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            v.parse::<usize>().map(Some).map_err(de::Error::custom)
        }
    }
    d.deserialize_any(LenientUsize)
}

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
    /// FFT window size (default: 2048). Rarely needs changing.
    #[serde(default, deserialize_with = "deserialize_lenient_usize")]
    n_fft: Option<usize>,
    /// Hop length between windows (default: n_fft/4). Rarely needs changing.
    #[serde(default, deserialize_with = "deserialize_lenient_usize")]
    hop_length: Option<usize>,
    /// Time-series resolution: "low" (~0.5/sec), "medium" (~1/sec), "high" (~4/sec), or a number. Omit for summary only. Token budget: "high" on 60s ≈ 240 rows — prefer for ≤20s windows. "medium" on 60s ≈ 60 rows — good general purpose. "low" for structural overview of long sections. Auto-capped at 800 rows.
    resolution: Option<String>,
    /// Start time in seconds — analyse from this point. Omit to start from the beginning.
    #[serde(default, deserialize_with = "deserialize_lenient_f32")]
    start_time: Option<f32>,
    /// End time in seconds — analyse up to this point. Omit to analyse to the end.
    #[serde(default, deserialize_with = "deserialize_lenient_f32")]
    end_time: Option<f32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct HarmonicParams {
    /// Absolute path to the audio file
    path: String,
    /// Time-series resolution: "low" (~0.5/sec), "medium" (~1/sec), "high" (~4/sec), or a number. Omit for summary only. Token budget: "high" on 60s ≈ 240 rows — prefer for ≤20s windows. "medium" on 60s ≈ 60 rows — good general purpose. "low" for structural overview of long sections. Auto-capped at 800 rows.
    resolution: Option<String>,
    /// Start time in seconds — analyse from this point. Omit to start from the beginning.
    #[serde(default, deserialize_with = "deserialize_lenient_f32")]
    start_time: Option<f32>,
    /// End time in seconds — analyse up to this point. Omit to analyse to the end.
    #[serde(default, deserialize_with = "deserialize_lenient_f32")]
    end_time: Option<f32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RhythmParams {
    /// Absolute path to the audio file
    path: String,
    /// Internal search range lower bound for tempo detection. Default 50. You almost never need to set this — only use if tempo detection gives wrong results and you know the approximate BPM range.
    #[serde(default, deserialize_with = "deserialize_lenient_f32")]
    min_bpm: Option<f32>,
    /// Internal search range upper bound for tempo detection. Default 220. You almost never need to set this — only use if tempo detection gives wrong results and you know the approximate BPM range.
    #[serde(default, deserialize_with = "deserialize_lenient_f32")]
    max_bpm: Option<f32>,
    /// Time-series resolution: "low" (~0.5/sec), "medium" (~1/sec), "high" (~4/sec), or a number. Omit for summary only. Token budget: "high" on 60s ≈ 240 rows — prefer for ≤20s windows. "medium" on 60s ≈ 60 rows — good general purpose. "low" for structural overview of long sections. Auto-capped at 800 rows.
    resolution: Option<String>,
    /// Start time in seconds — analyse from this point. Omit to start from the beginning.
    #[serde(default, deserialize_with = "deserialize_lenient_f32")]
    start_time: Option<f32>,
    /// End time in seconds — analyse up to this point. Omit to analyse to the end.
    #[serde(default, deserialize_with = "deserialize_lenient_f32")]
    end_time: Option<f32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct FullAnalysisParams {
    /// Absolute path to the audio file
    path: String,
    /// Time-series resolution: "low" (~0.5/sec), "medium" (~1/sec), "high" (~4/sec), or a number. Omit for summary only. Token budget: "high" on 60s ≈ 240 rows — prefer for ≤20s windows. "medium" on 60s ≈ 60 rows — good general purpose. "low" for structural overview of long sections. Auto-capped at 800 rows.
    resolution: Option<String>,
    /// Start time in seconds — analyse from this point. Omit to start from the beginning.
    #[serde(default, deserialize_with = "deserialize_lenient_f32")]
    start_time: Option<f32>,
    /// End time in seconds — analyse up to this point. Omit to analyse to the end.
    #[serde(default, deserialize_with = "deserialize_lenient_f32")]
    end_time: Option<f32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct CompareParams {
    /// Absolute path to the first audio file (your mix / Track A)
    path_a: String,
    /// Absolute path to the second audio file (the reference / Track B)
    path_b: String,
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

/// Result of loading and optionally slicing audio.
struct AnalysisInput {
    audio: audio_visualizer_rs::AudioData,
    spectrogram: spectral::Spectrogram,
    /// Time offset in seconds — non-zero when a start_time slice was applied.
    /// Add this to all output timestamps so they reflect absolute file position.
    time_offset: f32,
}

// Helper: load audio, optionally slice to a time range, and compute spectrogram.
fn load_and_analyse(
    path: &str,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    start_time: Option<f32>,
    end_time: Option<f32>,
) -> Result<AnalysisInput, String> {
    let mut audio = load_audio(path)?;

    // Apply time slicing if requested
    let time_offset = start_time.unwrap_or(0.0).max(0.0);
    let start_sample = (time_offset * audio.sample_rate as f32) as usize;
    let end_sample = end_time
        .map(|t| (t * audio.sample_rate as f32) as usize)
        .unwrap_or(audio.samples.len())
        .min(audio.samples.len());

    if start_sample >= end_sample || start_sample >= audio.samples.len() {
        return Err(format!(
            "Invalid time range: {:.1}s–{:.1}s (file is {:.1}s)",
            time_offset,
            end_time.unwrap_or(audio.duration as f32),
            audio.duration,
        ));
    }

    audio.samples = audio.samples[start_sample..end_sample].to_vec();
    audio.duration = audio.samples.len() as f64 / audio.sample_rate as f64;

    let spectrogram =
        spectral::compute_spectrogram(&audio.samples, audio.sample_rate, n_fft, hop_length);
    Ok(AnalysisInput {
        audio,
        spectrogram,
        time_offset,
    })
}

/// Load stereo audio with optional time slicing. Returns (left, right, channels, sample_rate).
fn load_stereo_sliced(
    path: &str,
    sample_rate: u32,
    start_time: Option<f32>,
    end_time: Option<f32>,
) -> Result<(Vec<f32>, Vec<f32>, u32), String> {
    let stereo_audio = load_audio_stereo(path)?;
    let time_offset = start_time.unwrap_or(0.0).max(0.0);
    let start_sample = (time_offset * sample_rate as f32) as usize;
    let end_sample = end_time
        .map(|t| (t * sample_rate as f32) as usize)
        .unwrap_or(stereo_audio.left.len())
        .min(stereo_audio.left.len());

    if start_sample >= end_sample || start_sample >= stereo_audio.left.len() {
        return Err("Invalid time range for stereo".to_string());
    }

    Ok((
        stereo_audio.left[start_sample..end_sample].to_vec(),
        stereo_audio.right[start_sample..end_sample].to_vec(),
        stereo_audio.channels,
    ))
}

/// Offset timestamps in downsampled f32 series by adding time_offset.
fn offset_times(series: &mut [(f32, f32)], offset: f32) {
    if offset > 0.0 {
        for (t, _) in series.iter_mut() {
            *t += offset;
        }
    }
}

/// Offset timestamps in downsampled array series by adding time_offset.
fn offset_times_array<const N: usize>(series: &mut [(f32, [f32; N])], offset: f32) {
    if offset > 0.0 {
        for (t, _) in series.iter_mut() {
            *t += offset;
        }
    }
}

/// Format the difference between measured LUFS and a platform target.
fn format_platform_diff(measured: f32, target: f32) -> String {
    let diff = measured - target;
    if diff.abs() < 0.5 {
        "on target".to_string()
    } else if diff > 0.0 {
        format!("down {:.1} dB", diff)
    } else {
        format!("up {:.1} dB", -diff)
    }
}

// `#[tool_router]` collects all #[tool] methods and builds the routing table.
// Notice the methods are NOT async — they're plain synchronous functions.
// The rmcp framework handles the async wrapping.
#[tool_router]
impl AudioAnalyzerServer {
    #[tool(
        description = "Get basic information about an audio file (duration, sample rate, sample count). Quick and cheap — use this first to confirm the file is readable and see how long it is before running heavier analysis.",
        annotations(
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    fn audio_info(&self, Parameters(params): Parameters<AudioInfoParams>) -> String {
        match load_audio(&params.path) {
            Ok(audio) => {
                format!(
                    "File: {}\nSample rate: {} Hz\nSamples: {}\nDuration: {:.2} seconds",
                    params.path,
                    audio.sample_rate,
                    audio.len(),
                    audio.duration,
                )
            }
            Err(e) => format!("Error: {}", e),
        }
    }

    #[tool(
        description = "Analyse spectral and temporal features: brightness (centroid), richness (bandwidth), energy distribution (rolloff), tonality (flatness), frequency band energy (sub-bass through brilliance — essential for mix diagnosis), spectral contrast (peak vs valley per band — reveals clarity vs muddiness), dynamic range (crest factor, loudness range, peak dBFS), LUFS loudness (EBU R128 integrated, true peak, LRA, streaming platform targets), stereo field (phase correlation, stereo width, balance, mono compatibility), loudness (RMS), texture (zero crossing rate), and timbre (MFCCs). Use when you need spectral detail without harmonic/rhythm overhead. Omit resolution for a quick summary; set resolution='low' for time-series overview; use start_time/end_time with resolution='high' to zoom into specific sections.",
        annotations(
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    fn spectral_features(&self, Parameters(params): Parameters<SpectralParams>) -> String {
        match load_and_analyse(
            &params.path,
            params.n_fft,
            params.hop_length,
            params.start_time,
            params.end_time,
        ) {
            Ok(AnalysisInput {
                audio,
                spectrogram,
                time_offset,
            }) => {
                let centroid = spectral::spectral_centroid(&spectrogram);
                let bandwidth = spectral::spectral_bandwidth(&spectrogram);
                let rolloff = spectral::spectral_rolloff(&spectrogram, None);
                let flatness = spectral::spectral_flatness(&spectrogram);
                let bands = spectral::frequency_band_energy(&spectrogram);
                let sc = spectral::spectral_contrast(&spectrogram, None);
                let rms =
                    temporal::rms_energy(&audio.samples, spectrogram.n_fft, spectrogram.hop_length);
                let zcr = temporal::zero_crossing_rate(
                    &audio.samples,
                    spectrogram.n_fft,
                    spectrogram.hop_length,
                );
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
                let mfcc_summary: Vec<String> = avg_mfcc
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| format!("MFCC-{}: {:.2}", i, v))
                    .collect();

                // Average band energies for summary
                let mut avg_bands = [0.0_f32; 7];
                for frame in &bands.band_energies {
                    for (i, &val) in frame.iter().enumerate() {
                        avg_bands[i] += val;
                    }
                }
                for val in &mut avg_bands {
                    *val /= bands.n_frames as f32;
                }
                let band_names = spectral::FREQUENCY_BANDS;
                let band_summary: Vec<String> = band_names
                    .iter()
                    .enumerate()
                    .map(|(i, &(name, lo, hi))| {
                        format!("{} ({:.0}–{:.0} Hz): {:.6}", name, lo, hi, avg_bands[i])
                    })
                    .collect();

                // Average spectral contrast for summary
                let mut avg_contrast = [0.0_f32; 7];
                for frame in &sc.contrast {
                    for (i, &val) in frame.iter().enumerate() {
                        avg_contrast[i] += val;
                    }
                }
                for val in &mut avg_contrast {
                    *val /= sc.n_frames as f32;
                }
                let contrast_summary: Vec<String> = band_names
                    .iter()
                    .enumerate()
                    .map(|(i, &(name, lo, hi))| {
                        format!(
                            "{} ({:.0}–{:.0} Hz): {:.1} dB",
                            name, lo, hi, avg_contrast[i]
                        )
                    })
                    .collect();

                let dr = temporal::dynamic_range(
                    &audio.samples,
                    spectrogram.n_fft,
                    spectrogram.hop_length,
                );

                // Load stereo for both LUFS (needs L/R per ITU-R BS.1770-4) and stereo analysis
                let stereo_loaded = load_stereo_sliced(
                    &params.path,
                    audio.sample_rate,
                    params.start_time,
                    params.end_time,
                )
                .ok();
                let lufs = if let Some((ref left, ref right, _)) = stereo_loaded {
                    temporal::measure_lufs_stereo(left, right, audio.sample_rate)
                } else {
                    temporal::measure_lufs(&audio.samples, audio.sample_rate)
                };

                let mut result = format!(
                    "Spectral Analysis: {}\n\
                     Duration: {:.2} sec | Frames: {} | Freq bins: {}\n\n\
                     Centroid (brightness):     avg {:.0} Hz — {}\n\
                     Bandwidth (richness):      avg {:.0} Hz — {}\n\
                     Rolloff (energy focus):    avg {:.0} Hz\n\
                     Flatness (tonality):       avg {:.4} — {}\n\
                     RMS Energy (loudness):     avg {:.4}\n\
                     Zero Crossing Rate:        avg {:.4} — {}\n\
                     MFCCs (timbre):            {}\n\
                     \nFrequency Band Energy:\n  {}\n\
                     \nSpectral Contrast (peak–valley):\n  {}\n\
                     \n── Dynamic Range ──\n\
                     Peak:            {:.2} dBFS\n\
                     Crest factor:    {:.1} dB — {}\n\
                     Loudness range:  {:.1} dB — {}\n\
                     Quiet sections:  {:.1} dBFS | Loud sections: {:.1} dBFS\n\
                     \n── Loudness (EBU R128) ──\n\
                     Integrated:      {:.1} LUFS\n\
                     True peak:       {:.1} dBTP\n\
                     Loudness range:  {:.1} LU\n\
                     Spotify (-14):   {} | Apple (-16): {} | YouTube (-14): {}\n",
                    params.path,
                    spectrogram.duration(),
                    spectrogram.n_frames,
                    spectrogram.n_freq_bins,
                    avg(&centroid),
                    if avg(&centroid) > 3000.0 {
                        "bright/sharp"
                    } else if avg(&centroid) > 1500.0 {
                        "moderate"
                    } else {
                        "dark/warm"
                    },
                    avg(&bandwidth),
                    if avg(&bandwidth) > 3000.0 {
                        "wide/complex"
                    } else if avg(&bandwidth) > 1500.0 {
                        "moderate"
                    } else {
                        "narrow/pure"
                    },
                    avg(&rolloff),
                    avg(&flatness),
                    if avg(&flatness) > 0.5 {
                        "noisy"
                    } else if avg(&flatness) > 0.1 {
                        "mixed tonal/noisy"
                    } else {
                        "strongly tonal"
                    },
                    avg(&rms),
                    avg(&zcr),
                    if avg(&zcr) > 0.1 {
                        "percussive/noisy"
                    } else if avg(&zcr) > 0.03 {
                        "mixed"
                    } else {
                        "tonal"
                    },
                    mfcc_summary.join(", "),
                    band_summary.join("\n  "),
                    contrast_summary.join("\n  "),
                    dr.peak_dbfs,
                    dr.overall_crest_db,
                    if dr.overall_crest_db > 12.0 {
                        "very dynamic"
                    } else if dr.overall_crest_db > 6.0 {
                        "healthy"
                    } else {
                        "compressed"
                    },
                    dr.loudness_range_db,
                    if dr.loudness_range_db > 12.0 {
                        "very dynamic"
                    } else if dr.loudness_range_db > 6.0 {
                        "moderate"
                    } else if dr.loudness_range_db > 3.0 {
                        "compressed"
                    } else {
                        "brick-walled"
                    },
                    dr.rms_5th_db,
                    dr.rms_95th_db,
                    lufs.integrated,
                    lufs.true_peak_dbtp,
                    lufs.loudness_range,
                    format_platform_diff(lufs.integrated, -14.0),
                    format_platform_diff(lufs.integrated, -16.0),
                    format_platform_diff(lufs.integrated, -14.0),
                );

                // Stereo analysis
                if let Some((left, right, channels)) = stereo_loaded {
                    let stereo_result = stereo::analyse_stereo(
                        &left,
                        &right,
                        channels,
                        spectrogram.n_fft,
                        spectrogram.hop_length,
                    );
                    let stereo_sum = stereo::stereo_summary(&stereo_result);
                    result.push_str("\n── Stereo Field ──\n");
                    result.push_str(&stereo::format_stereo_summary(&stereo_sum, channels));
                }

                if let Some(ref res) = params.resolution {
                    match downsample::resolution_to_fps(res) {
                        Ok(mut target_fps) => {
                            let fps = downsample::native_fps(
                                spectrogram.sample_rate,
                                spectrogram.hop_length,
                            );

                            const MAX_ROWS: usize = 800;
                            let expected_rows =
                                (audio.duration as f32 * target_fps).ceil() as usize;
                            if expected_rows > MAX_ROWS {
                                let original_fps = target_fps;
                                target_fps = MAX_ROWS as f32 / audio.duration as f32;
                                result.push_str(&format!(
                                    "\n⚠ Resolution auto-reduced from {:.1} to \
                                     {:.1} pts/sec ({} row cap) to fit context \
                                     window. Zoom into a shorter section with \
                                     start_time/end_time for higher detail.\n",
                                    original_fps, target_fps, MAX_ROWS,
                                ));
                            }

                            let mut ds_centroid =
                                downsample::downsample_f32(&centroid, fps, target_fps);
                            let mut ds_bandwidth =
                                downsample::downsample_f32(&bandwidth, fps, target_fps);
                            let mut ds_rolloff =
                                downsample::downsample_f32(&rolloff, fps, target_fps);
                            let mut ds_flatness =
                                downsample::downsample_f32(&flatness, fps, target_fps);
                            let mut ds_rms = downsample::downsample_f32(&rms, fps, target_fps);
                            let mut ds_zcr = downsample::downsample_f32(&zcr, fps, target_fps);
                            let mut ds_crest =
                                downsample::downsample_f32(&dr.crest_factor_db, fps, target_fps);
                            let mut ds_bands =
                                downsample::downsample_array(&bands.band_energies, fps, target_fps);
                            let mut ds_contrast =
                                downsample::downsample_array(&sc.contrast, fps, target_fps);
                            offset_times(&mut ds_centroid, time_offset);
                            offset_times(&mut ds_bandwidth, time_offset);
                            offset_times(&mut ds_rolloff, time_offset);
                            offset_times(&mut ds_flatness, time_offset);
                            offset_times(&mut ds_rms, time_offset);
                            offset_times(&mut ds_zcr, time_offset);
                            offset_times(&mut ds_crest, time_offset);
                            offset_times_array(&mut ds_bands, time_offset);
                            offset_times_array(&mut ds_contrast, time_offset);
                            result.push_str(&downsample::format_f32_series(
                                "Spectral/Temporal Features Over Time",
                                &[
                                    "centroid_hz",
                                    "bandwidth_hz",
                                    "rolloff_hz",
                                    "flatness",
                                    "rms",
                                    "zcr",
                                    "crest_db",
                                ],
                                &[
                                    &ds_centroid,
                                    &ds_bandwidth,
                                    &ds_rolloff,
                                    &ds_flatness,
                                    &ds_rms,
                                    &ds_zcr,
                                    &ds_crest,
                                ],
                            ));
                            let band_col_names: Vec<&str> = spectral::FREQUENCY_BANDS
                                .iter()
                                .map(|&(name, _, _)| name)
                                .collect();
                            result.push_str(&downsample::format_array_series(
                                "Frequency Band Energy Over Time",
                                &band_col_names,
                                &ds_bands,
                            ));
                            let contrast_col_names: Vec<&str> = spectral::FREQUENCY_BANDS
                                .iter()
                                .map(|&(name, _, _)| name)
                                .collect();
                            result.push_str(&downsample::format_array_series(
                                "Spectral Contrast Over Time",
                                &contrast_col_names,
                                &ds_contrast,
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

    #[tool(
        description = "Analyse harmonic content: key detection, pitch class distribution (which notes are prominent), and tonal relationships (tonnetz). Essential for understanding melody, chords, and harmony. Note: key detection uses major/minor profiles only — for modal music, check the pitch class distribution for the actual tonal centre. Omit resolution for a quick summary; set resolution='low' for time-series overview; use start_time/end_time with resolution='high' to zoom into specific sections.",
        annotations(
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    fn harmonic_analysis(&self, Parameters(params): Parameters<HarmonicParams>) -> String {
        match load_and_analyse(&params.path, None, None, params.start_time, params.end_time) {
            Ok(AnalysisInput {
                audio,
                spectrogram,
                time_offset,
            }) => {
                let chromagram =
                    harmonic::compute_chromagram(&audio.samples, audio.sample_rate, &spectrogram);
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

                let pitch_names = [
                    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
                ];
                let mut ranked: Vec<(usize, f32)> =
                    avg_chroma.iter().copied().enumerate().collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let mut result = format!(
                    "Harmonic Analysis: {}\n\nEstimated Key: {} {} (confidence: {:.3})\n\nPitch Class Distribution:\n",
                    params.path, key, mode, confidence,
                );

                for (idx, (pc, energy)) in ranked.iter().enumerate() {
                    let bar_len = (energy * 30.0) as usize;
                    let bar: String = "█".repeat(bar_len);
                    result.push_str(&format!(
                        "  {:>2}. {:<2} {:.3} {}\n",
                        idx + 1,
                        pitch_names[*pc],
                        energy,
                        bar
                    ));
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
                        Ok(mut target_fps) => {
                            let fps = downsample::native_fps(
                                spectrogram.sample_rate,
                                spectrogram.hop_length,
                            );

                            const MAX_ROWS: usize = 800;
                            let expected_rows =
                                (audio.duration as f32 * target_fps).ceil() as usize;
                            if expected_rows > MAX_ROWS {
                                let original_fps = target_fps;
                                target_fps = MAX_ROWS as f32 / audio.duration as f32;
                                result.push_str(&format!(
                                    "\n⚠ Resolution auto-reduced from {:.1} to \
                                     {:.1} pts/sec ({} row cap) to fit context \
                                     window. Zoom into a shorter section with \
                                     start_time/end_time for higher detail.\n",
                                    original_fps, target_fps, MAX_ROWS,
                                ));
                            }

                            let mut ds_chroma =
                                downsample::downsample_array(&chromagram.chroma, fps, target_fps);
                            offset_times_array(&mut ds_chroma, time_offset);
                            result.push_str(&downsample::format_array_series(
                                "Chromagram Over Time",
                                &[
                                    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
                                ],
                                &ds_chroma,
                            ));
                            let mut ds_tonnetz =
                                downsample::downsample_array(&tonnetz, fps, target_fps);
                            offset_times_array(&mut ds_tonnetz, time_offset);
                            result.push_str(&downsample::format_array_series(
                                "Tonnetz Over Time",
                                &[
                                    "fifth_sin",
                                    "fifth_cos",
                                    "min3_sin",
                                    "min3_cos",
                                    "maj3_sin",
                                    "maj3_cos",
                                ],
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

    #[tool(
        description = "Analyse rhythm: tempo estimation (BPM), beat positions, tempo stability, and beat statistics. Shows whether music has a steady beat or is free-tempo. Tempo detection may report half/double time on electronic music or solo instruments — use min_bpm/max_bpm to constrain if needed. Omit resolution for a quick summary; set resolution='low' for onset strength overview; use start_time/end_time with resolution='high' to zoom into specific sections.",
        annotations(
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    fn rhythm_analysis(&self, Parameters(params): Parameters<RhythmParams>) -> String {
        match load_and_analyse(&params.path, None, None, params.start_time, params.end_time) {
            Ok(AnalysisInput {
                audio,
                spectrogram,
                time_offset,
            }) => {
                let analysis = rhythm::analyse_rhythm(&spectrogram, params.min_bpm, params.max_bpm);

                let mut result = format!(
                    "Rhythm Analysis: {}\n\nEstimated Tempo: {:.1} BPM (confidence: {:.3})\nDetected Beats: {}\n",
                    params.path,
                    analysis.tempo_bpm,
                    analysis.tempo_confidence,
                    analysis.beat_times.len(),
                );

                if let Some(stats) = rhythm::beat_statistics(&analysis.beat_times) {
                    result.push_str(&format!(
                        "\nBeat Statistics:\n  Mean tempo: {:.1} BPM | Median: {:.1} BPM\n  Stability: {:.3} (0=erratic, 1=metronomic)\n  IBI std dev: {:.3} sec\n",
                        stats.mean_bpm, stats.median_bpm, stats.tempo_stability, stats.std_dev_ibi,
                    ));
                }

                if !analysis.beat_times.is_empty() {
                    let show = analysis.beat_times.len().min(20);
                    let beats: Vec<String> = analysis.beat_times[..show]
                        .iter()
                        .map(|t| format!("{:.2}s", t + time_offset))
                        .collect();
                    result.push_str(&format!("\nFirst {} beats: {}", show, beats.join(", ")));
                }

                if let Some(ref res) = params.resolution {
                    match downsample::resolution_to_fps(res) {
                        Ok(mut target_fps) => {
                            let fps = downsample::native_fps(
                                spectrogram.sample_rate,
                                spectrogram.hop_length,
                            );

                            const MAX_ROWS: usize = 800;
                            let expected_rows =
                                (audio.duration as f32 * target_fps).ceil() as usize;
                            if expected_rows > MAX_ROWS {
                                let original_fps = target_fps;
                                target_fps = MAX_ROWS as f32 / audio.duration as f32;
                                result.push_str(&format!(
                                    "\n⚠ Resolution auto-reduced from {:.1} to \
                                     {:.1} pts/sec ({} row cap) to fit context \
                                     window. Zoom into a shorter section with \
                                     start_time/end_time for higher detail.\n",
                                    original_fps, target_fps, MAX_ROWS,
                                ));
                            }

                            let mut ds_onset = downsample::downsample_f32(
                                &analysis.onset_envelope,
                                fps,
                                target_fps,
                            );
                            offset_times(&mut ds_onset, time_offset);
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

    #[tool(
        description = "Run complete analysis: basic info, spectral/temporal features (brightness, richness, loudness, texture, timbre, frequency band energy, spectral contrast, dynamic range), LUFS loudness (EBU R128 integrated, true peak, LRA, streaming platform targets), stereo field (phase correlation, width, balance, mono compatibility), harmonic content (key, notes), rhythm (tempo, beats), percussive character (attack sharpness, onset density, harmonic/percussive balance), and section boundaries (structural changes detected via multi-feature novelty — energy, spectral, harmonic, texture). Recommended workflow: (1) call with NO resolution to get summary + section boundaries, (2) use boundary timestamps to pick interesting sections, (3) call with start_time/end_time and resolution='high' on SHORT sections (≤20s). Token budget: resolution='high' on a 60s section returns ~240 rows (~20K tokens). Prefer resolution='medium' or 'low' for sections longer than 20s. The tool will auto-reduce resolution if output would exceed 800 rows.",
        annotations(
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    fn full_analysis(&self, Parameters(params): Parameters<FullAnalysisParams>) -> String {
        let start = std::time::Instant::now();

        match load_and_analyse(&params.path, None, None, params.start_time, params.end_time) {
            Ok(AnalysisInput {
                audio,
                spectrogram,
                time_offset,
            }) => {
                let centroid = spectral::spectral_centroid(&spectrogram);
                let bandwidth = spectral::spectral_bandwidth(&spectrogram);
                let rolloff = spectral::spectral_rolloff(&spectrogram, None);
                let flatness = spectral::spectral_flatness(&spectrogram);
                let bands = spectral::frequency_band_energy(&spectrogram);
                let sc = spectral::spectral_contrast(&spectrogram, None);
                let rms =
                    temporal::rms_energy(&audio.samples, spectrogram.n_fft, spectrogram.hop_length);
                let zcr = temporal::zero_crossing_rate(
                    &audio.samples,
                    spectrogram.n_fft,
                    spectrogram.hop_length,
                );
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

                // Average band energies for summary
                let mut avg_bands = [0.0_f32; 7];
                for frame in &bands.band_energies {
                    for (i, &val) in frame.iter().enumerate() {
                        avg_bands[i] += val;
                    }
                }
                for val in &mut avg_bands {
                    *val /= bands.n_frames as f32;
                }

                // Average spectral contrast for summary
                let mut avg_contrast = [0.0_f32; 7];
                for frame in &sc.contrast {
                    for (i, &val) in frame.iter().enumerate() {
                        avg_contrast[i] += val;
                    }
                }
                for val in &mut avg_contrast {
                    *val /= sc.n_frames as f32;
                }

                let chromagram =
                    harmonic::compute_chromagram(&audio.samples, audio.sample_rate, &spectrogram);
                let tonnetz = harmonic::compute_tonnetz(&chromagram);
                let (key, mode, key_confidence) = chromagram.estimate_key();

                let pitch_names = [
                    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
                ];
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
                let mut ranked: Vec<(usize, f32)> =
                    avg_chroma.iter().copied().enumerate().collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let rhythm_result = rhythm::analyse_rhythm(&spectrogram, None, None);
                let beat_stats = rhythm::beat_statistics(&rhythm_result.beat_times);

                let hpss_result = percussive::hpss(&spectrogram, None);
                let perc_feats = percussive::percussive_features(
                    &hpss_result,
                    audio.sample_rate,
                    spectrogram.hop_length,
                );

                let dr = temporal::dynamic_range(
                    &audio.samples,
                    spectrogram.n_fft,
                    spectrogram.hop_length,
                );

                // Load stereo early — needed for correct LUFS (ITU-R BS.1770-4 sums L/R power)
                // and for stereo analysis + time-series
                let stereo_loaded = load_stereo_sliced(
                    &params.path,
                    audio.sample_rate,
                    params.start_time,
                    params.end_time,
                )
                .ok();
                let lufs = if let Some((ref left, ref right, _)) = stereo_loaded {
                    temporal::measure_lufs_stereo(left, right, audio.sample_rate)
                } else {
                    temporal::measure_lufs(&audio.samples, audio.sample_rate)
                };

                let elapsed = start.elapsed();

                let section_info = if time_offset > 0.0
                    || params.start_time.is_some()
                    || params.end_time.is_some()
                {
                    let end_t = time_offset + audio.duration as f32;
                    format!("Section: {:.1}s–{:.1}s | ", time_offset, end_t)
                } else {
                    String::new()
                };

                let mut result = format!(
                    "═══ Full Audio Analysis ═══\n\
                     File: {}\n\
                     {}Duration: {:.2} sec | Sample rate: {} Hz | Samples: {}\n\
                     Analysis completed in: {:.2?}\n\n\
                     ── Spectral/Temporal Features ──\n\
                     Centroid (brightness):  {:.0} Hz — {}\n\
                     Bandwidth (richness):   {:.0} Hz — {}\n\
                     Rolloff (energy focus): {:.0} Hz\n\
                     Flatness (tonality):    {:.4} — {}\n\
                     RMS Energy (loudness):  {:.4}\n\
                     Zero Crossing Rate:     {:.4} — {}\n\
                     MFCCs (timbre):         [{}]\n\n\
                     ── Frequency Band Energy ──\n\
                     Sub bass  (20–60 Hz):     {:.6}\n\
                     Bass      (60–250 Hz):    {:.6}\n\
                     Low-mid   (250–500 Hz):   {:.6}\n\
                     Mid       (500–2k Hz):    {:.6}\n\
                     Upper-mid (2k–4k Hz):     {:.6}\n\
                     Presence  (4k–6k Hz):     {:.6}\n\
                     Brilliance(6k–20k Hz):    {:.6}\n\n\
                     ── Spectral Contrast (peak–valley dB) ──\n\
                     Sub bass  (20–60 Hz):     {:.1}\n\
                     Bass      (60–250 Hz):    {:.1}\n\
                     Low-mid   (250–500 Hz):   {:.1}\n\
                     Mid       (500–2k Hz):    {:.1}\n\
                     Upper-mid (2k–4k Hz):     {:.1}\n\
                     Presence  (4k–6k Hz):     {:.1}\n\
                     Brilliance(6k–20k Hz):    {:.1}\n\n\
                     ── Harmonic Content ──\n\
                     Estimated key: {} {} (confidence: {:.3})\n\
                     Top pitch classes:\n",
                    params.path,
                    section_info,
                    audio.duration,
                    audio.sample_rate,
                    audio.len(),
                    elapsed,
                    avg(&centroid),
                    if avg(&centroid) > 3000.0 {
                        "bright"
                    } else if avg(&centroid) > 1500.0 {
                        "moderate"
                    } else {
                        "warm/dark"
                    },
                    avg(&bandwidth),
                    if avg(&bandwidth) > 3000.0 {
                        "complex"
                    } else if avg(&bandwidth) > 1500.0 {
                        "moderate"
                    } else {
                        "pure/simple"
                    },
                    avg(&rolloff),
                    avg(&flatness),
                    if avg(&flatness) > 0.5 {
                        "noisy"
                    } else if avg(&flatness) > 0.1 {
                        "mixed"
                    } else {
                        "strongly tonal"
                    },
                    avg(&rms),
                    avg(&zcr),
                    if avg(&zcr) > 0.1 {
                        "percussive/noisy"
                    } else if avg(&zcr) > 0.03 {
                        "mixed"
                    } else {
                        "tonal"
                    },
                    avg_mfcc
                        .iter()
                        .map(|v| format!("{:.1}", v))
                        .collect::<Vec<_>>()
                        .join(", "),
                    avg_bands[0],
                    avg_bands[1],
                    avg_bands[2],
                    avg_bands[3],
                    avg_bands[4],
                    avg_bands[5],
                    avg_bands[6],
                    avg_contrast[0],
                    avg_contrast[1],
                    avg_contrast[2],
                    avg_contrast[3],
                    avg_contrast[4],
                    avg_contrast[5],
                    avg_contrast[6],
                    key,
                    mode,
                    key_confidence,
                );

                for (idx, (pc, energy)) in ranked.iter().take(6).enumerate() {
                    let bar_len = (energy * 25.0) as usize;
                    let bar: String = "█".repeat(bar_len);
                    result.push_str(&format!(
                        "  {:>2}. {:<2} {:.3} {}\n",
                        idx + 1,
                        pitch_names[*pc],
                        energy,
                        bar
                    ));
                }

                result.push_str(&format!(
                    "\n── Rhythm ──\nTempo: {:.1} BPM (confidence: {:.3})\nBeats detected: {}\n",
                    rhythm_result.tempo_bpm,
                    rhythm_result.tempo_confidence,
                    rhythm_result.beat_times.len(),
                ));

                if let Some(stats) = beat_stats {
                    result.push_str(&format!(
                        "Mean tempo: {:.1} BPM | Median: {:.1} BPM\nStability: {:.3} (0=free, 1=locked)\n",
                        stats.mean_bpm, stats.median_bpm, stats.tempo_stability,
                    ));
                }

                // Percussive analysis summary
                let avg_perc_ratio = avg(&perc_feats.percussive_ratio);
                let avg_onset_density = avg(&perc_feats.onset_density);
                let max_sharpness = perc_feats
                    .attack_sharpness
                    .iter()
                    .cloned()
                    .fold(0.0_f32, f32::max);
                result.push_str(&format!(
                    "\n── Percussive Character ──\n\
                     Percussive ratio:    {:.3} — {}\n\
                     Onset density:       {:.1}/sec — {}\n\
                     Peak attack sharp:   {:.3}\n",
                    avg_perc_ratio,
                    if avg_perc_ratio > 0.6 {
                        "percussion-dominated"
                    } else if avg_perc_ratio > 0.35 {
                        "balanced"
                    } else {
                        "harmony-dominated"
                    },
                    avg_onset_density,
                    if avg_onset_density > 4.0 {
                        "very dense"
                    } else if avg_onset_density > 2.0 {
                        "moderate"
                    } else if avg_onset_density > 0.5 {
                        "sparse"
                    } else {
                        "minimal"
                    },
                    max_sharpness,
                ));

                // Dynamic range summary
                result.push_str(&format!(
                    "\n── Dynamic Range ──\n\
                     Peak:            {:.2} dBFS\n\
                     Crest factor:    {:.1} dB — {}\n\
                     Loudness range:  {:.1} dB — {}\n\
                     Quiet sections:  {:.1} dBFS | Loud sections: {:.1} dBFS\n",
                    dr.peak_dbfs,
                    dr.overall_crest_db,
                    if dr.overall_crest_db > 12.0 {
                        "very dynamic"
                    } else if dr.overall_crest_db > 6.0 {
                        "healthy"
                    } else {
                        "compressed"
                    },
                    dr.loudness_range_db,
                    if dr.loudness_range_db > 12.0 {
                        "very dynamic"
                    } else if dr.loudness_range_db > 6.0 {
                        "moderate"
                    } else if dr.loudness_range_db > 3.0 {
                        "compressed"
                    } else {
                        "brick-walled"
                    },
                    dr.rms_5th_db,
                    dr.rms_95th_db,
                ));

                // LUFS loudness summary
                result.push_str(&format!(
                    "\n── Loudness (EBU R128) ──\n\
                     Integrated:      {:.1} LUFS\n\
                     True peak:       {:.1} dBTP\n\
                     Loudness range:  {:.1} LU\n\
                     Spotify (-14):   {} | Apple (-16): {} | YouTube (-14): {}\n",
                    lufs.integrated,
                    lufs.true_peak_dbtp,
                    lufs.loudness_range,
                    format_platform_diff(lufs.integrated, -14.0),
                    format_platform_diff(lufs.integrated, -16.0),
                    format_platform_diff(lufs.integrated, -14.0),
                ));

                // Stereo analysis — reuse the stereo data already loaded for LUFS
                let stereo_data = stereo_loaded.map(|(left, right, channels)| {
                    stereo::analyse_stereo(
                        &left,
                        &right,
                        channels,
                        spectrogram.n_fft,
                        spectrogram.hop_length,
                    )
                });

                if let Some(ref stereo_result) = stereo_data {
                    let stereo_sum = stereo::stereo_summary(stereo_result);
                    result.push_str("\n── Stereo Field ──\n");
                    result.push_str(&stereo::format_stereo_summary(
                        &stereo_sum,
                        stereo_result.source_channels,
                    ));
                }

                // Section boundary detection — only for full-track analysis,
                // not zoomed-in sections (where boundaries aren't meaningful)
                if params.start_time.is_none() && params.end_time.is_none() {
                    let section_result =
                        sections::detect_sections(&spectrogram, Some(&chromagram), None);
                    if !section_result.boundaries.is_empty() {
                        result.push_str("\n── Section Boundaries ──\n");
                        result.push_str(&sections::format_sections(&section_result));
                    }
                }

                // Append unified time-series table if resolution was requested
                if let Some(ref res) = params.resolution {
                    match downsample::resolution_to_fps(res) {
                        Ok(mut target_fps) => {
                            let fps = downsample::native_fps(
                                spectrogram.sample_rate,
                                spectrogram.hop_length,
                            );

                            // Auto-cap: keep output under 800 rows to avoid
                            // blowing context windows.  If the requested fps
                            // would produce too many rows, reduce it and note
                            // the change in the output.
                            const MAX_ROWS: usize = 800;
                            let expected_rows =
                                (audio.duration as f32 * target_fps).ceil() as usize;
                            if expected_rows > MAX_ROWS {
                                let original_fps = target_fps;
                                target_fps = MAX_ROWS as f32 / audio.duration as f32;
                                result.push_str(&format!(
                                    "\n⚠ Resolution auto-reduced from {:.1} to \
                                     {:.1} pts/sec ({} row cap) to fit context \
                                     window. Zoom into a shorter section with \
                                     start_time/end_time for higher detail.\n",
                                    original_fps, target_fps, MAX_ROWS,
                                ));
                            }

                            let mut ds_centroid =
                                downsample::downsample_f32(&centroid, fps, target_fps);
                            let mut ds_bandwidth =
                                downsample::downsample_f32(&bandwidth, fps, target_fps);
                            let mut ds_rolloff =
                                downsample::downsample_f32(&rolloff, fps, target_fps);
                            let mut ds_flatness =
                                downsample::downsample_f32(&flatness, fps, target_fps);
                            let mut ds_rms = downsample::downsample_f32(&rms, fps, target_fps);
                            let mut ds_zcr = downsample::downsample_f32(&zcr, fps, target_fps);
                            let mut ds_onset = downsample::downsample_f32(
                                &rhythm_result.onset_envelope,
                                fps,
                                target_fps,
                            );
                            let mut ds_perc_ratio = downsample::downsample_f32(
                                &perc_feats.percussive_ratio,
                                fps,
                                target_fps,
                            );
                            let mut ds_attack = downsample::downsample_f32(
                                &perc_feats.attack_sharpness,
                                fps,
                                target_fps,
                            );
                            let mut ds_density = downsample::downsample_f32(
                                &perc_feats.onset_density,
                                fps,
                                target_fps,
                            );
                            let mut ds_crest =
                                downsample::downsample_f32(&dr.crest_factor_db, fps, target_fps);

                            // Stereo time-series (if available)
                            let ds_phase_corr = stereo_data.as_ref().map(|s| {
                                let mut ds = downsample::downsample_f32(
                                    &s.phase_correlation,
                                    fps,
                                    target_fps,
                                );
                                offset_times(&mut ds, time_offset);
                                ds
                            });
                            let ds_stereo_width = stereo_data.as_ref().map(|s| {
                                let mut ds =
                                    downsample::downsample_f32(&s.stereo_width, fps, target_fps);
                                offset_times(&mut ds, time_offset);
                                ds
                            });
                            let ds_balance = stereo_data.as_ref().map(|s| {
                                let mut ds =
                                    downsample::downsample_f32(&s.balance, fps, target_fps);
                                offset_times(&mut ds, time_offset);
                                ds
                            });
                            let ds_mono_compat = stereo_data.as_ref().map(|s| {
                                let mut ds = downsample::downsample_f32(
                                    &s.mono_compatibility,
                                    fps,
                                    target_fps,
                                );
                                offset_times(&mut ds, time_offset);
                                ds
                            });

                            let mut ds_chroma =
                                downsample::downsample_array(&chromagram.chroma, fps, target_fps);
                            let mut ds_tonnetz =
                                downsample::downsample_array(&tonnetz, fps, target_fps);
                            let mut ds_bands =
                                downsample::downsample_array(&bands.band_energies, fps, target_fps);
                            let mut ds_contrast =
                                downsample::downsample_array(&sc.contrast, fps, target_fps);

                            offset_times(&mut ds_centroid, time_offset);
                            offset_times(&mut ds_bandwidth, time_offset);
                            offset_times(&mut ds_rolloff, time_offset);
                            offset_times(&mut ds_flatness, time_offset);
                            offset_times(&mut ds_rms, time_offset);
                            offset_times(&mut ds_zcr, time_offset);
                            offset_times(&mut ds_onset, time_offset);
                            offset_times(&mut ds_perc_ratio, time_offset);
                            offset_times(&mut ds_attack, time_offset);
                            offset_times(&mut ds_density, time_offset);
                            offset_times(&mut ds_crest, time_offset);
                            offset_times_array(&mut ds_chroma, time_offset);
                            offset_times_array(&mut ds_tonnetz, time_offset);
                            offset_times_array(&mut ds_bands, time_offset);
                            offset_times_array(&mut ds_contrast, time_offset);

                            let chroma_cols: &[&str] = &[
                                "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
                            ];
                            let tonnetz_cols: &[&str] = &[
                                "fifth_sin",
                                "fifth_cos",
                                "min3_sin",
                                "min3_cos",
                                "maj3_sin",
                                "maj3_cos",
                            ];
                            let band_cols: Vec<&str> = spectral::FREQUENCY_BANDS
                                .iter()
                                .map(|&(name, _, _)| name)
                                .collect();
                            let contrast_cols: Vec<&str> = spectral::FREQUENCY_BANDS
                                .iter()
                                .map(|&(name, _, _)| {
                                    // Prefix with "sc_" to distinguish from band energy columns
                                    match name {
                                        "sub_bass" => "sc_sub_bass",
                                        "bass" => "sc_bass",
                                        "low_mid" => "sc_low_mid",
                                        "mid" => "sc_mid",
                                        "upper_mid" => "sc_upper_mid",
                                        "presence" => "sc_presence",
                                        "brilliance" => "sc_brilliance",
                                        _ => name,
                                    }
                                })
                                .collect();

                            // Use the minimum length across all series — the chromagram
                            // uses a larger internal FFT (n_fft=8192) so it can produce
                            // fewer frames than the spectrogram, leading to fewer
                            // downsampled points. Without this, indexing out of bounds
                            // panics the tokio task and silently hangs the MCP response.
                            let n_points = ds_centroid
                                .len()
                                .min(ds_chroma.len())
                                .min(ds_tonnetz.len())
                                .min(ds_bands.len());

                            // Build f32 series list — stereo columns are optional
                            let mut f32_entries: Vec<(&[&str], &[(f32, f32)])> = vec![
                                (&["centroid_hz"], &ds_centroid[..]),
                                (&["bandwidth_hz"], &ds_bandwidth[..]),
                                (&["rolloff_hz"], &ds_rolloff[..]),
                                (&["flatness"], &ds_flatness[..]),
                                (&["rms"], &ds_rms[..]),
                                (&["zcr"], &ds_zcr[..]),
                                (&["onset"], &ds_onset[..]),
                                (&["perc_ratio"], &ds_perc_ratio[..]),
                                (&["attack"], &ds_attack[..]),
                                (&["density"], &ds_density[..]),
                                (&["crest_db"], &ds_crest[..]),
                            ];

                            // Stereo column name slices (need stable references)
                            let pc_cols: &[&str] = &["phase_corr"];
                            let sw_cols: &[&str] = &["stereo_w"];
                            let bal_cols: &[&str] = &["balance"];
                            let mc_cols: &[&str] = &["mono_compat"];

                            if let Some(ref ds) = ds_phase_corr {
                                f32_entries.push((pc_cols, &ds[..]));
                            }
                            if let Some(ref ds) = ds_stereo_width {
                                f32_entries.push((sw_cols, &ds[..]));
                            }
                            if let Some(ref ds) = ds_balance {
                                f32_entries.push((bal_cols, &ds[..]));
                            }
                            if let Some(ref ds) = ds_mono_compat {
                                f32_entries.push((mc_cols, &ds[..]));
                            }

                            result.push_str(&downsample::format_unified_timeseries(
                                n_points,
                                &f32_entries,
                                Some((&ds_chroma, chroma_cols)),
                                Some((&ds_tonnetz, tonnetz_cols)),
                                Some((&ds_bands, &band_cols)),
                                Some((&ds_contrast, &contrast_cols)),
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

    #[tool(
        description = "Compare two audio files side by side — your mix vs a reference track. Returns structured deltas for loudness (LUFS, true peak, LRA), dynamics (crest factor, loudness range), spectral balance (7 frequency bands in dB), spectral contrast, tonal character (brightness, richness), stereo field (phase, width, balance, mono compatibility), key, and tempo. No time-series — just the summary metrics that matter for mix comparison, in one compact table. Use this to diagnose how a mix differs from a reference and what to adjust.",
        annotations(
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    fn compare(&self, Parameters(params): Parameters<CompareParams>) -> String {
        compare::compare_tracks(&params.path_a, &params.path_b)
    }
}

// `#[tool_handler]` generates the `list_tools` and `call_tool` implementations
// on ServerHandler, wiring them to the tool_router field. Without this, the
// server connects but reports zero tools — the handler doesn't know where to
// find them. This was the missing piece!
#[tool_handler]
impl ServerHandler for AudioAnalyzerServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Audio analysis server for examining music files. \
                 Provides tools for spectral features (including frequency \
                 band energy for mix diagnosis, spectral contrast for \
                 clarity analysis, dynamic range, and LUFS loudness), \
                 stereo field analysis (phase correlation, width, balance, \
                 mono compatibility), harmonic analysis (key detection, \
                 pitch classes), and rhythm analysis (tempo, beats). \
                 Pass absolute file paths to the tools. \
                 This server reads files directly from the local filesystem — \
                 do not ask the user to upload files. Instead, ask them for \
                 the file path on their machine.\n\n\
                 Use the compare tool to A/B your mix against a reference track — \
                 it returns one compact diff table instead of two separate analyses.\n\n\
                 Recommended workflow for long tracks (>3 min): start with \
                 full_analysis at \"low\" resolution for an overview, identify \
                 sections of interest (breakdowns, drops, transitions, problem \
                 areas), then call again with start_time/end_time and \"high\" \
                 resolution to zoom in on specific sections. This saves tokens \
                 while giving detailed insight where it matters.\n\n\
                 Section boundaries are automatically included in full-track \
                 analysis (no start_time/end_time). They show where the music \
                 changes structurally — use these timestamps to guide your \
                 zoom-in calls instead of guessing.\n\n\
                 Key detection uses Krumhansl-Schmuckler profiles which only \
                 know major and minor modes. For modal music (dorian, mixolydian, \
                 etc.), the detected key will be the closest major/minor relative — \
                 check the pitch class distribution for the actual tonal centre. \
                 Tempo detection can report half or double time on electronic music \
                 and solo instruments.",
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
    let service = server.serve(rmcp::transport::stdio()).await?;

    tracing::info!("Server running. Waiting for requests...");
    service.waiting().await?;
    tracing::info!("Server shutting down.");

    Ok(())
}
