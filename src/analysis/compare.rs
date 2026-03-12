// compare.rs — A/B comparison of two audio files
//
// Runs summary analysis on both tracks and computes structured deltas.
// No time-series, no HPSS — just the metrics that matter for mix comparison.
// Designed to be token-efficient: one compact diff table instead of two
// full analysis dumps.

use super::{harmonic, rhythm, spectral, stereo, temporal};
use crate::{load_audio, load_audio_stereo};

/// Summary metrics for one track, extracted for comparison.
struct TrackSummary {
    duration: f64,
    sample_rate: u32,

    // Spectral
    centroid: f32,
    bandwidth: f32,
    flatness: f32,

    // Band energy (7 bands)
    band_energy: [f32; 7],

    // Spectral contrast (7 bands)
    contrast: [f32; 7],

    // Dynamic range
    peak_dbfs: f32,
    crest_db: f32,
    loudness_range_db: f32,

    // LUFS
    lufs_integrated: f32,
    true_peak_dbtp: f32,
    lufs_lra: f32,

    // Stereo
    stereo: Option<stereo::StereoSummary>,

    // Harmonic
    key: String,
    mode: String,

    // Rhythm
    tempo: f32,
}

fn analyse_track(path: &str) -> Result<TrackSummary, String> {
    let audio = load_audio(path)?;
    let spectrogram = spectral::compute_spectrogram(&audio.samples, audio.sample_rate, None, None);

    let centroid = spectral::spectral_centroid(&spectrogram);
    let bandwidth = spectral::spectral_bandwidth(&spectrogram);
    let flatness = spectral::spectral_flatness(&spectrogram);
    let bands = spectral::frequency_band_energy(&spectrogram);
    let sc = spectral::spectral_contrast(&spectrogram, None);
    let dr = temporal::dynamic_range(&audio.samples, spectrogram.n_fft, spectrogram.hop_length);

    let chromagram = harmonic::compute_chromagram(&audio.samples, audio.sample_rate, &spectrogram);
    let (key, mode, _key_confidence) = chromagram.estimate_key();

    let rhythm_result = rhythm::analyse_rhythm(&spectrogram, None, None);

    // Load stereo for LUFS and stereo analysis
    let stereo_audio = load_audio_stereo(path).ok();
    let lufs = if let Some(ref sa) = stereo_audio {
        temporal::measure_lufs_stereo(&sa.left, &sa.right, audio.sample_rate)
    } else {
        temporal::measure_lufs(&audio.samples, audio.sample_rate)
    };

    let stereo_result = stereo_audio.as_ref().map(|sa| {
        let analysis = stereo::analyse_stereo(
            &sa.left,
            &sa.right,
            sa.channels,
            spectrogram.n_fft,
            spectrogram.hop_length,
        );
        stereo::stereo_summary(&analysis)
    });

    let avg = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;

    // Average band energies
    let mut avg_bands = [0.0_f32; 7];
    for frame in &bands.band_energies {
        for (i, &val) in frame.iter().enumerate() {
            avg_bands[i] += val;
        }
    }
    for val in &mut avg_bands {
        *val /= bands.n_frames as f32;
    }

    // Average spectral contrast
    let mut avg_contrast = [0.0_f32; 7];
    for frame in &sc.contrast {
        for (i, &val) in frame.iter().enumerate() {
            avg_contrast[i] += val;
        }
    }
    for val in &mut avg_contrast {
        *val /= sc.n_frames as f32;
    }

    Ok(TrackSummary {
        duration: audio.duration,
        sample_rate: audio.sample_rate,
        centroid: avg(&centroid),
        bandwidth: avg(&bandwidth),
        flatness: avg(&flatness),
        band_energy: avg_bands,
        contrast: avg_contrast,
        peak_dbfs: dr.peak_dbfs,
        crest_db: dr.overall_crest_db,
        loudness_range_db: dr.loudness_range_db,
        lufs_integrated: lufs.integrated,
        true_peak_dbtp: lufs.true_peak_dbtp,
        lufs_lra: lufs.loudness_range,
        stereo: stereo_result,
        key: key.to_string(),
        mode: mode.to_string(),
        tempo: rhythm_result.tempo_bpm,
    })
}

/// Format a dB delta with direction label.
fn fmt_db_delta(a: f32, b: f32, label_a: &str, label_b: &str) -> String {
    let diff = a - b;
    if diff.abs() < 0.5 {
        "~same".to_string()
    } else if diff > 0.0 {
        format!("{} +{:.1} dB", label_a, diff)
    } else {
        format!("{} +{:.1} dB", label_b, -diff)
    }
}

/// Format an energy delta in dB (converting from linear RMS).
fn fmt_energy_delta_db(a: f32, b: f32, label_a: &str, label_b: &str) -> String {
    if a <= 0.0 || b <= 0.0 {
        return "—".to_string();
    }
    let db_diff = 20.0 * (a / b).log10();
    if db_diff.abs() < 0.5 {
        "~same".to_string()
    } else if db_diff > 0.0 {
        format!("{} +{:.1} dB", label_a, db_diff)
    } else {
        format!("{} +{:.1} dB", label_b, -db_diff)
    }
}

/// Compare two audio files and return a formatted comparison string.
pub fn compare_tracks(path_a: &str, path_b: &str) -> String {
    let start = std::time::Instant::now();

    let a = match analyse_track(path_a) {
        Ok(s) => s,
        Err(e) => return format!("Error analysing Track A: {}", e),
    };
    let b = match analyse_track(path_b) {
        Ok(s) => s,
        Err(e) => return format!("Error analysing Track B: {}", e),
    };

    let elapsed = start.elapsed();

    // Extract just filenames for cleaner display
    let name_a = std::path::Path::new(path_a)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path_a.to_string());
    let name_b = std::path::Path::new(path_b)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path_b.to_string());

    let mut out = format!(
        "═══ A/B Comparison ═══\n\
         Track A: {} ({:.1}s, {} Hz)\n\
         Track B: {} ({:.1}s, {} Hz)\n\
         Compared in: {:.2?}\n",
        name_a, a.duration, a.sample_rate, name_b, b.duration, b.sample_rate, elapsed,
    );

    // ── Loudness & Dynamics ──
    out.push_str(&format!(
        "\n── Loudness & Dynamics ──\n\
         {:22} {:>10} {:>10}   {}\n\
         {:22} {:>10} {:>10}   {}\n\
         {:22} {:>10} {:>10}   {}\n\
         {:22} {:>10} {:>10}   {}\n\
         {:22} {:>10} {:>10}   {}\n\
         {:22} {:>10} {:>10}   {}\n\
         {:22} {:>10} {:>10}   {}\n",
        "",
        "Track A",
        "Track B",
        "Delta",
        "LUFS integrated:",
        format!("{:.1}", a.lufs_integrated),
        format!("{:.1}", b.lufs_integrated),
        fmt_db_delta(a.lufs_integrated, b.lufs_integrated, "A louder", "B louder"),
        "True peak:",
        format!("{:.1} dBTP", a.true_peak_dbtp),
        format!("{:.1} dBTP", b.true_peak_dbtp),
        fmt_db_delta(a.true_peak_dbtp, b.true_peak_dbtp, "A higher", "B higher"),
        "LRA:",
        format!("{:.1} LU", a.lufs_lra),
        format!("{:.1} LU", b.lufs_lra),
        fmt_db_delta(a.lufs_lra, b.lufs_lra, "A wider", "B wider"),
        "Crest factor:",
        format!("{:.1} dB", a.crest_db),
        format!("{:.1} dB", b.crest_db),
        fmt_db_delta(a.crest_db, b.crest_db, "A more dynamic", "B more dynamic"),
        "Peak:",
        format!("{:.1} dBFS", a.peak_dbfs),
        format!("{:.1} dBFS", b.peak_dbfs),
        fmt_db_delta(a.peak_dbfs, b.peak_dbfs, "A higher", "B higher"),
        "Loudness range:",
        format!("{:.1} dB", a.loudness_range_db),
        format!("{:.1} dB", b.loudness_range_db),
        fmt_db_delta(
            a.loudness_range_db,
            b.loudness_range_db,
            "A wider",
            "B wider"
        ),
    ));

    // Streaming platform comparison
    let spotify_a = a.lufs_integrated - (-14.0);
    let spotify_b = b.lufs_integrated - (-14.0);
    let apple_a = a.lufs_integrated - (-16.0);
    let apple_b = b.lufs_integrated - (-16.0);
    let fmt_plat = |diff: f32| -> String {
        if diff.abs() < 0.5 {
            "on target".to_string()
        } else if diff > 0.0 {
            format!("down {:.1} dB", diff)
        } else {
            format!("up {:.1} dB", -diff)
        }
    };
    out.push_str(&format!(
        "\nStreaming targets:\n\
         {:22} {:>10} {:>10}\n\
         {:22} {:>10} {:>10}\n",
        "Spotify (-14 LUFS):",
        fmt_plat(spotify_a),
        fmt_plat(spotify_b),
        "Apple (-16 LUFS):",
        fmt_plat(apple_a),
        fmt_plat(apple_b),
    ));

    // ── Spectral Balance ──
    let short_band_names = [
        "Sub bass (20–60):",
        "Bass (60–250):",
        "Low-mid (250–500):",
        "Mid (500–2k):",
        "Upper-mid (2k–4k):",
        "Presence (4k–6k):",
        "Brilliance (6k–20k):",
    ];
    out.push_str(&format!(
        "\n── Spectral Balance ──\n\
         {:22} {:>10} {:>10}   {}\n",
        "", "Track A", "Track B", "Delta",
    ));
    for (i, label) in short_band_names.iter().enumerate() {
        out.push_str(&format!(
            "{:22} {:>10} {:>10}   {}\n",
            label,
            format!("{:.6}", a.band_energy[i]),
            format!("{:.6}", b.band_energy[i]),
            fmt_energy_delta_db(a.band_energy[i], b.band_energy[i], "A", "B"),
        ));
    }

    // ── Spectral Contrast ──
    out.push_str(&format!(
        "\n── Spectral Contrast (peak–valley dB) ──\n\
         {:22} {:>10} {:>10}   {}\n",
        "", "Track A", "Track B", "Delta",
    ));
    for (i, label) in short_band_names.iter().enumerate() {
        out.push_str(&format!(
            "{:22} {:>10} {:>10}   {}\n",
            label,
            format!("{:.1}", a.contrast[i]),
            format!("{:.1}", b.contrast[i]),
            fmt_db_delta(a.contrast[i], b.contrast[i], "A clearer", "B clearer"),
        ));
    }

    // ── Tonal Character ──
    out.push_str(&format!(
        "\n── Tonal Character ──\n\
         {:22} {:>10} {:>10}   {}\n\
         {:22} {:>10} {:>10}   {}\n\
         {:22} {:>10} {:>10}   {}\n\
         {:22} {:>10} {:>10}   {}\n",
        "",
        "Track A",
        "Track B",
        "Delta",
        "Brightness:",
        format!("{:.0} Hz", a.centroid),
        format!("{:.0} Hz", b.centroid),
        fmt_energy_delta_db(a.centroid, b.centroid, "A brighter", "B brighter"),
        "Richness:",
        format!("{:.0} Hz", a.bandwidth),
        format!("{:.0} Hz", b.bandwidth),
        fmt_energy_delta_db(a.bandwidth, b.bandwidth, "A richer", "B richer"),
        "Tonality:",
        format!("{:.4}", a.flatness),
        format!("{:.4}", b.flatness),
        if (a.flatness - b.flatness).abs() < 0.01 {
            "~same".to_string()
        } else if a.flatness > b.flatness {
            "A noisier".to_string()
        } else {
            "B noisier".to_string()
        },
    ));

    // ── Stereo Field ──
    if let (Some(sa), Some(sb)) = (&a.stereo, &b.stereo) {
        out.push_str(&format!(
            "\n── Stereo Field ──\n\
             {:22} {:>10} {:>10}   {}\n\
             {:22} {:>10} {:>10}   {}\n\
             {:22} {:>10} {:>10}   {}\n\
             {:22} {:>10} {:>10}   {}\n\
             {:22} {:>10} {:>10}   {}\n\
             {:22} {:>10} {:>10}   {}\n",
            "",
            "Track A",
            "Track B",
            "Delta",
            "Phase correlation:",
            format!("{:.3}", sa.avg_phase_correlation),
            format!("{:.3}", sb.avg_phase_correlation),
            if (sa.avg_phase_correlation - sb.avg_phase_correlation).abs() < 0.05 {
                "~same".to_string()
            } else if sa.avg_phase_correlation > sb.avg_phase_correlation {
                "A more coherent".to_string()
            } else {
                "B more coherent".to_string()
            },
            "Stereo width:",
            format!("{:.3}", sa.avg_stereo_width),
            format!("{:.3}", sb.avg_stereo_width),
            if (sa.avg_stereo_width - sb.avg_stereo_width).abs() < 0.05 {
                "~same".to_string()
            } else if sa.avg_stereo_width > sb.avg_stereo_width {
                "A wider".to_string()
            } else {
                "B wider".to_string()
            },
            "Balance:",
            format!("{:.3}", sa.avg_balance),
            format!("{:.3}", sb.avg_balance),
            if sa.avg_balance.abs() < 0.02 && sb.avg_balance.abs() < 0.02 {
                "both centred".to_string()
            } else {
                let a_dir = if sa.avg_balance < -0.02 {
                    "left"
                } else if sa.avg_balance > 0.02 {
                    "right"
                } else {
                    "centre"
                };
                let b_dir = if sb.avg_balance < -0.02 {
                    "left"
                } else if sb.avg_balance > 0.02 {
                    "right"
                } else {
                    "centre"
                };
                format!("A {}, B {}", a_dir, b_dir)
            },
            "Mono compatibility:",
            format!("{:.3}", sa.avg_mono_compatibility),
            format!("{:.3}", sb.avg_mono_compatibility),
            if (sa.avg_mono_compatibility - sb.avg_mono_compatibility).abs() < 0.05 {
                "~same".to_string()
            } else if sa.avg_mono_compatibility > sb.avg_mono_compatibility {
                "A safer".to_string()
            } else {
                "B safer".to_string()
            },
            "Phase warnings:",
            format!("{:.1}%", sa.phase_warning_fraction * 100.0),
            format!("{:.1}%", sb.phase_warning_fraction * 100.0),
            if (sa.phase_warning_fraction - sb.phase_warning_fraction).abs() < 0.02 {
                "~same".to_string()
            } else if sa.phase_warning_fraction > sb.phase_warning_fraction {
                "A more issues".to_string()
            } else {
                "B more issues".to_string()
            },
        ));
    }

    // ── Key & Tempo ──
    let key_match = if a.key == b.key && a.mode == b.mode {
        "match".to_string()
    } else {
        "different".to_string()
    };
    let tempo_delta = (a.tempo - b.tempo).abs();
    let tempo_match = if tempo_delta < 1.0 {
        "~match".to_string()
    } else {
        format!("{:.1} BPM apart", tempo_delta)
    };

    out.push_str(&format!(
        "\n── Key & Tempo ──\n\
         {:22} {:>10} {:>10}   {}\n\
         {:22} {:>10} {:>10}   {}\n",
        "Key:",
        format!("{} {}", a.key, a.mode),
        format!("{} {}", b.key, b.mode),
        key_match,
        "Tempo:",
        format!("{:.1} BPM", a.tempo),
        format!("{:.1} BPM", b.tempo),
        tempo_match,
    ));

    out
}
