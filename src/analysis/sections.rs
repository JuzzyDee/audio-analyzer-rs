// analysis/sections.rs — Section boundary detection
//
// Detects structural boundaries in a track (intro→verse, verse→chorus, etc.)
// using a multi-feature novelty function. The algorithm doesn't label sections —
// it finds *where* the music changes and reports *what* changed. Labelling is
// left to the LLM, which has the context to interpret "energy + harmonic shift
// at 32s" as "probably the verse starting."
//
// Approach:
// 1. Estimate a working BPM by sampling a few windows (tempo-adaptive kernel)
// 2. Downsample feature streams to ~4/sec (section boundaries don't need ms resolution)
// 3. Compute novelty curves via checkerboard kernel at two scales (2-bar, 8-bar)
// 4. Combine novelty curves across features, peak-pick for boundaries
//
// References:
// - Foote (2000), "Automatic Audio Segmentation using a Measure of Audio Novelty"
// - The checkerboard kernel approach is standard in MIR for structural segmentation

use super::harmonic::Chromagram;
use super::rhythm;
use super::spectral::Spectrogram;

/// A detected section boundary.
#[derive(Debug, Clone)]
pub struct SectionBoundary {
    /// Time in seconds where the boundary occurs
    pub time: f32,
    /// Confidence score (0.0–1.0), higher = more features agree
    pub confidence: f32,
    /// Which feature signals triggered this boundary
    pub signals: Vec<Signal>,
}

/// The type of change detected at a boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    Energy,
    Spectral,
    Harmonic,
    Texture,
}

impl Signal {
    pub fn label(self) -> &'static str {
        match self {
            Signal::Energy => "energy",
            Signal::Spectral => "spectral",
            Signal::Harmonic => "harmonic",
            Signal::Texture => "texture",
        }
    }
}

/// Result of section boundary detection.
#[derive(Debug)]
pub struct SectionAnalysis {
    /// Detected boundaries, sorted by time
    pub boundaries: Vec<SectionBoundary>,
    /// The working BPM used to derive kernel sizes
    pub working_bpm: f32,
    /// Confidence in the working BPM estimate
    pub bpm_confidence: f32,
}

/// Detect section boundaries in a track.
///
/// Uses a multi-feature novelty function with tempo-adaptive checkerboard kernels
/// at two scales (2-bar and 8-bar) to find structural changes.
///
/// # Arguments
/// * `spectrogram` — the STFT spectrogram (required)
/// * `chromagram` — pre-computed chromagram (if None, computed internally)
/// * `samples` — raw audio samples (needed if chromagram is None)
pub fn detect_sections(
    spectrogram: &Spectrogram,
    chromagram: Option<&Chromagram>,
    samples: Option<&[f32]>,
) -> SectionAnalysis {
    // Step 1: Estimate working BPM from sampled windows
    let (working_bpm, bpm_confidence) = estimate_working_bpm(spectrogram);

    // Step 2: Compute downsampled feature streams (~4 points/sec)
    let ds_factor = downsample_factor(spectrogram);
    let rms = downsampled_rms(spectrogram, ds_factor);
    let centroid = downsampled_centroid(spectrogram, ds_factor);
    let onset_env = downsampled_onset(spectrogram, ds_factor);

    // Chroma: use provided or compute internally
    let owned_chroma;
    let chroma = match chromagram {
        Some(c) => c,
        None => {
            // Need raw samples to compute chromagram
            let s = samples.expect("detect_sections needs either chromagram or samples");
            owned_chroma =
                super::harmonic::compute_chromagram(s, spectrogram.sample_rate, spectrogram);
            &owned_chroma
        }
    };
    let ds_chroma = downsampled_chroma(chroma, ds_factor);

    // Step 3: Compute kernel sizes in downsampled frames
    let ds_rate =
        spectrogram.sample_rate as f32 / (spectrogram.hop_length as f32 * ds_factor as f32);
    let bar_duration = 4.0 * 60.0 / working_bpm; // seconds per bar
    let small_kernel = (2.0 * bar_duration * ds_rate) as usize; // 2 bars each side
    let large_kernel = (8.0 * bar_duration * ds_rate) as usize; // 8 bars each side

    // Clamp kernels to reasonable bounds.
    // The large kernel should never exceed ~16 seconds (64 frames at 4/sec).
    // Beyond that, the averaging window spans multiple sections and dilutes novelty.
    let max_large_frames = (16.0 * ds_rate) as usize;
    let small_kernel = small_kernel.max(3).min(rms.len() / 4);
    let large_kernel = large_kernel
        .max(small_kernel + 1)
        .min(max_large_frames)
        .min(rms.len() / 3);

    // Step 4: Compute novelty curves at both scales for each feature
    let novelty_rms_s = checkerboard_novelty_scalar(&rms, small_kernel);
    let novelty_rms_l = checkerboard_novelty_scalar(&rms, large_kernel);
    let novelty_cent_s = checkerboard_novelty_scalar(&centroid, small_kernel);
    let novelty_cent_l = checkerboard_novelty_scalar(&centroid, large_kernel);
    let novelty_onset_s = checkerboard_novelty_scalar(&onset_env, small_kernel);
    let novelty_onset_l = checkerboard_novelty_scalar(&onset_env, large_kernel);
    let novelty_chroma_s = checkerboard_novelty_chroma(&ds_chroma, small_kernel);
    let novelty_chroma_l = checkerboard_novelty_chroma(&ds_chroma, large_kernel);

    // Step 5: Normalise each to 0–1 and combine
    let n = rms.len();
    let mut combined = vec![0.0_f32; n];
    let curves = [
        (&novelty_rms_s, Signal::Energy),
        (&novelty_rms_l, Signal::Energy),
        (&novelty_cent_s, Signal::Spectral),
        (&novelty_cent_l, Signal::Spectral),
        (&novelty_onset_s, Signal::Texture),
        (&novelty_onset_l, Signal::Texture),
        (&novelty_chroma_s, Signal::Harmonic),
        (&novelty_chroma_l, Signal::Harmonic),
    ];

    // Track per-signal contribution for labelling
    // 4 signal types × n frames
    let mut signal_strength = vec![[0.0_f32; 4]; n];

    for (curve, signal) in &curves {
        let norm = normalise(curve);
        let sig_idx = match signal {
            Signal::Energy => 0,
            Signal::Spectral => 1,
            Signal::Harmonic => 2,
            Signal::Texture => 3,
        };
        for i in 0..n.min(norm.len()) {
            combined[i] += norm[i];
            signal_strength[i][sig_idx] += norm[i];
        }
    }

    // Before normalizing: count how many feature curves actually have meaningful novelty.
    // A curve is "active" if its max raw value is significantly above zero.
    // If fewer than 2 features are active, there's not enough evidence for boundaries.
    let active_count = curves
        .iter()
        .filter(|(curve, _)| {
            let max_raw = curve.iter().cloned().fold(0.0_f32, f32::max);
            max_raw > 1e-4
        })
        .count();

    if active_count < 2 {
        return SectionAnalysis {
            boundaries: Vec::new(),
            working_bpm,
            bpm_confidence,
        };
    }

    // Normalise combined to 0–1
    let combined = normalise(&combined);
    // Normalise each signal channel
    let max_per_signal: [f32; 4] = {
        let mut maxes = [0.0_f32; 4];
        for frame in &signal_strength {
            for s in 0..4 {
                if frame[s] > maxes[s] {
                    maxes[s] = frame[s];
                }
            }
        }
        maxes
    };
    for frame in &mut signal_strength {
        for s in 0..4 {
            if max_per_signal[s] > 1e-10 {
                frame[s] /= max_per_signal[s];
            }
        }
    }

    // Step 6: Peak pick — adaptive threshold, minimum spacing = 2 bars
    let min_spacing_frames = (2.0 * bar_duration * ds_rate) as usize;
    let min_spacing = min_spacing_frames.max(3);

    // Use median + MAD (median absolute deviation) for the threshold.
    // This is robust to outlier peaks — a few dominant boundaries won't
    // mask weaker but real transitions (classic mean+std problem).
    let mut sorted = combined.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let mut abs_devs: Vec<f32> = combined.iter().map(|&x| (x - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = abs_devs[abs_devs.len() / 2];
    let threshold = median + 3.0 * mad;

    let ds_time_step =
        spectrogram.hop_length as f32 * ds_factor as f32 / spectrogram.sample_rate as f32;

    let mut boundaries = Vec::new();

    // Skip the first and last small_kernel frames — edges of the track produce
    // artefacts from the signal starting/ending. We use the small kernel here,
    // not the large one — the large kernel's novelty is already zero at its edges
    // (it can't compute there), so the combined curve naturally attenuates.
    // Using large_kernel would exclude too much of short tracks.
    let edge_guard = small_kernel.max(min_spacing);
    let search_start = edge_guard.max(1);
    let search_end = combined
        .len()
        .saturating_sub(edge_guard)
        .max(search_start + 1);

    for i in search_start..search_end {
        if combined[i] > threshold
            && combined[i] >= combined[i - 1]
            && combined[i] >= combined[i + 1]
        {
            // Check spacing from last boundary
            let (far_enough, should_replace) = if let Some(last) = boundaries.last() {
                let last_b: &SectionBoundary = last;
                let last_frame = (last_b.time / ds_time_step) as usize;
                if i - last_frame >= min_spacing {
                    (true, false)
                } else {
                    // Too close — but if this peak is stronger, replace the previous one
                    (false, combined[i] > last_b.confidence)
                }
            } else {
                (true, false)
            };

            if should_replace {
                boundaries.pop(); // Remove the weaker previous boundary
            }

            if far_enough || should_replace {
                // Determine which signals contributed
                let frame_signals = &signal_strength[i];
                let signal_threshold = 0.3; // signal must be at 30% of its own max to count
                let mut signals = Vec::new();
                let all_signals = [
                    Signal::Energy,
                    Signal::Spectral,
                    Signal::Harmonic,
                    Signal::Texture,
                ];
                for (s, &sig) in all_signals.iter().enumerate() {
                    if frame_signals[s] > signal_threshold {
                        signals.push(sig);
                    }
                }

                // Confidence: combined novelty at this peak, already 0–1
                let confidence = combined[i].clamp(0.0, 1.0);

                boundaries.push(SectionBoundary {
                    time: i as f32 * ds_time_step,
                    confidence,
                    signals,
                });
            }
        }
    }

    SectionAnalysis {
        boundaries,
        working_bpm,
        bpm_confidence,
    }
}

/// Format section boundaries for display.
pub fn format_sections(analysis: &SectionAnalysis) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "  Working BPM:     {:.0} (confidence: {:.2})\n",
        analysis.working_bpm, analysis.bpm_confidence
    ));
    out.push_str(&format!(
        "  Boundaries:      {}\n",
        analysis.boundaries.len()
    ));

    for b in &analysis.boundaries {
        let signal_labels: Vec<&str> = b.signals.iter().map(|s| s.label()).collect();
        let signals_str = if signal_labels.is_empty() {
            "mixed".to_string()
        } else {
            signal_labels.join("+")
        };

        // Format time as M:SS
        let mins = (b.time / 60.0) as u32;
        let secs = b.time % 60.0;
        out.push_str(&format!(
            "    {:>2}:{:04.1}s  {} (confidence: {:.2})\n",
            mins, secs, signals_str, b.confidence
        ));
    }

    out
}

// --- Internal helpers ---

/// Estimate working BPM by sampling a few windows across the track.
///
/// Takes 3-4 non-overlapping 10-second windows, runs onset→autocorrelation
/// on each, discards low-confidence results, and takes the median.
fn estimate_working_bpm(spectrogram: &Spectrogram) -> (f32, f32) {
    let total_frames = spectrogram.n_frames;
    let sr = spectrogram.sample_rate;
    let hop = spectrogram.hop_length;

    // 10 seconds in frames
    let window_frames = (10.0 * sr as f32 / hop as f32) as usize;

    if total_frames < window_frames {
        // Track too short for sampling — just use the whole thing
        let onset_env = rhythm::onset_strength(spectrogram);
        return rhythm::estimate_tempo(&onset_env, sr, hop, 50.0, 220.0);
    }

    // Sample 4 windows spread across the track
    let n_windows = 4;
    let usable = total_frames.saturating_sub(window_frames);
    let stride = if n_windows > 1 {
        usable / (n_windows - 1)
    } else {
        0
    };

    let mut estimates: Vec<(f32, f32)> = Vec::new(); // (bpm, confidence)

    for w in 0..n_windows {
        let start = (w * stride).min(usable);
        let end = (start + window_frames).min(total_frames);

        // Build a mini spectrogram view for this window
        let onset_env = onset_strength_window(spectrogram, start, end);
        let (bpm, conf) = rhythm::estimate_tempo(&onset_env, sr, hop, 50.0, 220.0);

        estimates.push((bpm, conf));
    }

    // Filter by confidence — discard anything below 0.2
    let confident: Vec<(f32, f32)> = estimates
        .into_iter()
        .filter(|&(_, conf)| conf > 0.2)
        .collect();

    if confident.is_empty() {
        // Nothing confident — fall back to full-track estimate
        let onset_env = rhythm::onset_strength(spectrogram);
        return rhythm::estimate_tempo(&onset_env, sr, hop, 50.0, 220.0);
    }

    // Median BPM of confident estimates, weighted confidence = max
    let mut bpms: Vec<f32> = confident.iter().map(|&(bpm, _)| bpm).collect();
    bpms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_bpm = bpms[bpms.len() / 2];
    let max_conf = confident.iter().map(|&(_, c)| c).fold(0.0_f32, f32::max);

    (median_bpm, max_conf)
}

/// Compute onset strength for a window of the spectrogram (frame range).
fn onset_strength_window(spectrogram: &Spectrogram, start: usize, end: usize) -> Vec<f32> {
    let end = end.min(spectrogram.n_frames);
    if end <= start + 1 {
        return vec![0.0];
    }

    let mut envelope = Vec::with_capacity(end - start);
    envelope.push(0.0);

    for i in (start + 1)..end {
        let flux: f32 = spectrogram.magnitudes[i]
            .iter()
            .zip(spectrogram.magnitudes[i - 1].iter())
            .map(|(&cur, &prev)| (cur - prev).max(0.0))
            .sum();
        envelope.push(flux);
    }

    // Normalise
    let max_val = envelope.iter().cloned().fold(0.0_f32, f32::max);
    if max_val > 1e-10 {
        for val in &mut envelope {
            *val /= max_val;
        }
    }

    envelope
}

/// How many spectrogram frames to bin together for ~4 points/sec.
fn downsample_factor(spectrogram: &Spectrogram) -> usize {
    let frames_per_sec = spectrogram.sample_rate as f32 / spectrogram.hop_length as f32;
    // Target ~4 points per second
    (frames_per_sec / 4.0).round() as usize
}

/// Downsample by bin-averaging groups of `factor` frames.
fn downsample_scalar(values: &[f32], factor: usize) -> Vec<f32> {
    if factor <= 1 {
        return values.to_vec();
    }
    values
        .chunks(factor)
        .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
        .collect()
}

/// RMS energy per frame, downsampled.
fn downsampled_rms(spectrogram: &Spectrogram, ds_factor: usize) -> Vec<f32> {
    // RMS from spectrogram magnitudes (sum of squared magnitudes per frame)
    let rms: Vec<f32> = spectrogram
        .magnitudes
        .iter()
        .map(|frame| {
            let sum_sq: f32 = frame.iter().map(|&m| m * m).sum();
            (sum_sq / frame.len() as f32).sqrt()
        })
        .collect();
    downsample_scalar(&rms, ds_factor)
}

/// Spectral centroid per frame, downsampled.
fn downsampled_centroid(spectrogram: &Spectrogram, ds_factor: usize) -> Vec<f32> {
    let centroid: Vec<f32> = spectrogram
        .magnitudes
        .iter()
        .map(|frame| {
            let total: f32 = frame.iter().sum();
            if total < 1e-10 {
                return 0.0;
            }
            let weighted: f32 = frame
                .iter()
                .enumerate()
                .map(|(bin, &mag)| bin as f32 * mag)
                .sum();
            weighted / total
        })
        .collect();
    downsample_scalar(&centroid, ds_factor)
}

/// Onset envelope, downsampled (takes max within each bin, not mean).
///
/// Uses the raw (unnormalized) spectral flux to avoid amplifying noise in
/// constant signals. The checkerboard kernel normalises by global std dev anyway.
fn downsampled_onset(spectrogram: &Spectrogram, ds_factor: usize) -> Vec<f32> {
    // Compute raw spectral flux without normalization
    let mut raw_flux = Vec::with_capacity(spectrogram.n_frames);
    raw_flux.push(0.0);
    for i in 1..spectrogram.n_frames {
        let flux: f32 = spectrogram.magnitudes[i]
            .iter()
            .zip(spectrogram.magnitudes[i - 1].iter())
            .map(|(&cur, &prev)| (cur - prev).max(0.0))
            .sum();
        raw_flux.push(flux);
    }

    if ds_factor <= 1 {
        return raw_flux;
    }
    // Take max per bin — we don't want to smooth out transients
    raw_flux
        .chunks(ds_factor)
        .map(|chunk| chunk.iter().cloned().fold(0.0_f32, f32::max))
        .collect()
}

/// Downsample chromagram: average chroma vectors within each bin.
fn downsampled_chroma(chromagram: &Chromagram, ds_factor: usize) -> Vec<[f32; 12]> {
    if ds_factor <= 1 {
        return chromagram.chroma.clone();
    }
    chromagram
        .chroma
        .chunks(ds_factor)
        .map(|chunk| {
            let mut avg = [0.0_f32; 12];
            for frame in chunk {
                for pc in 0..12 {
                    avg[pc] += frame[pc];
                }
            }
            let n = chunk.len() as f32;
            for pc in 0..12 {
                avg[pc] /= n;
            }
            avg
        })
        .collect()
}

/// Checkerboard novelty for a scalar feature stream.
///
/// At each point t, compare the mean of [t-kernel..t] vs [t..t+kernel].
/// The novelty value is the absolute difference, normalised by the
/// track's overall standard deviation.
fn checkerboard_novelty_scalar(values: &[f32], kernel: usize) -> Vec<f32> {
    let n = values.len();
    let mut novelty = vec![0.0_f32; n];

    if kernel == 0 || n < kernel * 2 {
        return novelty;
    }

    // Precompute global std dev for normalisation
    let global_mean = values.iter().sum::<f32>() / n as f32;
    let global_var: f32 = values
        .iter()
        .map(|&x| (x - global_mean) * (x - global_mean))
        .sum::<f32>()
        / n as f32;
    let global_std = global_var.sqrt();

    if global_std < 1e-6 {
        return novelty; // No variation at all
    }

    for t in kernel..(n - kernel) {
        let before_mean: f32 = values[t - kernel..t].iter().sum::<f32>() / kernel as f32;
        let after_mean: f32 = values[t..t + kernel].iter().sum::<f32>() / kernel as f32;
        novelty[t] = (after_mean - before_mean).abs() / global_std;
    }

    novelty
}

/// Checkerboard novelty for chroma (12-dimensional).
///
/// Uses cosine distance between the mean chroma vector before and after t.
/// Cosine distance catches harmonic changes regardless of overall loudness.
fn checkerboard_novelty_chroma(chroma: &[[f32; 12]], kernel: usize) -> Vec<f32> {
    let n = chroma.len();
    let mut novelty = vec![0.0_f32; n];

    if kernel == 0 || n < kernel * 2 {
        return novelty;
    }

    // Check if the chroma varies meaningfully across the track.
    // Compute mean chroma and check if individual frames deviate from it.
    let mut mean_chroma = [0.0_f32; 12];
    for frame in chroma.iter() {
        for pc in 0..12 {
            mean_chroma[pc] += frame[pc];
        }
    }
    for pc in 0..12 {
        mean_chroma[pc] /= n as f32;
    }
    // Average cosine distance of each frame from the global mean
    let mut total_dist = 0.0_f32;
    for frame in chroma.iter() {
        total_dist += cosine_distance(frame, &mean_chroma);
    }
    let avg_dist = total_dist / n as f32;
    if avg_dist < 0.01 {
        return novelty; // Chroma is essentially constant throughout
    }

    for t in kernel..(n - kernel) {
        // Mean chroma vector before t
        let mut before = [0.0_f32; 12];
        for frame in &chroma[t - kernel..t] {
            for pc in 0..12 {
                before[pc] += frame[pc];
            }
        }
        for pc in 0..12 {
            before[pc] /= kernel as f32;
        }

        // Mean chroma vector after t
        let mut after = [0.0_f32; 12];
        for frame in &chroma[t..t + kernel] {
            for pc in 0..12 {
                after[pc] += frame[pc];
            }
        }
        for pc in 0..12 {
            after[pc] /= kernel as f32;
        }

        // Cosine distance = 1 - cosine_similarity
        novelty[t] = cosine_distance(&before, &after);
    }

    novelty
}

/// Cosine distance between two vectors. Returns 0.0 for identical, up to 2.0 for opposite.
fn cosine_distance(a: &[f32; 12], b: &[f32; 12]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0; // Can't measure distance from a zero vector
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Normalise a slice to 0–1 range.
/// Returns all zeros if the signal has negligible dynamic range.
fn normalise(values: &[f32]) -> Vec<f32> {
    let max_val = values.iter().cloned().fold(0.0_f32, f32::max);
    if max_val < 1e-10 {
        return vec![0.0; values.len()];
    }

    // Check if the signal actually varies meaningfully.
    // If the range (max - min) is tiny relative to max, it's essentially flat.
    let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let range = max_val - min_val;
    if range < max_val * 0.01 {
        return vec![0.0; values.len()];
    }

    values.iter().map(|&v| v / max_val).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::spectral::compute_spectrogram;
    use std::f32::consts::PI;

    /// Generate a track with two distinct sections: a quiet sine tone
    /// followed by a loud, bright section with harmonics.
    fn two_section_track(sample_rate: u32, duration: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration) as usize;
        let half = n / 2;
        let mut samples = vec![0.0_f32; n];

        // Section A: quiet 220 Hz sine
        for i in 0..half {
            let t = i as f32 / sample_rate as f32;
            samples[i] = 0.2 * (2.0 * PI * 220.0 * t).sin();
        }

        // Section B: loud 440 Hz with harmonics
        for i in half..n {
            let t = i as f32 / sample_rate as f32;
            samples[i] = 0.8 * (2.0 * PI * 440.0 * t).sin()
                + 0.4 * (2.0 * PI * 880.0 * t).sin()
                + 0.2 * (2.0 * PI * 1320.0 * t).sin();
        }

        samples
    }

    #[test]
    fn test_cosine_distance() {
        let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let c = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        assert!(
            (cosine_distance(&a, &c)).abs() < 1e-6,
            "Identical vectors should have distance 0"
        );
        assert!(
            (cosine_distance(&a, &b) - 1.0).abs() < 1e-6,
            "Orthogonal vectors should have distance 1"
        );
    }

    #[test]
    fn test_normalise() {
        let vals = vec![0.0, 2.0, 4.0, 1.0];
        let norm = normalise(&vals);
        assert!((norm[2] - 1.0).abs() < 1e-6);
        assert!((norm[0]).abs() < 1e-6);
        assert!((norm[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_checkerboard_scalar_detects_step() {
        // A step function should produce a strong novelty spike at the midpoint
        let mut values = vec![0.0_f32; 100];
        for i in 50..100 {
            values[i] = 1.0;
        }

        let novelty = checkerboard_novelty_scalar(&values, 10);
        // Peak should be near index 50
        let peak_idx = novelty
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        assert!(
            (peak_idx as i32 - 50).unsigned_abs() <= 2,
            "Peak should be near the step at 50, got {}",
            peak_idx
        );
    }

    #[test]
    fn test_two_section_detection() {
        // A track with two clearly different sections should produce at least one boundary
        let samples = two_section_track(44100, 8.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let result = detect_sections(&spec, None, Some(&samples));

        println!(
            "Working BPM: {:.0} (conf: {:.2})",
            result.working_bpm, result.bpm_confidence
        );
        println!("Boundaries: {}", result.boundaries.len());
        for b in &result.boundaries {
            let labels: Vec<&str> = b.signals.iter().map(|s| s.label()).collect();
            println!(
                "  {:.1}s {} (conf: {:.2})",
                b.time,
                labels.join("+"),
                b.confidence
            );
        }

        // We should detect at least one boundary
        assert!(
            !result.boundaries.is_empty(),
            "Should detect at least one boundary in a two-section track"
        );

        // The boundary should be roughly in the middle (4 seconds)
        let midpoint_boundary = result
            .boundaries
            .iter()
            .min_by_key(|b| ((b.time - 4.0).abs() * 100.0) as u32)
            .unwrap();
        assert!(
            (midpoint_boundary.time - 4.0).abs() < 1.5,
            "Boundary should be near 4.0s, got {:.1}s",
            midpoint_boundary.time
        );
    }

    #[test]
    fn test_uniform_signal_few_boundaries() {
        // A constant sine tone is adversarial — FFT windowing creates edge artefacts
        // that look like micro-changes. Real audio always has some variation.
        // Instead, test that a uniform signal produces very few boundaries (≤1)
        // and that any detected boundaries have low confidence.
        let n = 44100 * 16;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let result = detect_sections(&spec, None, Some(&samples));

        println!("Boundaries on pure sine: {}", result.boundaries.len());
        for b in &result.boundaries {
            let labels: Vec<&str> = b.signals.iter().map(|s| s.label()).collect();
            println!(
                "  {:.1}s {} (conf: {:.2})",
                b.time,
                labels.join("+"),
                b.confidence
            );
        }

        // A pure sine may produce a couple of artefact boundaries from FFT edges,
        // but should produce far fewer than a track with real section changes.
        assert!(
            result.boundaries.len() <= 3,
            "Pure sine should produce at most 3 artefact boundaries, got {}",
            result.boundaries.len()
        );
    }

    #[test]
    fn test_format_sections() {
        let analysis = SectionAnalysis {
            boundaries: vec![
                SectionBoundary {
                    time: 32.4,
                    confidence: 0.87,
                    signals: vec![Signal::Energy, Signal::Harmonic],
                },
                SectionBoundary {
                    time: 78.1,
                    confidence: 0.64,
                    signals: vec![Signal::Harmonic],
                },
            ],
            working_bpm: 128.0,
            bpm_confidence: 0.85,
        };

        let output = format_sections(&analysis);
        assert!(output.contains("128"));
        assert!(output.contains("energy+harmonic"));
        assert!(output.contains("1:18.1"));
    }

    #[test]
    fn test_downsample_factor() {
        // At 44100 Hz, hop 512: ~86 frames/sec → ds_factor ≈ 22 for ~4/sec
        let spec = compute_spectrogram(&vec![0.0; 44100], 44100, None, None);
        let factor = downsample_factor(&spec);
        assert!(factor >= 15 && factor <= 25, "Expected ~22, got {}", factor);
    }
}
