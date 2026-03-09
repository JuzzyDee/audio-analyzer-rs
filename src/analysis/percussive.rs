// analysis/percussive.rs — Harmonic/Percussive Source Separation (HPSS) and attack features
//
// HPSS separates a spectrogram into two components:
// - Harmonic: sustained tones (notes, chords, vocals) — energy that persists
//   across time frames (horizontal lines in a spectrogram)
// - Percussive: transients (drum hits, plucks, consonants) — energy that
//   appears suddenly across many frequencies (vertical lines in a spectrogram)
//
// The technique: apply median filtering along two axes of the spectrogram.
// - Median across TIME at each frequency bin → harmonic component (smooths
//   out transients, keeps sustained energy)
// - Median across FREQUENCY at each time frame → percussive component (smooths
//   out harmonics, keeps broadband transients)
//
// Then use soft masking to assign each time-frequency cell to one component
// based on which median filter gives a higher value.
//
// From the percussive component, we extract attack features:
// - Onset density: how many transients per time window
// - Attack sharpness: how sudden the transients are (rise time)
// - Percussive energy ratio: percussive vs total energy

use super::spectral::Spectrogram;

/// Result of harmonic/percussive source separation.
#[derive(Debug)]
pub struct HpssResult {
    /// Harmonic spectrogram magnitudes (same shape as input)
    pub harmonic: Vec<Vec<f32>>,
    /// Percussive spectrogram magnitudes (same shape as input)
    pub percussive: Vec<Vec<f32>>,
    /// Number of time frames
    pub n_frames: usize,
    /// Number of frequency bins
    pub n_freq_bins: usize,
}

/// Per-frame percussive features, aligned to the spectrogram time axis.
#[derive(Debug)]
pub struct PercussiveFeatures {
    /// Percussive energy ratio per frame: percussive / (harmonic + percussive).
    /// 0.0 = purely harmonic, 1.0 = purely percussive.
    pub percussive_ratio: Vec<f32>,

    /// Attack sharpness per frame: how much the percussive energy increased
    /// compared to the previous frame. High values = sharp transient.
    /// Normalised to 0.0..1.0 range.
    pub attack_sharpness: Vec<f32>,

    /// Onset density per frame: rolling count of detected percussive onsets
    /// within a window around each frame, normalised to onsets per second.
    pub onset_density: Vec<f32>,
}

/// Compute the median of a mutable slice of f32 values.
///
/// Uses `select_nth_unstable` for O(n) average-case performance instead
/// of O(n log n) full sort. Critical for HPSS where we compute millions
/// of medians across the spectrogram.
fn median(values: &mut Vec<f32>) -> f32 {
    let len = values.len();
    if len == 0 {
        return 0.0;
    }
    let mid = len / 2;
    values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    if len % 2 == 0 {
        // For even length, we need the average of mid-1 and mid.
        // After select_nth at mid, elements before mid are <= values[mid]
        // but not sorted, so find the max of the left partition.
        let left_max = values[..mid].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        (left_max + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Perform Harmonic/Percussive Source Separation on a spectrogram.
///
/// # Arguments
/// * `spectrogram` - The input STFT spectrogram
/// * `kernel_size` - Size of the median filter kernel (default: 31).
///   Larger = smoother separation but may smear short notes.
///   Must be odd.
///
/// # Returns
/// An `HpssResult` containing the separated harmonic and percussive spectrograms.
pub fn hpss(spectrogram: &Spectrogram, kernel_size: Option<usize>) -> HpssResult {
    let kernel = kernel_size.unwrap_or(15);
    // Ensure kernel is odd for symmetric median filtering
    let kernel = if kernel % 2 == 0 { kernel + 1 } else { kernel };
    let half_k = kernel / 2;

    let n_frames = spectrogram.n_frames;
    let n_freq = spectrogram.n_freq_bins;

    // Step 1: Compute harmonic-enhanced spectrogram (median filter across TIME)
    // For each frequency bin, slide a window across time frames and take the median.
    // This preserves energy that is sustained over time (harmonics) and
    // smooths out brief spikes (transients).
    //
    // Optimization: transpose to freq-major layout so the time-axis median
    // filter reads contiguous memory. Then transpose back.
    let mut transposed = vec![vec![0.0_f32; n_frames]; n_freq];
    for frame in 0..n_frames {
        for freq in 0..n_freq {
            transposed[freq][frame] = spectrogram.magnitudes[frame][freq];
        }
    }

    let mut harmonic_enhanced = vec![vec![0.0_f32; n_freq]; n_frames];
    let mut buf = Vec::with_capacity(kernel);
    for freq in 0..n_freq {
        let row = &transposed[freq];
        for frame in 0..n_frames {
            let start = frame.saturating_sub(half_k);
            let end = (frame + half_k + 1).min(n_frames);
            buf.clear();
            buf.extend_from_slice(&row[start..end]);
            harmonic_enhanced[frame][freq] = median(&mut buf);
        }
    }

    // Step 2: Compute percussive-enhanced spectrogram (median filter across FREQUENCY)
    // For each time frame, slide a window across frequency bins and take the median.
    // This preserves energy that spans many frequencies simultaneously (transients)
    // and smooths out narrowband energy (harmonics).
    //
    // Optimization: frame is outer loop for cache locality on the frequency axis.
    let mut percussive_enhanced = vec![vec![0.0_f32; n_freq]; n_frames];
    for frame in 0..n_frames {
        let row = &spectrogram.magnitudes[frame];
        for freq in 0..n_freq {
            let start = freq.saturating_sub(half_k);
            let end = (freq + half_k + 1).min(n_freq);
            buf.clear();
            buf.extend_from_slice(&row[start..end]);
            percussive_enhanced[frame][freq] = median(&mut buf);
        }
    }

    // Step 3: Soft masking — assign each cell based on relative strength.
    // mask_h = H_enhanced^2 / (H_enhanced^2 + P_enhanced^2 + epsilon)
    // mask_p = P_enhanced^2 / (H_enhanced^2 + P_enhanced^2 + epsilon)
    // Then: harmonic = mask_h * original, percussive = mask_p * original
    //
    // Squaring gives Wiener-filter-like behaviour with sharper separation
    // than a simple ratio.
    let epsilon = 1e-10_f32;
    let mut harmonic = vec![vec![0.0_f32; n_freq]; n_frames];
    let mut percussive = vec![vec![0.0_f32; n_freq]; n_frames];

    for frame in 0..n_frames {
        for freq in 0..n_freq {
            let h = harmonic_enhanced[frame][freq];
            let p = percussive_enhanced[frame][freq];
            let h2 = h * h;
            let p2 = p * p;
            let denom = h2 + p2 + epsilon;

            let original = spectrogram.magnitudes[frame][freq];
            harmonic[frame][freq] = (h2 / denom) * original;
            percussive[frame][freq] = (p2 / denom) * original;
        }
    }

    HpssResult {
        harmonic,
        percussive,
        n_frames,
        n_freq_bins: n_freq,
    }
}

/// Extract percussive features from an HPSS result.
///
/// # Arguments
/// * `hpss` - The separated harmonic/percussive spectrograms
/// * `sample_rate` - Audio sample rate
/// * `hop_length` - STFT hop length
///
/// # Returns
/// `PercussiveFeatures` with per-frame percussive ratio, attack sharpness,
/// and onset density — all aligned to the spectrogram time axis.
pub fn percussive_features(
    hpss_result: &HpssResult,
    sample_rate: u32,
    hop_length: usize,
) -> PercussiveFeatures {
    let n_frames = hpss_result.n_frames;
    let fps = sample_rate as f32 / hop_length as f32;

    // --- Percussive energy ratio ---
    // Per frame: sum of percussive magnitudes / sum of (harmonic + percussive)
    let mut percussive_ratio = Vec::with_capacity(n_frames);
    let mut perc_energies = Vec::with_capacity(n_frames);

    for frame in 0..n_frames {
        let h_energy: f32 = hpss_result.harmonic[frame].iter().sum();
        let p_energy: f32 = hpss_result.percussive[frame].iter().sum();
        let total = h_energy + p_energy;

        percussive_ratio.push(if total > 1e-10 { p_energy / total } else { 0.0 });
        perc_energies.push(p_energy);
    }

    // --- Attack sharpness ---
    // How much percussive energy increased from the previous frame.
    // Large positive increases indicate a sharp attack/transient.
    let mut attack_sharpness = Vec::with_capacity(n_frames);
    attack_sharpness.push(0.0); // First frame has no predecessor

    for i in 1..n_frames {
        let diff = perc_energies[i] - perc_energies[i - 1];
        // Only count positive increases (attacks, not decays)
        attack_sharpness.push(if diff > 0.0 { diff } else { 0.0 });
    }

    // Normalise attack sharpness to 0..1
    let max_sharpness = attack_sharpness
        .iter()
        .cloned()
        .fold(0.0_f32, f32::max);
    if max_sharpness > 1e-10 {
        for val in &mut attack_sharpness {
            *val /= max_sharpness;
        }
    }

    // --- Onset density ---
    // Count percussive onsets in a rolling window around each frame.
    // An "onset" is a frame where attack sharpness exceeds a threshold.
    // We report density in onsets per second.

    // Threshold: mean + 0.5 * std_dev of attack sharpness
    let mean_sharp: f32 = attack_sharpness.iter().sum::<f32>() / n_frames as f32;
    let var_sharp: f32 = attack_sharpness
        .iter()
        .map(|&x| (x - mean_sharp) * (x - mean_sharp))
        .sum::<f32>()
        / n_frames as f32;
    let threshold = mean_sharp + 0.5 * var_sharp.sqrt();
    let threshold = threshold.max(0.05); // Floor to avoid detecting everything

    // Mark onset frames
    let is_onset: Vec<bool> = attack_sharpness
        .iter()
        .map(|&s| s > threshold)
        .collect();

    // Rolling window of ~2 seconds (centered)
    let window_frames = (2.0 * fps) as usize;
    let half_window = window_frames / 2;

    let mut onset_density = Vec::with_capacity(n_frames);
    for frame in 0..n_frames {
        let start = frame.saturating_sub(half_window);
        let end = (frame + half_window + 1).min(n_frames);
        let count: usize = is_onset[start..end].iter().filter(|&&x| x).count();
        let window_secs = (end - start) as f32 / fps;
        onset_density.push(count as f32 / window_secs);
    }

    PercussiveFeatures {
        percussive_ratio,
        attack_sharpness,
        onset_density,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::spectral::compute_spectrogram;
    use std::f32::consts::PI;

    fn sine_wave(freq: f32, sample_rate: u32, duration: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    fn click_track(bpm: f32, sample_rate: u32, duration: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration) as usize;
        let samples_per_beat = (60.0 / bpm * sample_rate as f32) as usize;
        let mut samples = vec![0.0_f32; n];
        let mut pos = 0;
        while pos < n {
            let click_len = (sample_rate as f32 * 0.005) as usize; // 5ms click
            for i in 0..click_len.min(n - pos) {
                let decay = (-(i as f32) / click_len as f32 * 8.0).exp();
                samples[pos + i] = decay;
            }
            pos += samples_per_beat;
        }
        samples
    }

    #[test]
    fn test_hpss_output_shape() {
        let samples = sine_wave(440.0, 44100, 1.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let result = hpss(&spec, Some(11));

        assert_eq!(result.n_frames, spec.n_frames);
        assert_eq!(result.n_freq_bins, spec.n_freq_bins);
        assert_eq!(result.harmonic.len(), spec.n_frames);
        assert_eq!(result.percussive.len(), spec.n_frames);
        assert_eq!(result.harmonic[0].len(), spec.n_freq_bins);
    }

    #[test]
    fn test_hpss_sine_mostly_harmonic() {
        // A pure sine wave should be almost entirely harmonic
        let samples = sine_wave(440.0, 44100, 1.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let result = hpss(&spec, Some(11));

        // Sum total energy in harmonic vs percussive
        let h_total: f32 = result.harmonic.iter()
            .flat_map(|f| f.iter())
            .sum();
        let p_total: f32 = result.percussive.iter()
            .flat_map(|f| f.iter())
            .sum();

        let ratio = h_total / (h_total + p_total);
        assert!(
            ratio > 0.7,
            "Sine wave should be mostly harmonic, got ratio {:.3}",
            ratio
        );
    }

    #[test]
    fn test_hpss_clicks_mostly_percussive() {
        // A click track should be mostly percussive
        let samples = click_track(120.0, 44100, 2.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let result = hpss(&spec, Some(11));

        let h_total: f32 = result.harmonic.iter()
            .flat_map(|f| f.iter())
            .sum();
        let p_total: f32 = result.percussive.iter()
            .flat_map(|f| f.iter())
            .sum();

        let ratio = p_total / (h_total + p_total);
        assert!(
            ratio > 0.5,
            "Click track should be mostly percussive, got ratio {:.3}",
            ratio
        );
    }

    #[test]
    fn test_percussive_features_shape() {
        let samples = sine_wave(440.0, 44100, 1.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let result = hpss(&spec, Some(11));
        let feats = percussive_features(&result, 44100, 512);

        assert_eq!(feats.percussive_ratio.len(), spec.n_frames);
        assert_eq!(feats.attack_sharpness.len(), spec.n_frames);
        assert_eq!(feats.onset_density.len(), spec.n_frames);
    }

    #[test]
    fn test_percussive_features_sine_low_ratio() {
        // Sine wave should have low percussive ratio
        let samples = sine_wave(440.0, 44100, 1.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let result = hpss(&spec, Some(11));
        let feats = percussive_features(&result, 44100, 512);

        let avg_ratio: f32 = feats.percussive_ratio.iter().sum::<f32>()
            / feats.percussive_ratio.len() as f32;
        assert!(
            avg_ratio < 0.5,
            "Sine wave should have low percussive ratio, got {:.3}",
            avg_ratio
        );
    }

    #[test]
    fn test_percussive_features_clicks_high_density() {
        // Click track at 120 BPM = 2 beats/sec, should have measurable onset density
        let samples = click_track(120.0, 44100, 3.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let result = hpss(&spec, Some(11));
        let feats = percussive_features(&result, 44100, 512);

        let avg_density: f32 = feats.onset_density.iter().sum::<f32>()
            / feats.onset_density.len() as f32;
        assert!(
            avg_density > 0.5,
            "Click track should have detectable onset density, got {:.3}/sec",
            avg_density
        );
    }

    #[test]
    fn test_attack_sharpness_normalised() {
        let samples = click_track(120.0, 44100, 2.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let result = hpss(&spec, Some(11));
        let feats = percussive_features(&result, 44100, 512);

        // All values should be in 0..1
        for &val in &feats.attack_sharpness {
            assert!(val >= 0.0 && val <= 1.0, "Attack sharpness out of range: {}", val);
        }
    }

    #[test]
    fn test_median() {
        let mut odd = vec![3.0, 1.0, 2.0];
        assert_eq!(median(&mut odd), 2.0);

        let mut even = vec![4.0, 1.0, 3.0, 2.0];
        assert_eq!(median(&mut even), 2.5);

        let mut empty: Vec<f32> = vec![];
        assert_eq!(median(&mut empty), 0.0);
    }
}
