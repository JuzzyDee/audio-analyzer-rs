// analysis/rhythm.rs — Rhythm and tempo analysis
//
// This module handles beat tracking, tempo estimation, and onset detection.
// These are some of the trickiest problems in music information retrieval
// because rhythm is perceptual — humans infer beats from patterns that
// aren't always explicitly marked in the audio.
//
// The approach:
// 1. Compute an "onset strength envelope" — a signal that peaks whenever
//    something new happens in the audio (a note attack, a drum hit, etc.)
// 2. Find the tempo by looking for periodicity in the onset envelope
//    (autocorrelation — "how much does this signal repeat at interval X?")
// 3. Track individual beats by finding onset peaks that align with the tempo

use super::spectral::Spectrogram;

/// Result of rhythm analysis.
#[derive(Debug)]
pub struct RhythmAnalysis {
    /// Estimated tempo in BPM
    pub tempo_bpm: f32,

    /// Confidence of the tempo estimate (0.0 to 1.0)
    pub tempo_confidence: f32,

    /// Onset strength envelope — one value per spectrogram frame.
    /// Peaks indicate where new sounds begin.
    pub onset_envelope: Vec<f32>,

    /// Beat times in seconds — estimated positions of each beat
    pub beat_times: Vec<f32>,

    /// Sample rate (inherited)
    pub sample_rate: u32,

    /// Hop length (inherited)
    pub hop_length: usize,
}

/// Compute the onset strength envelope from a spectrogram.
///
/// The onset envelope measures how much the spectrum changes from one
/// frame to the next. A big change = something new happened (a note
/// started, a drum was hit). We use "spectral flux" — the sum of
/// positive differences between consecutive frames.
///
/// This is the foundation for both tempo estimation and beat tracking.
///
/// Python equivalent: librosa.onset.onset_strength()
fn onset_strength(spectrogram: &Spectrogram) -> Vec<f32> {
    if spectrogram.n_frames < 2 {
        return vec![0.0; spectrogram.n_frames];
    }

    let mut envelope = Vec::with_capacity(spectrogram.n_frames);
    envelope.push(0.0); // First frame has no predecessor to compare against

    for i in 1..spectrogram.n_frames {
        // Spectral flux: sum of positive magnitude increases.
        // We only count *increases* (half-wave rectification) because
        // we care about new energy appearing, not energy disappearing.
        let flux: f32 = spectrogram.magnitudes[i]
            .iter()
            .zip(spectrogram.magnitudes[i - 1].iter())
            .map(|(&current, &previous)| {
                let diff = current - previous;
                if diff > 0.0 { diff } else { 0.0 }
                // This `if` is an expression in Rust — it returns a value.
                // No ternary operator needed; `if/else` IS the ternary.
            })
            .sum();

        envelope.push(flux);
    }

    // Normalise the envelope to 0.0..1.0
    let max_val = envelope
        .iter()
        .cloned()
        .fold(0.0_f32, f32::max);

    if max_val > 1e-10 {
        for val in &mut envelope {
            *val /= max_val;
        }
    }

    envelope
}

/// Estimate tempo using autocorrelation of the onset envelope.
///
/// Autocorrelation asks: "if I shift this signal by N frames, how well
/// does it line up with itself?" For a rhythmic signal, it will line up
/// best when N corresponds to the beat period. We convert that period
/// from frames to BPM.
///
/// We search within a reasonable BPM range (typically 50-220 BPM) to
/// avoid octave errors (detecting half or double the tempo).
///
/// Python equivalent: librosa.beat.beat_track() (tempo estimation part)
fn estimate_tempo(
    onset_env: &[f32],
    sample_rate: u32,
    hop_length: usize,
    min_bpm: f32,
    max_bpm: f32,
) -> (f32, f32) {
    // Convert BPM range to lag range in frames.
    // BPM = 60 * sample_rate / (lag_frames * hop_length)
    // So: lag_frames = 60 * sample_rate / (BPM * hop_length)
    let sr_hop = sample_rate as f32 / hop_length as f32;

    // Note: min_bpm → max_lag, max_bpm → min_lag (inverse relationship)
    let min_lag = (60.0 * sr_hop / max_bpm) as usize;
    let max_lag = (60.0 * sr_hop / min_bpm) as usize;

    // Clamp to available data
    // `.min()` on usize — Rust's primitive types have comparison methods built in.
    let max_lag = max_lag.min(onset_env.len() / 2);

    if min_lag >= max_lag {
        return (120.0, 0.0); // Fallback if not enough data
    }

    // Compute autocorrelation for each lag in our BPM range.
    // This is O(n * lags) — not the fastest approach but clear and correct.
    // For a production system you'd use FFT-based autocorrelation.
    let mut best_lag = min_lag;
    let mut best_corr = f32::NEG_INFINITY;
    let mut correlations = Vec::with_capacity(max_lag - min_lag);

    for lag in min_lag..max_lag {
        // Autocorrelation at this lag: dot product of signal with shifted self
        let corr: f32 = onset_env
            .iter()
            .zip(onset_env[lag..].iter())
            .map(|(&a, &b)| a * b)
            .sum();

        correlations.push(corr);

        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    // Convert best lag to BPM
    let tempo = 60.0 * sr_hop / best_lag as f32;

    // Compute confidence as the ratio of the best peak to the mean correlation.
    // Higher ratio = clearer rhythmic periodicity = more confident estimate.
    let mean_corr: f32 = if correlations.is_empty() {
        1.0
    } else {
        correlations.iter().sum::<f32>() / correlations.len() as f32
    };

    let confidence = if mean_corr > 1e-10 {
        // Normalise to roughly 0..1 range using a sigmoid-like scaling
        let ratio = best_corr / mean_corr;
        // A ratio of ~2 is moderate confidence, ~5+ is very strong
        1.0 - 1.0 / (1.0 + (ratio - 1.0) * 0.5)
    } else {
        0.0
    };

    (tempo, confidence.clamp(0.0, 1.0))
    // `.clamp(min, max)` restricts a value to a range — cleaner than
    // chaining .min().max(). Added in Rust 1.50.
}

/// Simple peak picking on the onset envelope.
///
/// Finds local maxima that are above a threshold and spaced at least
/// `min_spacing` frames apart. Returns frame indices of detected peaks.
fn pick_peaks(envelope: &[f32], threshold: f32, min_spacing: usize) -> Vec<usize> {
    let mut peaks = Vec::new();

    // Start from index 1, end before the last — peaks need neighbours
    for i in 1..envelope.len().saturating_sub(1) {
        // `.saturating_sub(1)` subtracts 1 but floors at 0 instead of
        // panicking on underflow. For `usize` (unsigned), 0 - 1 would
        // panic in debug or wrap in release. Saturating arithmetic is
        // the safe way to handle this.

        // Is this a local maximum above threshold?
        if envelope[i] > threshold
            && envelope[i] >= envelope[i - 1]
            && envelope[i] >= envelope[i + 1]
        {
            // Is it far enough from the last detected peak?
            let far_enough = peaks
                .last()
                .map_or(true, |&last: &usize| i - last >= min_spacing);
            // `.last()` returns Option<&T> — the last element if it exists.
            // `.map_or(default, f)` applies f if Some, returns default if None.
            // So: if no previous peak, always accept; otherwise check spacing.

            if far_enough {
                peaks.push(i);
            }
        }
    }

    peaks
}

/// Perform full rhythm analysis on a spectrogram.
///
/// Computes onset envelope, estimates tempo, and tracks individual beats.
///
/// # Arguments
/// * `spectrogram` - The STFT spectrogram
/// * `min_bpm` - Minimum tempo to consider (default: 50)
/// * `max_bpm` - Maximum tempo to consider (default: 220)
pub fn analyse_rhythm(
    spectrogram: &Spectrogram,
    min_bpm: Option<f32>,
    max_bpm: Option<f32>,
) -> RhythmAnalysis {
    let min_bpm = min_bpm.unwrap_or(50.0);
    let max_bpm = max_bpm.unwrap_or(220.0);

    // Step 1: Compute onset strength envelope
    let onset_env = onset_strength(spectrogram);

    // Step 2: Estimate tempo from the envelope's periodicity
    let (tempo_bpm, tempo_confidence) = estimate_tempo(
        &onset_env,
        spectrogram.sample_rate,
        spectrogram.hop_length,
        min_bpm,
        max_bpm,
    );

    // Step 3: Track individual beats
    // Use the estimated tempo to set the minimum spacing between beats.
    // Expected frames between beats = 60 / tempo * sample_rate / hop_length
    let sr_hop = spectrogram.sample_rate as f32 / spectrogram.hop_length as f32;
    let expected_beat_frames = (60.0 * sr_hop / tempo_bpm) as usize;

    // Allow beats to be a bit closer than the expected spacing (80%)
    // to handle slight tempo variations and syncopation.
    let min_spacing = (expected_beat_frames as f32 * 0.8) as usize;

    // Adaptive threshold: use the mean + 0.5 * std deviation of the onset envelope.
    // This adapts to the dynamic range of the specific track.
    let mean: f32 = onset_env.iter().sum::<f32>() / onset_env.len() as f32;
    let variance: f32 = onset_env
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f32>()
        / onset_env.len() as f32;
    let std_dev = variance.sqrt();
    let threshold = (mean + 0.5 * std_dev).max(0.05);
    // `.max(0.05)` ensures a minimum threshold — for very quiet passages
    // we don't want to detect everything as a beat.

    let beat_frames = pick_peaks(&onset_env, threshold, min_spacing);

    // Convert beat frame indices to times in seconds
    let beat_times: Vec<f32> = beat_frames
        .iter()
        .map(|&frame| frame as f32 * spectrogram.hop_length as f32 / spectrogram.sample_rate as f32)
        .collect();

    RhythmAnalysis {
        tempo_bpm,
        tempo_confidence,
        onset_envelope: onset_env,
        beat_times,
        sample_rate: spectrogram.sample_rate,
        hop_length: spectrogram.hop_length,
    }
}

/// Compute inter-beat intervals and their statistics.
///
/// Useful for understanding tempo stability — a metronome-locked track
/// will have very consistent IBIs, while a free-tempo performance
/// will show more variation.
pub fn beat_statistics(beat_times: &[f32]) -> Option<BeatStats> {
    // Need at least 2 beats to compute intervals
    if beat_times.len() < 2 {
        return None;
        // Returning `None` from an `Option`-returning function.
        // This is Rust's way of saying "there's no meaningful result."
    }

    // Compute inter-beat intervals (time between consecutive beats)
    // `.windows(2)` creates a sliding window of size 2 over the slice:
    // [a, b, c, d] → [a,b], [b,c], [c,d]
    // This is incredibly useful and something Python doesn't have built-in
    // (you'd use zip(list, list[1:]) which allocates a new list).
    let ibis: Vec<f32> = beat_times
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect();

    let n = ibis.len() as f32;
    let mean_ibi = ibis.iter().sum::<f32>() / n;
    let median_ibi = {
        let mut sorted = ibis.clone();
        // Sorting f32 requires special handling because NaN breaks ordering.
        // `partial_cmp` returns Option<Ordering>; `.unwrap()` is safe here
        // because we know our data has no NaN values.
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };

    let variance: f32 = ibis.iter().map(|&x| (x - mean_ibi).powi(2)).sum::<f32>() / n;
    // `.powi(2)` is integer power — more efficient than `.powf(2.0)`.
    let std_dev = variance.sqrt();

    // Tempo stability: coefficient of variation (lower = more stable)
    let stability = if mean_ibi > 1e-10 {
        1.0 - (std_dev / mean_ibi).min(1.0)
    } else {
        0.0
    };

    Some(BeatStats {
        beat_count: beat_times.len(),
        mean_ibi,
        median_ibi,
        std_dev_ibi: std_dev,
        tempo_stability: stability,
        mean_bpm: 60.0 / mean_ibi,
        median_bpm: 60.0 / median_ibi,
    })
}

/// Statistics about detected beats.
#[derive(Debug)]
pub struct BeatStats {
    /// Total number of detected beats
    pub beat_count: usize,
    /// Mean inter-beat interval in seconds
    pub mean_ibi: f32,
    /// Median inter-beat interval in seconds
    pub median_ibi: f32,
    /// Standard deviation of inter-beat intervals
    pub std_dev_ibi: f32,
    /// Tempo stability (0.0 = erratic, 1.0 = metronomic)
    pub tempo_stability: f32,
    /// Tempo derived from mean IBI
    pub mean_bpm: f32,
    /// Tempo derived from median IBI
    pub median_bpm: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::spectral::compute_spectrogram;
    use std::f32::consts::PI;

    /// Generate a click track — periodic impulses at a given BPM.
    /// This is the ideal case for beat tracking: perfectly regular,
    /// clear transients, no harmonic content to confuse things.
    fn click_track(bpm: f32, sample_rate: u32, duration: f32) -> Vec<f32> {
        let n_samples = (sample_rate as f32 * duration) as usize;
        let samples_per_beat = (60.0 / bpm * sample_rate as f32) as usize;

        let mut samples = vec![0.0_f32; n_samples];
        let mut pos = 0;

        while pos < n_samples {
            // Each click is a short burst of noise (10ms)
            let click_len = (sample_rate as f32 * 0.01) as usize;
            for i in 0..click_len.min(n_samples - pos) {
                // Exponentially decaying click
                let decay = (-(i as f32) / click_len as f32 * 5.0).exp();
                samples[pos + i] = decay;
            }
            pos += samples_per_beat;
        }

        samples
    }

    #[test]
    fn test_onset_strength_shape() {
        let samples: Vec<f32> = (0..44100)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let env = onset_strength(&spec);
        assert_eq!(env.len(), spec.n_frames);
    }

    #[test]
    fn test_tempo_estimation_120bpm() {
        // A click track at 120 BPM should be detected near 120 BPM
        let clicks = click_track(120.0, 44100, 4.0);
        let spec = compute_spectrogram(&clicks, 44100, None, None);
        let rhythm = analyse_rhythm(&spec, None, None);

        println!("Estimated tempo: {:.1} BPM (confidence: {:.3})", rhythm.tempo_bpm, rhythm.tempo_confidence);

        assert!(
            (rhythm.tempo_bpm - 120.0).abs() < 5.0,
            "Expected ~120 BPM, got {:.1}",
            rhythm.tempo_bpm
        );
    }

    #[test]
    fn test_tempo_estimation_92bpm() {
        // Testing at 92 BPM since that's what Warm Starry Nights should be
        let clicks = click_track(92.0, 44100, 4.0);
        let spec = compute_spectrogram(&clicks, 44100, None, None);
        let rhythm = analyse_rhythm(&spec, None, None);

        println!("Estimated tempo: {:.1} BPM (confidence: {:.3})", rhythm.tempo_bpm, rhythm.tempo_confidence);

        assert!(
            (rhythm.tempo_bpm - 92.0).abs() < 5.0,
            "Expected ~92 BPM, got {:.1}",
            rhythm.tempo_bpm
        );
    }

    #[test]
    fn test_beat_statistics() {
        // Regular beats every 0.5 seconds = 120 BPM
        let beat_times = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let stats = beat_statistics(&beat_times).unwrap();

        assert_eq!(stats.beat_count, 7);
        assert!((stats.mean_ibi - 0.5).abs() < 0.01);
        assert!((stats.mean_bpm - 120.0).abs() < 1.0);
        assert!(stats.tempo_stability > 0.95, "Perfect tempo should have high stability");
    }

    #[test]
    fn test_beat_statistics_needs_two_beats() {
        // With 0 or 1 beats, there's no meaningful statistics
        assert!(beat_statistics(&[]).is_none());
        assert!(beat_statistics(&[1.0]).is_none());
        assert!(beat_statistics(&[1.0, 2.0]).is_some());
    }
}
