// analysis/temporal.rs — Time-domain audio features
//
// These features operate directly on the audio waveform (not the spectrogram).
// They're cheap to compute and capture aspects of the signal that frequency-domain
// analysis misses: loudness (RMS) and texture (zero crossing rate).

/// Compute RMS (Root Mean Square) energy for windowed frames.
///
/// RMS measures the loudness/power of the signal over time. It's the standard
/// way to track volume changes — much more meaningful than peak amplitude
/// because it reflects perceived loudness.
///
/// Returns one value per frame, matching the spectrogram's time axis when
/// using the same n_fft and hop_length.
///
/// Python equivalent: librosa.feature.rms()
pub fn rms_energy(samples: &[f32], frame_size: usize, hop_length: usize) -> Vec<f32> {
    if samples.len() < frame_size {
        return Vec::new();
    }

    let n_frames = (samples.len() - frame_size) / hop_length + 1;
    let mut result = Vec::with_capacity(n_frames);

    for frame in 0..n_frames {
        let start = frame * hop_length;
        let end = start + frame_size;

        let sum_sq: f32 = samples[start..end]
            .iter()
            .map(|&s| s * s)
            .sum();

        result.push((sum_sq / frame_size as f32).sqrt());
    }

    result
}

/// Compute the zero crossing rate for windowed frames.
///
/// ZCR counts how often the signal crosses zero within each frame.
/// High ZCR = noisy or percussive (cymbals, consonants).
/// Low ZCR = tonal/pitched (sustained notes, vowels).
///
/// Combined with spectral flatness, this gives a robust picture of
/// whether something is tonal, percussive, or noisy at any moment.
///
/// Returns one value per frame (as a rate: crossings / frame_size).
///
/// Python equivalent: librosa.feature.zero_crossing_rate()
pub fn zero_crossing_rate(samples: &[f32], frame_size: usize, hop_length: usize) -> Vec<f32> {
    if samples.len() < frame_size {
        return Vec::new();
    }

    let n_frames = (samples.len() - frame_size) / hop_length + 1;
    let mut result = Vec::with_capacity(n_frames);

    for frame in 0..n_frames {
        let start = frame * hop_length;
        let window = &samples[start..start + frame_size];

        let crossings: usize = window
            .windows(2)
            .filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
            .count();

        // Normalise to a rate (0.0 to 1.0 range)
        result.push(crossings as f32 / (frame_size - 1) as f32);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq: f32, sample_rate: u32, duration: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_rms_sine_wave() {
        // RMS of a sine wave with amplitude 1.0 should be ~0.707 (1/sqrt(2))
        let samples = sine_wave(440.0, 44100, 1.0);
        let rms = rms_energy(&samples, 2048, 512);

        assert!(!rms.is_empty());
        let avg_rms: f32 = rms.iter().sum::<f32>() / rms.len() as f32;
        assert!(
            (avg_rms - 0.707).abs() < 0.02,
            "Sine RMS should be ~0.707, got {:.4}",
            avg_rms
        );
    }

    #[test]
    fn test_rms_silence() {
        let samples = vec![0.0_f32; 44100];
        let rms = rms_energy(&samples, 2048, 512);
        assert!(!rms.is_empty());
        assert!(rms.iter().all(|&v| v < 1e-10));
    }

    #[test]
    fn test_rms_frame_count() {
        // Frame count should match spectrogram convention
        let samples = sine_wave(440.0, 44100, 1.0);
        let rms = rms_energy(&samples, 2048, 512);
        let expected = (44100 - 2048) / 512 + 1;
        assert_eq!(rms.len(), expected);
    }

    #[test]
    fn test_zcr_sine_wave() {
        // A 440 Hz sine at 44100 Hz has ~440 zero crossings per second
        // (actually 880 since it crosses twice per cycle)
        // In a 2048-sample frame: 880 * (2048/44100) ≈ 40.9 crossings
        // Rate: 40.9 / 2047 ≈ 0.020
        let samples = sine_wave(440.0, 44100, 1.0);
        let zcr = zero_crossing_rate(&samples, 2048, 512);

        assert!(!zcr.is_empty());
        let avg_zcr: f32 = zcr.iter().sum::<f32>() / zcr.len() as f32;
        assert!(
            avg_zcr > 0.01 && avg_zcr < 0.05,
            "440 Hz sine ZCR should be ~0.02, got {:.4}",
            avg_zcr
        );
    }

    #[test]
    fn test_zcr_high_freq_higher() {
        // Higher frequency = more zero crossings
        let low = sine_wave(100.0, 44100, 0.5);
        let high = sine_wave(4000.0, 44100, 0.5);

        let zcr_low = zero_crossing_rate(&low, 2048, 512);
        let zcr_high = zero_crossing_rate(&high, 2048, 512);

        let avg_low: f32 = zcr_low.iter().sum::<f32>() / zcr_low.len() as f32;
        let avg_high: f32 = zcr_high.iter().sum::<f32>() / zcr_high.len() as f32;

        assert!(
            avg_high > avg_low * 5.0,
            "4000 Hz ZCR ({:.4}) should be much higher than 100 Hz ({:.4})",
            avg_high, avg_low
        );
    }

    #[test]
    fn test_zcr_frame_count() {
        let samples = sine_wave(440.0, 44100, 1.0);
        let zcr = zero_crossing_rate(&samples, 2048, 512);
        let expected = (44100 - 2048) / 512 + 1;
        assert_eq!(zcr.len(), expected);
    }

    #[test]
    fn test_short_input() {
        // Input shorter than frame_size should return empty
        let samples = vec![0.0; 100];
        assert!(rms_energy(&samples, 2048, 512).is_empty());
        assert!(zero_crossing_rate(&samples, 2048, 512).is_empty());
    }
}
