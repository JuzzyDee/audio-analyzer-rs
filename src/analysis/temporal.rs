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

/// Dynamic range analysis results.
#[derive(Debug)]
pub struct DynamicRange {
    /// Crest factor per frame: peak / RMS. Higher = more dynamic.
    /// A sine wave is ~1.41 (sqrt(2)). A brick-walled master might be 1.1.
    /// Heavily compressed music: < 4 dB. Dynamic classical: > 15 dB.
    pub crest_factor_db: Vec<f32>,
    /// Overall crest factor in dB (from global peak and global RMS).
    pub overall_crest_db: f32,
    /// Loudness range: difference between 95th and 5th percentile of RMS (in dB).
    /// Captures how much the volume varies across the track.
    /// < 3 dB = heavily compressed, 6-9 dB = typical pop/rock, > 12 dB = very dynamic.
    pub loudness_range_db: f32,
    /// 5th percentile RMS in dB (quiet sections).
    pub rms_5th_db: f32,
    /// 95th percentile RMS in dB (loud sections).
    pub rms_95th_db: f32,
    /// Peak amplitude (linear, 0.0 to 1.0).
    pub peak_amplitude: f32,
    /// Peak amplitude in dBFS (0 dBFS = digital maximum).
    pub peak_dbfs: f32,
}

/// Convert a linear amplitude to dB, with a floor to avoid log(0).
fn amplitude_to_db(amp: f32) -> f32 {
    20.0 * amp.max(1e-10).log10()
}

/// Compute dynamic range analysis from raw audio samples.
///
/// Uses the same windowing as RMS/ZCR to stay on the shared time axis.
/// Returns crest factor per frame, overall crest, and loudness range.
pub fn dynamic_range(samples: &[f32], frame_size: usize, hop_length: usize) -> DynamicRange {
    let rms = rms_energy(samples, frame_size, hop_length);

    // Peak amplitude per frame
    let n_frames = rms.len();
    let mut crest_factor_db = Vec::with_capacity(n_frames);

    for frame in 0..n_frames {
        let start = frame * hop_length;
        let end = start + frame_size;
        let peak: f32 = samples[start..end]
            .iter()
            .map(|&s| s.abs())
            .fold(0.0_f32, f32::max);

        let rms_val = rms[frame];
        if rms_val > 1e-10 {
            crest_factor_db.push(amplitude_to_db(peak / rms_val));
        } else {
            crest_factor_db.push(0.0); // silence
        }
    }

    // Global peak
    let peak_amplitude = samples.iter().map(|&s| s.abs()).fold(0.0_f32, f32::max);
    let peak_dbfs = amplitude_to_db(peak_amplitude);

    // Global RMS
    let global_rms = if !rms.is_empty() {
        let sum_sq: f32 = rms.iter().map(|&r| r * r).sum();
        (sum_sq / rms.len() as f32).sqrt()
    } else {
        0.0
    };
    let overall_crest_db = if global_rms > 1e-10 {
        amplitude_to_db(peak_amplitude / global_rms)
    } else {
        0.0
    };

    // Loudness range: 95th - 5th percentile of RMS in dB
    // Filter out silence (< -60 dB) to avoid skewing the range
    let mut rms_db: Vec<f32> = rms.iter()
        .filter(|&&r| r > 1e-10)
        .map(|&r| amplitude_to_db(r))
        .collect();
    rms_db.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let (rms_5th_db, rms_95th_db, loudness_range_db) = if rms_db.len() >= 2 {
        let idx_5 = (rms_db.len() as f32 * 0.05) as usize;
        let idx_95 = ((rms_db.len() as f32 * 0.95) as usize).min(rms_db.len() - 1);
        (rms_db[idx_5], rms_db[idx_95], rms_db[idx_95] - rms_db[idx_5])
    } else {
        (0.0, 0.0, 0.0)
    };

    DynamicRange {
        crest_factor_db,
        overall_crest_db,
        loudness_range_db,
        rms_5th_db,
        rms_95th_db,
        peak_amplitude,
        peak_dbfs,
    }
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

    #[test]
    fn test_dynamic_range_sine() {
        // Pure sine: crest factor should be ~3 dB (sqrt(2) = 1.414, 20*log10(1.414) ≈ 3.01)
        let samples = sine_wave(440.0, 44100, 1.0);
        let dr = dynamic_range(&samples, 2048, 512);

        assert!(!dr.crest_factor_db.is_empty());
        let avg_crest: f32 = dr.crest_factor_db.iter().sum::<f32>() / dr.crest_factor_db.len() as f32;
        assert!(
            (avg_crest - 3.01).abs() < 0.5,
            "Sine crest factor should be ~3.01 dB, got {:.2} dB",
            avg_crest
        );
        assert!((dr.overall_crest_db - 3.01).abs() < 0.5);
    }

    #[test]
    fn test_dynamic_range_constant_loudness() {
        // A continuous sine has minimal loudness variation
        let samples = sine_wave(440.0, 44100, 2.0);
        let dr = dynamic_range(&samples, 2048, 512);

        // Loudness range should be very small (< 1 dB) for a steady tone
        assert!(
            dr.loudness_range_db < 1.0,
            "Steady sine loudness range should be < 1 dB, got {:.2} dB",
            dr.loudness_range_db
        );
    }

    #[test]
    fn test_dynamic_range_varying_loudness() {
        // Build a signal that's quiet then loud — should have meaningful DR
        let sr = 44100;
        let quiet: Vec<f32> = (0..sr)
            .map(|i| 0.1 * (2.0 * PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let loud: Vec<f32> = (0..sr)
            .map(|i| 1.0 * (2.0 * PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let mut samples = quiet;
        samples.extend(loud);

        let dr = dynamic_range(&samples, 2048, 512);

        // Should show ~20 dB range (amplitude ratio 10:1 = 20 dB)
        assert!(
            dr.loudness_range_db > 10.0,
            "Quiet→loud signal should have > 10 dB range, got {:.2} dB",
            dr.loudness_range_db
        );
    }

    #[test]
    fn test_peak_dbfs() {
        // Full-scale sine (amplitude 1.0) should have peak near 0 dBFS
        let samples = sine_wave(440.0, 44100, 0.5);
        let dr = dynamic_range(&samples, 2048, 512);
        assert!(
            (dr.peak_dbfs - 0.0).abs() < 0.5,
            "Full-scale sine peak should be near 0 dBFS, got {:.2}",
            dr.peak_dbfs
        );
    }
}
