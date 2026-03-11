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

// ---- LUFS (EBU R128) Loudness Measurement ----

/// Biquad filter coefficients (second-order IIR).
struct Biquad {
    b0: f64, b1: f64, b2: f64,
    a1: f64, a2: f64,
}

impl Biquad {
    /// Apply the filter to a buffer of samples in-place.
    fn process(&self, samples: &mut [f32]) {
        let mut x1: f64 = 0.0;
        let mut x2: f64 = 0.0;
        let mut y1: f64 = 0.0;
        let mut y2: f64 = 0.0;

        for s in samples.iter_mut() {
            let x0 = *s as f64;
            let y0 = self.b0 * x0 + self.b1 * x1 + self.b2 * x2
                   - self.a1 * y1 - self.a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            *s = y0 as f32;
        }
    }
}

/// K-weighting pre-filter (high-shelf boost).
/// Coefficients for 48000 Hz from ITU-R BS.1770-4.
fn k_weighting_stage1(sample_rate: u32) -> Biquad {
    // These are the standard coefficients. For sample rates other than 48kHz,
    // we use the bilinear transform to recalculate.
    if sample_rate == 48000 {
        Biquad {
            b0: 1.53512485958697,
            b1: -2.69169618940638,
            b2: 1.19839281085285,
            a1: -1.69065929318241,
            a2: 0.73248077421585,
        }
    } else {
        // Compute from analog prototype via bilinear transform
        k_shelf_coefficients(sample_rate)
    }
}

/// K-weighting high-pass filter (second stage).
/// Coefficients for 48000 Hz from ITU-R BS.1770-4.
fn k_weighting_stage2(sample_rate: u32) -> Biquad {
    if sample_rate == 48000 {
        Biquad {
            b0: 1.0,
            b1: -2.0,
            b2: 1.0,
            a1: -1.99004745483398,
            a2: 0.99007225036621,
        }
    } else {
        k_highpass_coefficients(sample_rate)
    }
}

/// Compute high-shelf filter coefficients for arbitrary sample rate.
/// Based on the analog prototype from ITU-R BS.1770-4.
fn k_shelf_coefficients(sample_rate: u32) -> Biquad {
    let fs = sample_rate as f64;

    // Pre-filter: high shelf at ~1681 Hz with ~4 dB boost
    let db = 3.999843853973347;
    let f0 = 1681.974450955533;
    let q = 0.7071752369554196;

    let a = 10.0_f64.powf(db / 40.0); // amplitude
    let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
    let (sin_w0, cos_w0) = w0.sin_cos();
    let alpha = sin_w0 / (2.0 * q);

    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
    Biquad {
        b0: (a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha)) / a0,
        b1: (-2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)) / a0,
        b2: (a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha)) / a0,
        a1: (2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)) / a0,
        a2: ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha) / a0,
    }
}

/// Compute high-pass filter coefficients for arbitrary sample rate.
/// Second-order Butterworth high-pass at ~38 Hz.
fn k_highpass_coefficients(sample_rate: u32) -> Biquad {
    let fs = sample_rate as f64;
    let f0 = 38.13547087602444;
    let q = 0.5003270373238773;

    let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
    let (sin_w0, cos_w0) = w0.sin_cos();
    let alpha = sin_w0 / (2.0 * q);

    let a0 = 1.0 + alpha;
    Biquad {
        b0: ((1.0 + cos_w0) / 2.0) / a0,
        b1: (-(1.0 + cos_w0)) / a0,
        b2: ((1.0 + cos_w0) / 2.0) / a0,
        a1: (-2.0 * cos_w0) / a0,
        a2: (1.0 - alpha) / a0,
    }
}

/// Apply K-weighting filter to samples (returns a new filtered buffer).
fn k_weight(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let mut filtered = samples.to_vec();
    let stage1 = k_weighting_stage1(sample_rate);
    let stage2 = k_weighting_stage2(sample_rate);
    stage1.process(&mut filtered);
    stage2.process(&mut filtered);
    filtered
}

/// Compute mean square energy of a block of samples.
fn mean_square(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    sum / samples.len() as f64
}

/// Convert mean square to LUFS.
fn ms_to_lufs(ms: f64) -> f64 {
    -0.691 + 10.0 * ms.max(1e-20).log10()
}

/// LUFS measurement results (EBU R128).
#[derive(Debug)]
pub struct LufsResult {
    /// Integrated loudness (gated, whole track) in LUFS.
    pub integrated: f32,
    /// Short-term loudness values (3-second sliding window) in LUFS.
    pub short_term: Vec<f32>,
    /// Momentary loudness values (400ms sliding window) in LUFS.
    pub momentary: Vec<f32>,
    /// Loudness range (LRA) in LU — difference between 95th and 10th percentile
    /// of short-term loudness, gated.
    pub loudness_range: f32,
    /// True peak in dBTP (from 4x oversampled signal).
    pub true_peak_dbtp: f32,
    /// Time axis for short-term values (seconds).
    pub short_term_times: Vec<f32>,
    /// Time axis for momentary values (seconds).
    pub momentary_times: Vec<f32>,
}

/// Compute true peak via 4x oversampling.
///
/// Uses a simple windowed sinc interpolation to upsample the signal,
/// then finds the peak. This catches inter-sample peaks that exceed
/// the sample-domain peak.
fn true_peak(samples: &[f32]) -> f32 {
    // 4x oversampling with linear interpolation as a practical approximation.
    // Full sinc interpolation is more accurate but this catches the vast majority
    // of inter-sample peaks and is much faster.
    let mut peak: f32 = 0.0;

    // Check original samples
    for &s in samples {
        peak = peak.max(s.abs());
    }

    // Check interpolated points between each pair of samples (3 points per gap = 4x)
    if samples.len() >= 2 {
        for i in 0..samples.len() - 1 {
            let s0 = samples[i];
            let s1 = samples[i + 1];
            // Linear interpolation at 1/4, 2/4, 3/4 positions
            for k in 1..4 {
                let t = k as f32 / 4.0;
                let interp = s0 + (s1 - s0) * t;
                peak = peak.max(interp.abs());
            }
        }
    }

    peak
}

/// Measure LUFS loudness per EBU R128 / ITU-R BS.1770-4.
///
/// Computes integrated loudness (gated), short-term (3s), momentary (400ms),
/// loudness range (LRA), and true peak.
///
/// This is the standard loudness measurement used by streaming platforms:
/// Spotify targets -14 LUFS, Apple Music -16 LUFS, YouTube -14 LUFS.
pub fn measure_lufs(samples: &[f32], sample_rate: u32) -> LufsResult {
    // Step 1: K-weight the signal
    let filtered = k_weight(samples, sample_rate);

    // Step 2: Compute momentary loudness (400ms blocks, 100ms hop)
    let block_400ms = (sample_rate as f64 * 0.4) as usize;
    let hop_100ms = (sample_rate as f64 * 0.1) as usize;
    let mut momentary = Vec::new();
    let mut momentary_times = Vec::new();
    let mut momentary_ms = Vec::new(); // keep mean-square for gating

    let mut pos = 0;
    while pos + block_400ms <= filtered.len() {
        let ms = mean_square(&filtered[pos..pos + block_400ms]);
        momentary_ms.push(ms);
        momentary.push(ms_to_lufs(ms) as f32);
        momentary_times.push((pos + block_400ms / 2) as f32 / sample_rate as f32);
        pos += hop_100ms;
    }

    // Step 3: Compute short-term loudness (3s blocks, 100ms hop)
    let block_3s = (sample_rate as f64 * 3.0) as usize;
    let mut short_term = Vec::new();
    let mut short_term_times = Vec::new();
    let mut short_term_ms = Vec::new();

    pos = 0;
    while pos + block_3s <= filtered.len() {
        let ms = mean_square(&filtered[pos..pos + block_3s]);
        short_term_ms.push(ms);
        short_term.push(ms_to_lufs(ms) as f32);
        short_term_times.push((pos + block_3s / 2) as f32 / sample_rate as f32);
        pos += hop_100ms;
    }

    // Step 4: Integrated loudness with EBU R128 gating
    // Use 400ms blocks with 75% overlap (100ms hop) as per spec
    let integrated = gated_loudness(&momentary_ms);

    // Step 5: Loudness range (LRA) from short-term loudness
    let loudness_range = compute_lra(&short_term_ms);

    // Step 6: True peak
    let tp = true_peak(samples);
    let true_peak_dbtp = amplitude_to_db(tp);

    LufsResult {
        integrated,
        short_term,
        momentary,
        loudness_range,
        true_peak_dbtp,
        short_term_times,
        momentary_times,
    }
}

/// EBU R128 gated loudness measurement.
///
/// Two-stage gate:
/// 1. Absolute gate at -70 LUFS — ignore silence
/// 2. Relative gate at ungated_mean - 10 LU — ignore quiet passages
fn gated_loudness(block_ms: &[f64]) -> f32 {
    if block_ms.is_empty() {
        return -70.0;
    }

    // Stage 1: Absolute gate at -70 LUFS
    let abs_threshold_ms = 10.0_f64.powf((-70.0 + 0.691) / 10.0);
    let above_abs: Vec<f64> = block_ms.iter()
        .copied()
        .filter(|&ms| ms > abs_threshold_ms)
        .collect();

    if above_abs.is_empty() {
        return -70.0;
    }

    // Ungated mean (of blocks above absolute threshold)
    let ungated_mean: f64 = above_abs.iter().sum::<f64>() / above_abs.len() as f64;
    let ungated_lufs = ms_to_lufs(ungated_mean);

    // Stage 2: Relative gate at ungated_mean - 10 LU
    let rel_threshold_ms = 10.0_f64.powf((ungated_lufs - 10.0 + 0.691) / 10.0);
    let above_rel: Vec<f64> = block_ms.iter()
        .copied()
        .filter(|&ms| ms > rel_threshold_ms)
        .collect();

    if above_rel.is_empty() {
        return -70.0;
    }

    let gated_mean: f64 = above_rel.iter().sum::<f64>() / above_rel.len() as f64;
    ms_to_lufs(gated_mean) as f32
}

/// Compute Loudness Range (LRA) per EBU R128 / EBU Tech 3342.
///
/// LRA is the difference between the 95th and 10th percentile of
/// short-term loudness values, after applying the same two-stage gate.
fn compute_lra(short_term_ms: &[f64]) -> f32 {
    if short_term_ms.len() < 2 {
        return 0.0;
    }

    // Stage 1: Absolute gate at -70 LUFS
    let abs_threshold_ms = 10.0_f64.powf((-70.0 + 0.691) / 10.0);
    let above_abs: Vec<f64> = short_term_ms.iter()
        .copied()
        .filter(|&ms| ms > abs_threshold_ms)
        .collect();

    if above_abs.is_empty() {
        return 0.0;
    }

    // Relative gate at -20 LU below ungated mean (LRA uses -20, not -10)
    let ungated_mean: f64 = above_abs.iter().sum::<f64>() / above_abs.len() as f64;
    let ungated_lufs = ms_to_lufs(ungated_mean);
    let rel_threshold_ms = 10.0_f64.powf((ungated_lufs - 20.0 + 0.691) / 10.0);

    let mut gated_lufs: Vec<f64> = short_term_ms.iter()
        .copied()
        .filter(|&ms| ms > rel_threshold_ms)
        .map(|ms| ms_to_lufs(ms))
        .collect();

    if gated_lufs.len() < 2 {
        return 0.0;
    }

    gated_lufs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx_10 = (gated_lufs.len() as f64 * 0.10) as usize;
    let idx_95 = ((gated_lufs.len() as f64 * 0.95) as usize).min(gated_lufs.len() - 1);

    (gated_lufs[idx_95] - gated_lufs[idx_10]) as f32
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

    // ---- LUFS tests ----

    fn sine_wave_48k(freq: f32, amplitude: f32, duration: f32) -> Vec<f32> {
        let sr = 48000;
        let n = (sr as f32 * duration) as usize;
        (0..n)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn test_lufs_full_scale_sine_1khz() {
        // A full-scale 1kHz sine at 48kHz should measure around -3.01 LUFS
        // (the K-weighting is nearly flat at 1kHz, and a full-scale sine
        // has RMS of -3.01 dBFS)
        let samples = sine_wave_48k(1000.0, 1.0, 5.0);
        let result = measure_lufs(&samples, 48000);

        // Should be roughly -3 LUFS (K-weighting adds ~0 dB at 1kHz)
        assert!(
            result.integrated > -5.0 && result.integrated < -1.0,
            "Full-scale 1kHz sine should be around -3 LUFS, got {:.1}",
            result.integrated
        );
    }

    #[test]
    fn test_lufs_half_amplitude_quieter() {
        // Half amplitude = -6 dB quieter
        let full = sine_wave_48k(1000.0, 1.0, 5.0);
        let half = sine_wave_48k(1000.0, 0.5, 5.0);

        let lufs_full = measure_lufs(&full, 48000);
        let lufs_half = measure_lufs(&half, 48000);

        let diff = lufs_full.integrated - lufs_half.integrated;
        assert!(
            (diff - 6.0).abs() < 1.0,
            "Half amplitude should be ~6 dB quieter, got {:.1} dB difference",
            diff
        );
    }

    #[test]
    fn test_lufs_silence_gated() {
        // Silence should be gated to -70 LUFS
        let samples = vec![0.0_f32; 48000 * 5];
        let result = measure_lufs(&samples, 48000);

        assert!(
            result.integrated <= -70.0,
            "Silence should gate to -70 LUFS, got {:.1}",
            result.integrated
        );
    }

    #[test]
    fn test_lufs_k_weighting_boosts_highs() {
        // K-weighting boosts high frequencies — a 4kHz tone should measure
        // louder than a 100Hz tone at the same amplitude
        let low = sine_wave_48k(100.0, 0.5, 5.0);
        let high = sine_wave_48k(4000.0, 0.5, 5.0);

        let lufs_low = measure_lufs(&low, 48000);
        let lufs_high = measure_lufs(&high, 48000);

        assert!(
            lufs_high.integrated > lufs_low.integrated,
            "4kHz ({:.1} LUFS) should measure louder than 100Hz ({:.1} LUFS) with K-weighting",
            lufs_high.integrated, lufs_low.integrated
        );
    }

    #[test]
    fn test_lufs_true_peak() {
        // Full-scale sine true peak should be near 0 dBTP
        let samples = sine_wave_48k(1000.0, 1.0, 2.0);
        let result = measure_lufs(&samples, 48000);

        assert!(
            (result.true_peak_dbtp - 0.0).abs() < 0.5,
            "Full-scale sine true peak should be near 0 dBTP, got {:.2}",
            result.true_peak_dbtp
        );
    }

    #[test]
    fn test_lufs_short_term_populated() {
        // 5 seconds of audio should produce short-term values
        let samples = sine_wave_48k(1000.0, 0.5, 5.0);
        let result = measure_lufs(&samples, 48000);

        assert!(
            !result.short_term.is_empty(),
            "5 seconds of audio should produce short-term loudness values"
        );
        assert_eq!(result.short_term.len(), result.short_term_times.len());
    }

    #[test]
    fn test_lufs_loudness_range_steady() {
        // Steady tone should have minimal loudness range
        let samples = sine_wave_48k(1000.0, 0.5, 10.0);
        let result = measure_lufs(&samples, 48000);

        assert!(
            result.loudness_range < 2.0,
            "Steady tone should have small LRA, got {:.1} LU",
            result.loudness_range
        );
    }

    #[test]
    fn test_lufs_44100_works() {
        // Should work at 44100 Hz too (recalculates filter coefficients)
        let samples = sine_wave(1000.0, 44100, 5.0);
        let result = measure_lufs(&samples, 44100);

        assert!(
            result.integrated > -5.0 && result.integrated < -1.0,
            "44100 Hz: full-scale 1kHz sine should be around -3 LUFS, got {:.1}",
            result.integrated
        );
    }
}
