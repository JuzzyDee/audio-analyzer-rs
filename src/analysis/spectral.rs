// analysis/spectral.rs — FFT and spectral feature extraction
//
// This module implements the Short-Time Fourier Transform (STFT) and
// features derived from it. The STFT is the foundation of almost every
// audio analysis technique: chop the audio into overlapping windows,
// FFT each window, and you get a spectrogram.
//
// Python equivalent: librosa.stft(), librosa.feature.spectral_centroid(), etc.
// The difference: we compute the raw data, not rendered images.

use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

/// Result of a Short-Time Fourier Transform.
///
/// Instead of rendering this to an image, we keep the raw data so an AI
/// can request exactly the analysis it needs via MCP.
#[derive(Debug)]
pub struct Spectrogram {
    /// 2D magnitude data: magnitudes[time_frame][frequency_bin]
    /// Each inner Vec has `n_fft/2 + 1` elements (positive frequencies only).
    pub magnitudes: Vec<Vec<f32>>,

    /// Sample rate of the source audio
    pub sample_rate: u32,

    /// FFT window size used
    pub n_fft: usize,

    /// Hop length (step size between windows) in samples
    pub hop_length: usize,

    /// Number of time frames
    pub n_frames: usize,

    /// Number of frequency bins (n_fft/2 + 1)
    pub n_freq_bins: usize,
}

impl Spectrogram {
    /// Get the frequency in Hz for a given bin index.
    /// Bin 0 = 0 Hz (DC), bin n_freq_bins-1 = sample_rate/2 (Nyquist).
    pub fn bin_to_freq(&self, bin: usize) -> f32 {
        bin as f32 * self.sample_rate as f32 / self.n_fft as f32
    }

    /// Get the time in seconds for a given frame index.
    pub fn frame_to_time(&self, frame: usize) -> f32 {
        frame as f32 * self.hop_length as f32 / self.sample_rate as f32
    }

    /// Total duration covered by the spectrogram in seconds.
    pub fn duration(&self) -> f32 {
        self.frame_to_time(self.n_frames)
    }
}

/// Generate a Hann window of the given size.
///
/// A window function tapers the edges of each audio chunk to zero, preventing
/// spectral leakage (artefacts from the FFT assuming the signal is periodic).
/// Hann is the standard choice — same as what librosa uses by default.
///
/// Python equivalent: scipy.signal.hann(size)
fn hann_window(size: usize) -> Vec<f32> {
    // `(0..size)` is a Range — Rust's equivalent of Python's range(size).
    // `.map(|i| ...)` transforms each element — like a list comprehension.
    // `.collect()` gathers the iterator into a Vec.
    //
    // This whole pattern: range → map → collect is VERY common in Rust.
    // It's like Python's `[expression for i in range(size)]` but lazy —
    // the iterator doesn't allocate intermediate results.
    (0..size)
        .map(|i| {
            let t = i as f32 / (size - 1) as f32;
            0.5 * (1.0 - (2.0 * PI * t).cos())
        })
        .collect()
}

/// Compute the Short-Time Fourier Transform (STFT).
///
/// This is the core operation that everything else builds on.
/// It chops the audio into overlapping windows, applies a Hann window
/// to each, then FFTs each window to get frequency content over time.
///
/// # Arguments
/// * `samples` - Raw audio samples (mono, f32)
/// * `n_fft` - FFT window size (default 2048 in librosa). Larger = better
///   frequency resolution but worse time resolution. It's a tradeoff.
/// * `hop_length` - Step size between windows (default n_fft/4 = 512).
///   Smaller = more overlap = smoother time axis but more computation.
///
/// # Returns
/// A `Spectrogram` containing magnitude data for each time frame.
pub fn stft(samples: &[f32], n_fft: usize, hop_length: usize) -> Spectrogram {
    // `&[f32]` is a "slice" — a borrowed view into a contiguous block of f32s.
    // It's like &Vec<f32> but more general: it can reference part of a Vec,
    // an array, or any contiguous memory. Prefer slices in function signatures
    // for flexibility — the caller can pass a Vec, a sub-slice, etc.

    // Create the FFT planner. RustFFT plans the optimal FFT algorithm
    // for the given size, then reuses it for every window. This is much
    // faster than planning each time (like FFTW's "plan" concept).
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Pre-compute the Hann window
    let window = hann_window(n_fft);

    // Number of frequency bins: we only keep the positive half of the FFT
    // output (it's symmetric for real-valued input).
    let n_freq_bins = n_fft / 2 + 1;

    // Calculate the number of frames
    let n_frames = if samples.len() >= n_fft {
        (samples.len() - n_fft) / hop_length + 1
    } else {
        0
    };

    // Pre-allocate the output. `Vec::with_capacity` is like Python's
    // equivalent of creating a list when you know the final size —
    // avoids repeated reallocation as the Vec grows.
    let mut magnitudes = Vec::with_capacity(n_frames);

    // Reusable buffer for FFT input — avoids allocating a new Vec each frame.
    // `mut` because we'll overwrite it each iteration.
    let mut fft_buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];
    // `vec![value; count]` creates a Vec of `count` copies of `value`.
    // Complex<f32> has real and imaginary parts — FFT works with complex numbers.

    // Process each frame
    for frame in 0..n_frames {
        let start = frame * hop_length;

        // Fill the FFT buffer: multiply each sample by the window function.
        // `.iter()` creates an iterator over references to the elements.
        // `.zip()` pairs up two iterators element-by-element — like Python's zip().
        // `.enumerate()` adds an index — like Python's enumerate().
        for (i, (sample, &w)) in samples[start..start + n_fft]
            .iter()
            .zip(window.iter())
            .enumerate()
        {
            // `&w` pattern-matches the reference to get the value directly.
            // This is "destructuring" — Rust lets you unpack values in patterns.
            fft_buffer[i] = Complex::new(sample * w, 0.0);
        }

        // Run the FFT in-place (modifies fft_buffer directly).
        fft.process(&mut fft_buffer);

        // Extract magnitudes for positive frequencies only.
        // `.iter()` → `.map()` → `.collect()` : the classic Rust data pipeline.
        let frame_magnitudes: Vec<f32> = fft_buffer[..n_freq_bins]
            .iter()
            .map(|c| c.norm() / n_fft as f32) // .norm() = sqrt(re² + im²), then normalise
            .collect();

        magnitudes.push(frame_magnitudes);
    }

    // Return the spectrogram. No semicolon = implicit return.
    Spectrogram {
        magnitudes,
        sample_rate: 0, // Will be set by the caller — we don't know it here
        n_fft,
        hop_length,
        n_frames,
        n_freq_bins,
    }
}

/// Compute the spectrogram from audio samples with a known sample rate.
///
/// This is the main public API — wraps `stft()` and sets the sample rate.
/// Uses librosa-compatible defaults: n_fft=2048, hop_length=512.
pub fn compute_spectrogram(
    samples: &[f32],
    sample_rate: u32,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Spectrogram {
    // `Option<T>` is Rust's null-replacement: either Some(value) or None.
    // `.unwrap_or(default)` extracts the value or uses a default — clean
    // and safe, unlike null checks that you might forget.
    let n_fft = n_fft.unwrap_or(2048);
    let hop_length = hop_length.unwrap_or(n_fft / 4);

    let mut spec = stft(samples, n_fft, hop_length);
    spec.sample_rate = sample_rate;
    spec
}

/// Convert spectrogram magnitudes to decibels (dB scale).
///
/// This compresses the dynamic range for better visualisation/analysis.
/// Same as librosa.amplitude_to_db().
pub fn to_db(magnitudes: &[Vec<f32>]) -> Vec<Vec<f32>> {
    // Find the global maximum for reference
    let max_val = magnitudes
        .iter()
        .flat_map(|frame| frame.iter())  // flatten 2D → 1D iterator
        .cloned()                         // convert &f32 → f32
        .fold(f32::MIN, f32::max);        // find maximum
    // `.fold()` is like Python's functools.reduce(). Starting from f32::MIN,
    // it applies f32::max to each element, accumulating the result.

    let ref_val = if max_val > 0.0 { max_val } else { 1.0 };

    magnitudes
        .iter()
        .map(|frame| {
            frame
                .iter()
                .map(|&mag| {
                    // 20 * log10(mag / ref) with a floor to avoid log(0)
                    let val = if mag > 1e-10 { mag } else { 1e-10 };
                    20.0 * (val / ref_val).log10()
                })
                .collect()
        })
        .collect()
}

/// Compute the spectral centroid for each frame.
///
/// The spectral centroid is the "centre of mass" of the spectrum — it
/// indicates the perceived brightness of the sound. High centroid = bright/sharp,
/// low centroid = dark/mellow.
///
/// Python equivalent: librosa.feature.spectral_centroid()
pub fn spectral_centroid(spectrogram: &Spectrogram) -> Vec<f32> {
    spectrogram
        .magnitudes
        .iter()
        .map(|frame| {
            let mut weighted_sum = 0.0_f32;
            let mut magnitude_sum = 0.0_f32;

            for (bin, &mag) in frame.iter().enumerate() {
                let freq = spectrogram.bin_to_freq(bin);
                weighted_sum += freq * mag;
                magnitude_sum += mag;
            }

            if magnitude_sum > 1e-10 {
                weighted_sum / magnitude_sum
            } else {
                0.0
            }
        })
        .collect()
}

/// Compute the spectral bandwidth for each frame.
///
/// Measures the width/spread of the spectrum around the centroid.
/// Wide bandwidth = rich/complex sound, narrow = pure/simple tone.
///
/// Python equivalent: librosa.feature.spectral_bandwidth()
pub fn spectral_bandwidth(spectrogram: &Spectrogram) -> Vec<f32> {
    let centroids = spectral_centroid(spectrogram);

    spectrogram
        .magnitudes
        .iter()
        .zip(centroids.iter())
        .map(|(frame, &centroid)| {
            let mut weighted_sum = 0.0_f32;
            let mut magnitude_sum = 0.0_f32;

            for (bin, &mag) in frame.iter().enumerate() {
                let freq = spectrogram.bin_to_freq(bin);
                let diff = freq - centroid;
                weighted_sum += diff * diff * mag;
                magnitude_sum += mag;
            }

            if magnitude_sum > 1e-10 {
                (weighted_sum / magnitude_sum).sqrt()
            } else {
                0.0
            }
        })
        .collect()
}

/// Compute the spectral rolloff for each frame.
///
/// The frequency below which a given percentage (default 85%) of the
/// total spectral energy is contained. Indicates where the "action" is
/// in the frequency spectrum.
///
/// Python equivalent: librosa.feature.spectral_rolloff()
pub fn spectral_rolloff(spectrogram: &Spectrogram, roll_percent: Option<f32>) -> Vec<f32> {
    let threshold = roll_percent.unwrap_or(0.85);

    spectrogram
        .magnitudes
        .iter()
        .map(|frame| {
            let total_energy: f32 = frame.iter().sum();
            let target = total_energy * threshold;

            let mut cumulative = 0.0_f32;
            for (bin, &mag) in frame.iter().enumerate() {
                cumulative += mag;
                if cumulative >= target {
                    return spectrogram.bin_to_freq(bin);
                }
            }

            // If we get here, return the Nyquist frequency
            spectrogram.bin_to_freq(spectrogram.n_freq_bins - 1)
        })
        .collect()
}

/// Compute spectral flatness for each frame.
///
/// Ratio of geometric mean to arithmetic mean of the spectrum.
/// Values near 1.0 = noise-like (flat spectrum), near 0.0 = tonal (peaked spectrum).
///
/// Python equivalent: librosa.feature.spectral_flatness()
pub fn spectral_flatness(spectrogram: &Spectrogram) -> Vec<f32> {
    spectrogram
        .magnitudes
        .iter()
        .map(|frame| {
            let n = frame.len() as f32;

            // Geometric mean via log: exp(mean(log(x)))
            // We add a tiny epsilon to avoid log(0).
            let log_sum: f32 = frame.iter().map(|&m| (m + 1e-10).ln()).sum();
            let geo_mean = (log_sum / n).exp();

            // Arithmetic mean
            let arith_mean: f32 = frame.iter().sum::<f32>() / n;

            if arith_mean > 1e-10 {
                geo_mean / arith_mean
            } else {
                0.0
            }
        })
        .collect()
}

// ---- Mel Filterbank & MFCCs ----
//
// MFCCs (Mel-Frequency Cepstral Coefficients) are the standard way to
// describe timbre — the "colour" of a sound that distinguishes a piano
// from a guitar playing the same note. They work by:
//
// 1. Mapping the spectrogram onto a mel scale (perceptual frequency scale)
// 2. Taking the log of the mel energies
// 3. Applying a DCT (Discrete Cosine Transform) to decorrelate them
//
// The result is a compact set of coefficients (typically 13) that capture
// the shape of the spectral envelope — the broad strokes of "what does
// this sound like" rather than "what note is playing."

/// Build a mel filterbank matrix.
///
/// Creates `n_mels` triangular filters spaced on the mel scale between
/// `fmin` and `fmax`. Each filter maps a range of FFT bins to one mel band.
///
/// Returns a 2D array: filterbank[mel_band][fft_bin]
fn mel_filterbank(
    n_fft: usize,
    sample_rate: u32,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    let n_freq_bins = n_fft / 2 + 1;

    // Hz to mel conversion: mel = 2595 * log10(1 + f/700)
    let hz_to_mel = |f: f32| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f32| 700.0 * (10.0_f32.powf(m / 2595.0) - 1.0);

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 points: includes the edges of the first and last filters
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert mel points back to Hz, then to FFT bin indices
    let bin_points: Vec<f32> = mel_points
        .iter()
        .map(|&m| mel_to_hz(m) * n_fft as f32 / sample_rate as f32)
        .collect();

    // Build triangular filters
    let mut filterbank = vec![vec![0.0_f32; n_freq_bins]; n_mels];

    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for k in 0..n_freq_bins {
            let k_f = k as f32;
            if k_f > left && k_f < center {
                // Rising slope
                filterbank[m][k] = (k_f - left) / (center - left);
            } else if k_f >= center && k_f < right {
                // Falling slope
                filterbank[m][k] = (right - k_f) / (right - center);
            }
        }
    }

    filterbank
}

/// Compute MFCCs (Mel-Frequency Cepstral Coefficients) from a spectrogram.
///
/// Returns `n_mfcc` coefficients per frame. The first coefficient (MFCC-0)
/// represents overall energy; coefficients 1-12 capture the spectral shape.
///
/// # Arguments
/// * `spectrogram` - The STFT spectrogram
/// * `n_mfcc` - Number of coefficients to return (default: 13)
/// * `n_mels` - Number of mel filter bands (default: 40)
///
/// Python equivalent: librosa.feature.mfcc()
pub fn compute_mfccs(
    spectrogram: &Spectrogram,
    n_mfcc: Option<usize>,
    n_mels: Option<usize>,
) -> Vec<Vec<f32>> {
    let n_mfcc = n_mfcc.unwrap_or(13);
    let n_mels = n_mels.unwrap_or(40);
    let fmax = spectrogram.sample_rate as f32 / 2.0; // Nyquist

    let filterbank = mel_filterbank(
        spectrogram.n_fft,
        spectrogram.sample_rate,
        n_mels,
        0.0,
        fmax,
    );

    spectrogram
        .magnitudes
        .iter()
        .map(|frame| {
            // Step 1: Apply mel filterbank — dot product of each filter with the frame
            let mel_energies: Vec<f32> = filterbank
                .iter()
                .map(|filter| {
                    let energy: f32 = filter
                        .iter()
                        .zip(frame.iter())
                        .map(|(&f, &m)| f * m * m) // power spectrum (magnitude squared)
                        .sum();
                    // Log of energy (with floor to avoid log(0))
                    (energy.max(1e-10)).ln()
                })
                .collect();

            // Step 2: DCT Type-II to get cepstral coefficients
            // This decorrelates the mel energies into independent components.
            // DCT-II: X[k] = sum_n x[n] * cos(pi/N * (n + 0.5) * k)
            let n = mel_energies.len() as f32;
            (0..n_mfcc)
                .map(|k| {
                    let sum: f32 = mel_energies
                        .iter()
                        .enumerate()
                        .map(|(i, &val)| {
                            val * (std::f32::consts::PI / n * (i as f32 + 0.5) * k as f32).cos()
                        })
                        .sum();
                    sum * (2.0 / n).sqrt() // Orthonormal scaling
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: generate a simple sine wave for testing
    fn sine_wave(freq: f32, sample_rate: u32, duration: f32) -> Vec<f32> {
        let n_samples = (sample_rate as f32 * duration) as usize;
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_hann_window() {
        let w = hann_window(4);
        // Hann window: edges should be near zero, middle should be near 1
        assert!(w[0] < 0.01);           // first sample ≈ 0
        assert!(w[3] < 0.01);           // last sample ≈ 0
        assert!((w[1] - 0.75).abs() < 0.01); // known value for size=4
        assert!((w[2] - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_stft_output_shape() {
        let samples = sine_wave(440.0, 44100, 1.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);

        // With n_fft=2048 and hop=512, we should get:
        // n_freq_bins = 1025
        // n_frames = (44100 - 2048) / 512 + 1 = 83 (approximately)
        assert_eq!(spec.n_freq_bins, 1025);
        assert!(spec.n_frames > 80);
        assert_eq!(spec.magnitudes.len(), spec.n_frames);
        assert_eq!(spec.magnitudes[0].len(), spec.n_freq_bins);
    }

    #[test]
    fn test_sine_wave_peak_frequency() {
        // A 440 Hz sine wave should have its peak energy near bin 440 Hz
        let samples = sine_wave(440.0, 44100, 1.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);

        // Find the bin with maximum magnitude in the middle frame
        let mid_frame = spec.n_frames / 2;
        let (peak_bin, _) = spec.magnitudes[mid_frame]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            // `.max_by()` finds the maximum using a custom comparator.
            // `partial_cmp` is needed because f32 can be NaN, so Rust
            // won't let you use regular comparison (it's not a total order).
            // `.unwrap()` here is safe because we know our data has no NaN.
            .unwrap();

        let peak_freq = spec.bin_to_freq(peak_bin);

        // Should be within ~25 Hz of 440 Hz (frequency resolution = sr/n_fft ≈ 21.5 Hz)
        assert!(
            (peak_freq - 440.0).abs() < 25.0,
            "Expected peak near 440 Hz, got {} Hz",
            peak_freq
        );
    }

    #[test]
    fn test_spectral_centroid_sine() {
        // For a pure sine wave, the spectral centroid should be near the frequency
        let samples = sine_wave(1000.0, 44100, 1.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let centroids = spectral_centroid(&spec);

        // Check the middle frame's centroid is near 1000 Hz
        let mid = centroids[centroids.len() / 2];
        assert!(
            (mid - 1000.0).abs() < 100.0,
            "Expected centroid near 1000 Hz, got {} Hz",
            mid
        );
    }

    #[test]
    fn test_mfcc_shape() {
        let samples = sine_wave(440.0, 44100, 1.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let mfccs = compute_mfccs(&spec, None, None);

        // Should have same number of frames as spectrogram
        assert_eq!(mfccs.len(), spec.n_frames);
        // Default 13 coefficients per frame
        assert_eq!(mfccs[0].len(), 13);
    }

    #[test]
    fn test_mfcc_custom_count() {
        let samples = sine_wave(440.0, 44100, 0.5);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let mfccs = compute_mfccs(&spec, Some(20), Some(64));

        assert_eq!(mfccs[0].len(), 20);
    }

    #[test]
    fn test_mfcc_different_timbres() {
        // A pure sine and a composite signal should have different MFCCs
        let sine = sine_wave(440.0, 44100, 0.5);
        // Composite: fundamental + harmonics (more like a real instrument)
        let composite: Vec<f32> = (0..(44100.0 * 0.5) as usize)
            .map(|i| {
                let t = i as f32 / 44100.0;
                0.5 * (2.0 * PI * 440.0 * t).sin()
                    + 0.3 * (2.0 * PI * 880.0 * t).sin()
                    + 0.2 * (2.0 * PI * 1320.0 * t).sin()
            })
            .collect();

        let spec_sine = compute_spectrogram(&sine, 44100, None, None);
        let spec_comp = compute_spectrogram(&composite, 44100, None, None);
        let mfcc_sine = compute_mfccs(&spec_sine, None, None);
        let mfcc_comp = compute_mfccs(&spec_comp, None, None);

        // Compare middle frames — they should differ in higher coefficients
        // (which capture spectral shape differences)
        let mid = mfcc_sine.len() / 2;
        let diff: f32 = mfcc_sine[mid]
            .iter()
            .zip(mfcc_comp[mid].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            diff > 1.0,
            "Pure sine and composite should have different MFCCs, total diff was {:.3}",
            diff
        );
    }

    #[test]
    fn test_spectral_flatness_sine_vs_noise() {
        // A pure sine wave should have low spectral flatness (tonal)
        let sine = sine_wave(440.0, 44100, 0.5);
        let spec_sine = compute_spectrogram(&sine, 44100, None, None);
        let flatness_sine = spectral_flatness(&spec_sine);
        let avg_sine: f32 = flatness_sine.iter().sum::<f32>() / flatness_sine.len() as f32;

        // Sine wave flatness should be low (close to 0)
        assert!(
            avg_sine < 0.1,
            "Sine wave flatness should be low, got {}",
            avg_sine
        );
    }
}
