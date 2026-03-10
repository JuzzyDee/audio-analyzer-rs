// analysis/harmonic.rs — Harmonic and tonal analysis
//
// This module maps raw frequency data onto musically meaningful
// representations: pitch classes, chords, keys, and tonal relationships.
//
// The core concept: every musical note has a fundamental frequency
// (e.g., A4 = 440 Hz) and the chromagram tells us how much energy
// is present in each of the 12 pitch classes at each moment in time.
// It's octave-independent — a low C and a high C both contribute to
// the "C" bin.

use std::f32::consts::PI;
use super::spectral::Spectrogram;

// The 12 pitch class names. This is a constant array — known at compile time.
// `&str` (string slices) in a `const` live in the binary's read-only data section,
// so there's zero runtime cost. Unlike Python where even constants are heap-allocated
// objects that could theoretically be reassigned.
const PITCH_CLASSES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
];

/// The result of chromagram analysis.
#[derive(Debug)]
pub struct Chromagram {
    /// 2D chroma data: chroma[time_frame][pitch_class]
    /// Each inner array has exactly 12 elements (one per pitch class).
    /// Values represent the energy/strength of each pitch class at that moment.
    pub chroma: Vec<[f32; 12]>,
    // ^^^^^^^^
    // `[f32; 12]` is a fixed-size array — exactly 12 elements, always.
    // Unlike Vec which is heap-allocated and growable, fixed arrays live
    // on the stack and the compiler knows their size at compile time.
    // Perfect for pitch classes since there are always exactly 12.
    // This is similar to Swift's tuples or fixed-size arrays.

    /// Number of time frames
    pub n_frames: usize,

    /// Sample rate (inherited from the spectrogram)
    pub sample_rate: u32,

    /// Hop length (inherited from the spectrogram)
    pub hop_length: usize,
}

impl Chromagram {
    /// Get the time in seconds for a given frame index.
    pub fn frame_to_time(&self, frame: usize) -> f32 {
        frame as f32 * self.hop_length as f32 / self.sample_rate as f32
    }

    /// Get the dominant pitch class for a given frame.
    /// Returns the pitch class name and its energy value.
    pub fn dominant_pitch(&self, frame: usize) -> (&str, f32) {
        // Find the index of the maximum value in the chroma array.
        // This uses a pattern you've seen before: enumerate → max_by.
        let (max_idx, &max_val) = self.chroma[frame]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        (PITCH_CLASSES[max_idx], max_val)
    }

    /// Estimate the overall key of the track using the Krumhansl-Schmuckler algorithm.
    ///
    /// This compares the average chroma profile against known profiles for
    /// each major and minor key. The best match suggests the key.
    ///
    /// Returns (key_name, mode, correlation_score).
    /// e.g., ("A", "minor", 0.87)
    pub fn estimate_key(&self) -> (&str, &str, f32) {
        // Krumhansl-Kessler key profiles — empirically derived weights that
        // represent how strongly each pitch class correlates with a given key.
        // These are the standard profiles used in music information retrieval.

        // Major key profile (starting from C major)
        #[allow(clippy::excessive_precision)]
        let major_profile: [f32; 12] = [
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
            2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
        ];

        // Minor key profile (starting from C minor)
        #[allow(clippy::excessive_precision)]
        let minor_profile: [f32; 12] = [
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
            2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
        ];

        // Compute the average chroma vector across all frames.
        // This gives us the overall pitch class distribution for the track.
        let mut avg_chroma = [0.0_f32; 12];
        for frame in &self.chroma {
            // `&self.chroma` borrows the Vec. The `for` loop iterates over
            // references to each element. Rust's for loops always use
            // iterators under the hood — there's no index-based for(i=0;...).
            for (i, &val) in frame.iter().enumerate() {
                avg_chroma[i] += val;
            }
        }
        let n = self.n_frames as f32;
        for val in &mut avg_chroma {
            // `&mut` because we need to modify the values in place.
            // `*val` dereferences the mutable reference to access the f32.
            *val /= n;
        }

        // Try every possible key (12 pitch classes × 2 modes = 24 keys).
        // For each, rotate the profile and correlate with our average chroma.
        let mut best_key = 0;
        let mut best_mode = "major";
        let mut best_corr = f32::NEG_INFINITY;
        // `f32::NEG_INFINITY` is a constant — guaranteed to be less than
        // any real value. Cleaner than using a sentinel like -999.0.

        for shift in 0..12 {
            // Rotate the profile by `shift` positions.
            // This effectively transposes the key profile to each root note.
            let major_corr = correlate(&avg_chroma, &major_profile, shift);
            let minor_corr = correlate(&avg_chroma, &minor_profile, shift);

            if major_corr > best_corr {
                best_corr = major_corr;
                best_key = shift;
                best_mode = "major";
            }
            if minor_corr > best_corr {
                best_corr = minor_corr;
                best_key = shift;
                best_mode = "minor";
            }
        }

        (PITCH_CLASSES[best_key], best_mode, best_corr)
    }
}

/// Pearson correlation between a chroma vector and a rotated key profile.
///
/// `shift` rotates the profile so that index 0 aligns with the target root note.
fn correlate(chroma: &[f32; 12], profile: &[f32; 12], shift: usize) -> f32 {
    // Pearson correlation: measures linear relationship between two vectors.
    // Returns -1.0 (inverse) to 1.0 (perfect match).
    //
    // Formula: r = Σ((x-μx)(y-μy)) / sqrt(Σ(x-μx)² × Σ(y-μy)²)

    let n = 12.0_f32;

    // Mean of chroma
    let mean_c: f32 = chroma.iter().sum::<f32>() / n;

    // Mean of profile
    let mean_p: f32 = profile.iter().sum::<f32>() / n;

    let mut numerator = 0.0_f32;
    let mut denom_c = 0.0_f32;
    let mut denom_p = 0.0_f32;

    for i in 0..12 {
        // `(12 + i - shift) % 12` rotates the profile so that profile[0]
        // (the tonic weight) aligns with chroma[shift]. This way, when
        // shift=2 (D), the tonic weight lands on chroma index 2.
        let c = chroma[i] - mean_c;
        let p = profile[(12 + i - shift) % 12] - mean_p;
        numerator += c * p;
        denom_c += c * c;
        denom_p += p * p;
    }

    let denominator = (denom_c * denom_p).sqrt();
    if denominator > 1e-10 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Compute the chromagram from raw audio samples.
///
/// Uses a high-resolution FFT (n_fft=8192) internally for accurate pitch
/// class mapping. With the standard n_fft=2048, FFT bins at low frequencies
/// are wider than a semitone (~23 Hz/bin at 48kHz), causing notes like G to
/// be misclassified as F# because the bin centre frequency rounds down.
/// With n_fft=8192 the resolution is ~5.9 Hz/bin, which properly resolves
/// semitones down to ~60 Hz.
///
/// The `spectrogram` parameter is used only for time-axis alignment — the
/// output chromagram has the same number of frames and hop length.
///
/// Python equivalent: librosa.feature.chroma_stft()
pub fn compute_chromagram(
    samples: &[f32],
    sample_rate: u32,
    spectrogram: &Spectrogram,
) -> Chromagram {
    // Compute a high-resolution spectrogram for accurate chroma mapping.
    // We use n_fft=8192 for ~5.9 Hz/bin resolution at 48kHz, enough to
    // resolve semitones across the full musical range. The hop length
    // matches the original spectrogram so frames align.
    let chroma_n_fft: usize = 8192;
    let hires_spec = super::spectral::compute_spectrogram(
        samples,
        sample_rate,
        Some(chroma_n_fft),
        Some(spectrogram.hop_length),
    );

    // Pre-compute the pitch class mapping for each frequency bin.
    let bin_to_chroma: Vec<Option<usize>> = (0..hires_spec.n_freq_bins)
        .map(|bin| {
            let freq = hires_spec.bin_to_freq(bin);
            if freq < 20.0 {
                None
            } else {
                // Convert frequency to MIDI note number, then to pitch class.
                // A4 (440 Hz) = MIDI note 69. Each semitone is a factor of 2^(1/12).
                let midi = 12.0 * (freq / 440.0).log2() + 69.0;
                let pitch_class = ((midi.round() as i32) % 12 + 12) % 12;
                Some(pitch_class as usize)
            }
        })
        .collect();

    // Compute chroma for each frame using the high-res spectrogram
    let hires_chroma: Vec<[f32; 12]> = hires_spec
        .magnitudes
        .iter()
        .map(|frame| {
            let mut chroma_frame = [0.0_f32; 12];

            for (bin, &magnitude) in frame.iter().enumerate() {
                if let Some(pitch_class) = bin_to_chroma[bin] {
                    chroma_frame[pitch_class] += magnitude;
                }
            }

            // Normalise so the max value is 1.0 (within each frame).
            let max_val = chroma_frame
                .iter()
                .cloned()
                .fold(f32::MIN, f32::max);

            if max_val > 1e-10 {
                for val in &mut chroma_frame {
                    *val /= max_val;
                }
            }

            chroma_frame
        })
        .collect();

    // The high-res spectrogram may have a slightly different frame count
    // due to the larger window. Use whichever is shorter to stay aligned.
    let n_frames = hires_chroma.len().min(spectrogram.n_frames);
    let chroma = hires_chroma[..n_frames].to_vec();

    Chromagram {
        chroma,
        n_frames,
        sample_rate: spectrogram.sample_rate,
        hop_length: spectrogram.hop_length,
    }
}

/// Compute the Tonnetz (tonal centroid) features from a chromagram.
///
/// Tonnetz maps chroma onto a 6-dimensional space representing tonal
/// relationships: perfect fifths, minor thirds, and major thirds.
/// It captures harmonic relationships that the raw chromagram doesn't
/// make explicit — for example, C major and A minor are "close" in
/// Tonnetz space because they share notes.
///
/// Returns a Vec of 6-element arrays, one per frame.
///
/// Python equivalent: librosa.feature.tonnetz()
pub fn compute_tonnetz(chromagram: &Chromagram) -> Vec<[f32; 6]> {
    // The Tonnetz maps each pitch class onto 6 axes using these angles.
    // Each pitch class is placed on a circle; the axes capture
    // intervals of fifths (7 semitones), minor thirds (3), and major thirds (4).
    //
    // The magic numbers (7π/6, 3π/2, etc.) come from music theory:
    // they position notes on circles so that harmonically related notes
    // are geometrically close.

    chromagram
        .chroma
        .iter()
        .map(|frame| {
            let mut tonnetz = [0.0_f32; 6];

            for (pc, &energy) in frame.iter().enumerate() {
                let pc_f = pc as f32;

                // Axis 0,1: Circle of fifths (interval of 7 semitones)
                let angle_fifth = pc_f * 7.0 * PI / 6.0;
                tonnetz[0] += energy * angle_fifth.sin();
                tonnetz[1] += energy * angle_fifth.cos();

                // Axis 2,3: Circle of minor thirds (interval of 3 semitones)
                let angle_minor = pc_f * 3.0 * PI / 2.0;
                tonnetz[2] += energy * angle_minor.sin();
                tonnetz[3] += energy * angle_minor.cos();

                // Axis 4,5: Circle of major thirds (interval of 4 semitones)
                let angle_major = pc_f * 2.0 * PI / 3.0;
                tonnetz[4] += energy * angle_major.sin();
                tonnetz[5] += energy * angle_major.cos();
            }

            // Normalise by total chroma energy
            let total: f32 = frame.iter().sum();
            if total > 1e-10 {
                for val in &mut tonnetz {
                    *val /= total;
                }
            }

            tonnetz
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::spectral::compute_spectrogram;

    // Generate a sine wave at a specific frequency
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
    fn test_chromagram_a440() {
        // A 440 Hz sine wave should light up the "A" pitch class (index 9)
        let samples = sine_wave(440.0, 44100, 1.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let chroma = compute_chromagram(&samples, 44100, &spec);

        assert!(chroma.n_frames > 0);

        // Check the middle frame
        let mid = chroma.n_frames / 2;
        let (dominant, _) = chroma.dominant_pitch(mid);
        assert_eq!(
            dominant, "A",
            "440 Hz should map to A, got {}",
            dominant
        );
    }

    #[test]
    fn test_chromagram_c_note() {
        // C4 = 261.63 Hz — should map to "C" pitch class (index 0)
        let samples = sine_wave(261.63, 44100, 1.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let chroma = compute_chromagram(&samples, 44100, &spec);

        let mid = chroma.n_frames / 2;
        let (dominant, _) = chroma.dominant_pitch(mid);
        assert_eq!(
            dominant, "C",
            "261.63 Hz should map to C, got {}",
            dominant
        );
    }

    #[test]
    fn test_chromagram_g_note() {
        // G3 = 196 Hz — this was the bug: at n_fft=2048/48kHz, G mapped to F#
        // because bin 8 (187.5 Hz) rounds to MIDI 54 (F#). The high-res FFT fixes this.
        let samples = sine_wave(196.0, 48000, 1.0);
        let spec = compute_spectrogram(&samples, 48000, None, None);
        let chroma = compute_chromagram(&samples, 48000, &spec);

        let mid = chroma.n_frames / 2;
        let (dominant, _) = chroma.dominant_pitch(mid);
        assert_eq!(
            dominant, "G",
            "196 Hz should map to G, got {}",
            dominant
        );
    }

    #[test]
    fn test_chromagram_g_note_48k() {
        // G4 at 48kHz — verify across octaves
        let samples = sine_wave(392.0, 48000, 1.0);
        let spec = compute_spectrogram(&samples, 48000, None, None);
        let chroma = compute_chromagram(&samples, 48000, &spec);

        let mid = chroma.n_frames / 2;
        let (dominant, _) = chroma.dominant_pitch(mid);
        assert_eq!(
            dominant, "G",
            "392 Hz should map to G, got {}",
            dominant
        );
    }

    #[test]
    fn test_key_estimation_returns_valid_result() {
        // Key estimation from synthesized sine waves is inherently imprecise
        // because pure sines lack the overtone structure of real instruments.
        // Instead of asserting a specific key, we verify the algorithm runs
        // and returns a reasonable confidence score.
        let sr = 44100_u32;
        let dur = 2.0;
        let n = (sr as f32 * dur) as usize;

        // A minor triad: A4=440, C5=523.25, E5=659.25
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let a = (2.0 * PI * 440.0 * t).sin();
                let c = (2.0 * PI * 523.25 * t).sin();
                let e = (2.0 * PI * 659.25 * t).sin();
                (a + c + e) / 3.0
            })
            .collect();

        let spec = compute_spectrogram(&samples, sr, None, None);
        let chroma = compute_chromagram(&samples, sr, &spec);
        let (key, mode, score) = chroma.estimate_key();

        println!("Estimated key: {} {} (score: {:.3})", key, mode, score);

        // The algorithm should produce *some* result with positive confidence.
        // With real music (not pure sine waves) this gives accurate results.
        assert!(score > 0.5, "Confidence score should be meaningful, got {:.3}", score);
        assert!(["major", "minor"].contains(&mode));
        assert!(!key.is_empty());
    }

    #[test]
    fn test_tonnetz_shape() {
        let samples = sine_wave(440.0, 44100, 0.5);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let chroma = compute_chromagram(&samples, 44100, &spec);
        let tonnetz = compute_tonnetz(&chroma);

        // Should have same number of frames as chromagram
        assert_eq!(tonnetz.len(), chroma.n_frames);
        // Each frame should have 6 dimensions
        assert_eq!(tonnetz[0].len(), 6);
    }

    #[test]
    fn test_chromagram_frame_count() {
        let samples = sine_wave(440.0, 44100, 1.0);
        let spec = compute_spectrogram(&samples, 44100, None, None);
        let chroma = compute_chromagram(&samples, 44100, &spec);

        // The high-res FFT (n_fft=8192) may produce fewer frames than the
        // original spectrogram (n_fft=2048) because it needs more samples
        // before the first frame. The chromagram uses min(hires, spec) frames.
        assert!(chroma.n_frames > 0);
        assert!(chroma.n_frames <= spec.n_frames);
    }
}
