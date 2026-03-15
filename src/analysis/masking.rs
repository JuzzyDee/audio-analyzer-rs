// analysis/masking.rs — Frequency masking detection
//
// Detects where frequency bands are crowded or competing, helping producers
// identify muddy areas in their mix. Three complementary indicators:
//
// 1. **Crowding score**: high energy + low spectral contrast = lots of sound
//    but nothing stands out. The classic "muddy mix" signature.
//
// 2. **Harmonic-percussive collision**: when both the sustained (harmonic) and
//    transient (percussive) components have significant energy in the same band,
//    they're fighting for attention. Classic examples: kick vs bass, snare vs vocal.
//
// 3. **Cross-band correlation**: if adjacent frequency bands have highly correlated
//    energy over time, something is bleeding across band boundaries. Low correlation
//    between neighbours = clean spectral separation.

use std::fmt::Write;

use super::spectral::{BandEnergy, FREQUENCY_BANDS, SpectralContrast, Spectrogram};

/// Per-frame masking analysis across 7 frequency bands.
#[derive(Debug)]
pub struct MaskingAnalysis {
    /// Crowding score per band per frame: crowding[frame][band_index]
    /// 0.0 = clean/open, 1.0 = maximally crowded.
    /// Derived from: normalised(band_energy) * (1.0 - normalised(contrast))
    pub crowding: Vec<[f32; 7]>,

    /// Harmonic-percussive collision per band per frame: hp_collision[frame][band_index]
    /// 0.0 = one component dominates, 1.0 = both equally strong.
    /// Derived from: min(h, p) / max(h, p)
    pub hp_collision: Vec<[f32; 7]>,

    /// Number of frames
    pub n_frames: usize,
}

/// Cross-band energy correlation between adjacent band pairs.
/// Computed over the full track (summary stat, not per-frame).
#[derive(Debug)]
pub struct CrossBandCorrelation {
    /// Pearson correlation between adjacent band pairs.
    /// 6 values: (sub_bass,bass), (bass,low_mid), (low_mid,mid),
    /// (mid,upper_mid), (upper_mid,presence), (presence,brilliance)
    pub correlations: [f32; 6],
}

/// Combined masking detection result.
#[derive(Debug)]
pub struct MaskingResult {
    /// Per-frame masking data
    pub analysis: MaskingAnalysis,
    /// Cross-band correlation (track-level summary)
    pub cross_band: CrossBandCorrelation,
}

/// Band pair names for display.
const BAND_PAIRS: [(&str, &str); 6] = [
    ("sub_bass", "bass"),
    ("bass", "low_mid"),
    ("low_mid", "mid"),
    ("mid", "upper_mid"),
    ("upper_mid", "presence"),
    ("presence", "brilliance"),
];

/// Build a Spectrogram from HPSS magnitude data for band energy computation.
///
/// This wraps the raw HPSS output (Vec<Vec<f32>>) into a Spectrogram struct
/// so we can reuse `frequency_band_energy()` on the separated components.
pub fn spectrogram_from_hpss(
    hpss_mags: &[Vec<f32>],
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
) -> Spectrogram {
    Spectrogram {
        n_frames: hpss_mags.len(),
        n_freq_bins: hpss_mags.first().map_or(0, |f| f.len()),
        magnitudes: hpss_mags.to_vec(),
        sample_rate,
        n_fft,
        hop_length,
    }
}

/// Detect frequency masking from pre-computed analysis data.
///
/// Takes band energy and spectral contrast from the full spectrogram, plus
/// band energy computed separately on the harmonic and percussive HPSS
/// spectrograms. All inputs must be frame-aligned.
pub fn detect_masking(
    band_energy: &BandEnergy,
    contrast: &SpectralContrast,
    harmonic_band_energy: &BandEnergy,
    percussive_band_energy: &BandEnergy,
) -> MaskingResult {
    let n_frames = band_energy.n_frames;

    // --- Step 1: Compute per-band max energy and max contrast for normalisation ---
    let mut max_energy = [f32::MIN; 7];
    let mut max_contrast = [f32::MIN; 7];
    for frame in &band_energy.band_energies {
        for (i, &val) in frame.iter().enumerate() {
            if val > max_energy[i] {
                max_energy[i] = val;
            }
        }
    }
    for frame in &contrast.contrast {
        for (i, &val) in frame.iter().enumerate() {
            if val > max_contrast[i] {
                max_contrast[i] = val;
            }
        }
    }

    // --- Step 2: Compute crowding and HP collision per frame ---
    let mut crowding = Vec::with_capacity(n_frames);
    let mut hp_collision = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let mut frame_crowding = [0.0_f32; 7];
        let mut frame_hp = [0.0_f32; 7];

        for band in 0..7 {
            // Crowding: normalised energy * (1 - normalised contrast)
            let norm_energy = if max_energy[band] > 0.0 {
                band_energy.band_energies[frame_idx][band] / max_energy[band]
            } else {
                0.0
            };
            let norm_contrast = if max_contrast[band] > 0.0 {
                contrast.contrast[frame_idx][band] / max_contrast[band]
            } else {
                0.0
            };
            frame_crowding[band] = norm_energy * (1.0 - norm_contrast).max(0.0);

            // HP collision: min(h, p) / max(h, p)
            let h = harmonic_band_energy.band_energies[frame_idx][band];
            let p = percussive_band_energy.band_energies[frame_idx][band];
            let max_hp = h.max(p);
            let min_hp = h.min(p);
            frame_hp[band] = if max_hp > 1e-10 { min_hp / max_hp } else { 0.0 };
        }

        crowding.push(frame_crowding);
        hp_collision.push(frame_hp);
    }

    // --- Step 3: Cross-band energy correlation ---
    let cross_band = compute_cross_band_correlation(band_energy);

    MaskingResult {
        analysis: MaskingAnalysis {
            crowding,
            hp_collision,
            n_frames,
        },
        cross_band,
    }
}

/// Compute Pearson correlation between adjacent frequency band energy time-series.
fn compute_cross_band_correlation(band_energy: &BandEnergy) -> CrossBandCorrelation {
    let mut correlations = [0.0_f32; 6];
    let n = band_energy.n_frames as f32;

    if band_energy.n_frames < 2 {
        return CrossBandCorrelation { correlations };
    }

    for pair_idx in 0..6 {
        let band_a = pair_idx;
        let band_b = pair_idx + 1;

        // Compute means
        let mut sum_a = 0.0_f32;
        let mut sum_b = 0.0_f32;
        for frame in &band_energy.band_energies {
            sum_a += frame[band_a];
            sum_b += frame[band_b];
        }
        let mean_a = sum_a / n;
        let mean_b = sum_b / n;

        // Compute Pearson correlation
        let mut cov = 0.0_f32;
        let mut var_a = 0.0_f32;
        let mut var_b = 0.0_f32;
        for frame in &band_energy.band_energies {
            let da = frame[band_a] - mean_a;
            let db = frame[band_b] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        let denom = (var_a * var_b).sqrt();
        correlations[pair_idx] = if denom > 1e-10 { cov / denom } else { 0.0 };
    }

    CrossBandCorrelation { correlations }
}

/// Format a human-readable masking summary.
pub fn format_masking_summary(result: &MaskingResult) -> String {
    let mut out = String::with_capacity(1024);
    let n = result.analysis.n_frames as f32;

    // Average crowding per band
    let mut avg_crowding = [0.0_f32; 7];
    for frame in &result.analysis.crowding {
        for (i, &val) in frame.iter().enumerate() {
            avg_crowding[i] += val;
        }
    }
    for val in &mut avg_crowding {
        *val /= n;
    }

    // Average HP collision per band
    let mut avg_hp = [0.0_f32; 7];
    for frame in &result.analysis.hp_collision {
        for (i, &val) in frame.iter().enumerate() {
            avg_hp[i] += val;
        }
    }
    for val in &mut avg_hp {
        *val /= n;
    }

    out.push_str("\n── Frequency Masking ──\n");
    out.push_str("Crowding (high energy + low contrast):\n");
    for (i, &(name, lo, hi)) in FREQUENCY_BANDS.iter().enumerate() {
        let label = if avg_crowding[i] > 0.5 {
            "CROWDED"
        } else if avg_crowding[i] > 0.3 {
            "moderate"
        } else {
            "clean"
        };
        writeln!(
            out,
            "  {:<12} ({:>5.0}–{:>5.0} Hz): {:.3}  {}",
            name, lo, hi, avg_crowding[i], label
        )
        .unwrap();
    }

    // Only show HP collision for bands where it's notable
    let notable_hp: Vec<(usize, f32)> = avg_hp
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, v)| v > 0.3)
        .collect();
    if !notable_hp.is_empty() {
        out.push_str("H/P collision (harmonic + percussive fighting):\n");
        for (i, val) in &notable_hp {
            let label = if *val > 0.5 { "COLLISION" } else { "moderate" };
            writeln!(
                out,
                "  {:<12}: {:.3}  {}",
                FREQUENCY_BANDS[*i].0, val, label
            )
            .unwrap();
        }
    }

    // Only show cross-band correlations that are notable
    let notable_corr: Vec<(usize, f32)> = result
        .cross_band
        .correlations
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, v)| v > 0.7)
        .collect();
    if !notable_corr.is_empty() {
        out.push_str("Cross-band bleed:\n");
        for (i, val) in &notable_corr {
            let label = if *val > 0.85 { "HIGH" } else { "moderate" };
            writeln!(
                out,
                "  {}↔{}: {:.3}  {}",
                BAND_PAIRS[*i].0, BAND_PAIRS[*i].1, val, label
            )
            .unwrap();
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::spectral;

    /// Helper: build a spectrogram from raw magnitude data.
    fn make_spectrogram(mags: Vec<Vec<f32>>, sample_rate: u32, n_fft: usize) -> Spectrogram {
        Spectrogram {
            n_frames: mags.len(),
            n_freq_bins: mags.first().map_or(0, |f| f.len()),
            magnitudes: mags,
            sample_rate,
            n_fft,
            hop_length: 512,
        }
    }

    #[test]
    fn test_crowding_shape() {
        // Create a simple spectrogram with known dimensions
        let n_fft = 2048;
        let n_bins = n_fft / 2 + 1;
        let n_frames = 100;
        let mags = vec![vec![0.01_f32; n_bins]; n_frames];
        let spec = make_spectrogram(mags, 44100, n_fft);

        let bands = spectral::frequency_band_energy(&spec);
        let sc = spectral::spectral_contrast(&spec, None);
        let h_bands = spectral::frequency_band_energy(&spec);
        let p_bands = spectral::frequency_band_energy(&spec);

        let result = detect_masking(&bands, &sc, &h_bands, &p_bands);

        assert_eq!(result.analysis.crowding.len(), n_frames);
        assert_eq!(result.analysis.hp_collision.len(), n_frames);
        assert_eq!(result.analysis.n_frames, n_frames);
        assert_eq!(result.cross_band.correlations.len(), 6);
    }

    #[test]
    fn test_uniform_signal_high_hp_collision() {
        // When harmonic and percussive are identical, HP collision should be ~1.0
        let n_fft = 2048;
        let n_bins = n_fft / 2 + 1;
        let n_frames = 50;
        let mags = vec![vec![0.1_f32; n_bins]; n_frames];
        let spec = make_spectrogram(mags, 44100, n_fft);

        let bands = spectral::frequency_band_energy(&spec);
        let sc = spectral::spectral_contrast(&spec, None);
        // Same spectrogram for both harmonic and percussive = equal energy
        let h_bands = spectral::frequency_band_energy(&spec);
        let p_bands = spectral::frequency_band_energy(&spec);

        let result = detect_masking(&bands, &sc, &h_bands, &p_bands);

        // All HP collision values should be 1.0 (identical energy)
        for frame in &result.analysis.hp_collision {
            for &val in frame {
                assert!(
                    (val - 1.0).abs() < 0.01,
                    "Expected HP collision ~1.0, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_harmonic_dominant_low_hp_collision() {
        // When percussive is much weaker, HP collision should be low
        let n_fft = 2048;
        let n_bins = n_fft / 2 + 1;
        let n_frames = 50;
        let harmonic_mags = vec![vec![0.5_f32; n_bins]; n_frames];
        let percussive_mags = vec![vec![0.001_f32; n_bins]; n_frames];
        let full_mags = vec![vec![0.5_f32; n_bins]; n_frames];

        let full_spec = make_spectrogram(full_mags, 44100, n_fft);
        let h_spec = make_spectrogram(harmonic_mags, 44100, n_fft);
        let p_spec = make_spectrogram(percussive_mags, 44100, n_fft);

        let bands = spectral::frequency_band_energy(&full_spec);
        let sc = spectral::spectral_contrast(&full_spec, None);
        let h_bands = spectral::frequency_band_energy(&h_spec);
        let p_bands = spectral::frequency_band_energy(&p_spec);

        let result = detect_masking(&bands, &sc, &h_bands, &p_bands);

        // HP collision should be low when one component dominates
        for frame in &result.analysis.hp_collision {
            for &val in frame {
                assert!(val < 0.1, "Expected low HP collision, got {}", val);
            }
        }
    }

    #[test]
    fn test_cross_band_correlation_broadband() {
        // White noise (uniform energy) should have high cross-band correlation
        let n_fft = 2048;
        let n_bins = n_fft / 2 + 1;
        let n_frames = 200;

        // All bins equal = same energy in every band = perfect correlation
        let mags = vec![vec![0.1_f32; n_bins]; n_frames];
        let spec = make_spectrogram(mags, 44100, n_fft);
        let bands = spectral::frequency_band_energy(&spec);

        let result = compute_cross_band_correlation(&bands);

        // With constant energy, variance is 0, so correlation is 0/0 → 0.0
        // This is actually correct: constant signals have no covariance pattern
        // Let's verify it doesn't crash and returns valid values
        for &corr in &result.correlations {
            assert!(corr.is_finite(), "Correlation should be finite");
        }
    }

    #[test]
    fn test_cross_band_correlation_independent() {
        // Two bands with opposite energy patterns should have negative correlation
        let n_fft = 2048;
        let n_bins = n_fft / 2 + 1;
        let n_frames = 100;

        // Create frames where bass is loud when mids are quiet and vice versa
        let mut mags = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let mut frame = vec![0.001_f32; n_bins];
            if i % 2 == 0 {
                // Even frames: energy in bass bins (60-250 Hz → bins ~3-12 at 44100/2048)
                for bin in 3..12 {
                    frame[bin] = 0.5;
                }
            } else {
                // Odd frames: energy in mid bins (500-2000 Hz → bins ~23-93)
                for bin in 23..93 {
                    frame[bin] = 0.5;
                }
            }
            mags.push(frame);
        }

        let spec = make_spectrogram(mags, 44100, n_fft);
        let bands = spectral::frequency_band_energy(&spec);
        let result = compute_cross_band_correlation(&bands);

        // bass↔low_mid correlation should be negative (they alternate)
        // Index 1 = (bass, low_mid)
        assert!(
            result.correlations[1] < 0.0,
            "Expected negative correlation between bass and low_mid, got {}",
            result.correlations[1]
        );
    }

    #[test]
    fn test_crowding_values_bounded() {
        // Crowding should always be in [0, 1]
        let n_fft = 2048;
        let n_bins = n_fft / 2 + 1;
        let n_frames = 50;

        // Varying energy to create non-trivial crowding
        let mut mags = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let level = (i as f32 / n_frames as f32) * 0.5;
            mags.push(vec![level; n_bins]);
        }

        let spec = make_spectrogram(mags, 44100, n_fft);
        let bands = spectral::frequency_band_energy(&spec);
        let sc = spectral::spectral_contrast(&spec, None);
        let h_bands = spectral::frequency_band_energy(&spec);
        let p_bands = spectral::frequency_band_energy(&spec);

        let result = detect_masking(&bands, &sc, &h_bands, &p_bands);

        for frame in &result.analysis.crowding {
            for &val in frame {
                assert!(
                    (0.0..=1.0).contains(&val),
                    "Crowding should be in [0,1], got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_format_masking_summary_runs() {
        // Smoke test: verify formatting doesn't panic
        let n_fft = 2048;
        let n_bins = n_fft / 2 + 1;
        let n_frames = 50;
        let mags = vec![vec![0.1_f32; n_bins]; n_frames];
        let spec = make_spectrogram(mags, 44100, n_fft);

        let bands = spectral::frequency_band_energy(&spec);
        let sc = spectral::spectral_contrast(&spec, None);
        let h_bands = spectral::frequency_band_energy(&spec);
        let p_bands = spectral::frequency_band_energy(&spec);

        let result = detect_masking(&bands, &sc, &h_bands, &p_bands);
        let summary = format_masking_summary(&result);

        assert!(summary.contains("Frequency Masking"));
        assert!(summary.contains("Crowding"));
    }
}
