// stereo.rs — Stereo field analysis
//
// Analyses the relationship between left and right channels:
// - Stereo width (mid/side energy ratio)
// - Phase correlation (mono compatibility)
// - Balance (left/right energy distribution)
//
// All functions operate on raw L/R sample vectors with windowed frames
// matching the spectrogram timing (n_fft=2048, hop=512 by default).

/// Results from stereo field analysis.
#[derive(Debug)]
pub struct StereoAnalysis {
    /// Per-frame phase correlation: +1 = in phase (mono compatible),
    /// 0 = unrelated, -1 = out of phase (cancels in mono).
    pub phase_correlation: Vec<f32>,

    /// Per-frame stereo width: 0 = mono, 1 = full stereo, >1 = out of phase content.
    /// Computed as side_energy / mid_energy ratio.
    pub stereo_width: Vec<f32>,

    /// Per-frame left/right balance: -1 = hard left, 0 = centre, +1 = hard right.
    pub balance: Vec<f32>,

    /// Per-frame mid channel RMS energy.
    pub mid_energy: Vec<f32>,

    /// Per-frame side channel RMS energy.
    pub side_energy: Vec<f32>,

    /// Mono compatibility score per frame: fraction of energy preserved when
    /// summing to mono. 1.0 = no loss, 0.0 = complete cancellation.
    pub mono_compatibility: Vec<f32>,

    /// Number of frames.
    pub n_frames: usize,

    /// Whether the source file was mono (1) or stereo (2+).
    pub source_channels: u32,
}

/// Compute stereo field analysis from left and right channels.
///
/// Uses windowed frames with the given n_fft and hop_length to match
/// the spectrogram timing. For mono source files, all values will be
/// trivial (correlation=1, width=0, balance=0, mono_compat=1).
pub fn analyse_stereo(
    left: &[f32],
    right: &[f32],
    source_channels: u32,
    n_fft: usize,
    hop_length: usize,
) -> StereoAnalysis {
    assert_eq!(left.len(), right.len(), "L/R channels must be same length");

    let n_samples = left.len();
    if n_samples < n_fft {
        return StereoAnalysis {
            phase_correlation: vec![],
            stereo_width: vec![],
            balance: vec![],
            mid_energy: vec![],
            side_energy: vec![],
            mono_compatibility: vec![],
            n_frames: 0,
            source_channels,
        };
    }

    let n_frames = (n_samples - n_fft) / hop_length + 1;

    let mut phase_correlation = Vec::with_capacity(n_frames);
    let mut stereo_width = Vec::with_capacity(n_frames);
    let mut balance = Vec::with_capacity(n_frames);
    let mut mid_energy = Vec::with_capacity(n_frames);
    let mut side_energy = Vec::with_capacity(n_frames);
    let mut mono_compatibility = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        let end = start + n_fft;

        let l = &left[start..end];
        let r = &right[start..end];

        // Mid/side encoding: mid = (L+R)/2, side = (L-R)/2
        // Mid is what you hear in mono, side is the stereo difference.
        let mut sum_ll = 0.0_f64;
        let mut sum_rr = 0.0_f64;
        let mut sum_lr = 0.0_f64;
        let mut sum_mid_sq = 0.0_f64;
        let mut sum_side_sq = 0.0_f64;
        let mut sum_mono_sq = 0.0_f64;

        for i in 0..n_fft {
            let li = l[i] as f64;
            let ri = r[i] as f64;
            let mid = (li + ri) / 2.0;
            let side = (li - ri) / 2.0;

            sum_ll += li * li;
            sum_rr += ri * ri;
            sum_lr += li * ri;
            sum_mid_sq += mid * mid;
            sum_side_sq += side * side;
            sum_mono_sq += (li + ri) * (li + ri); // unnormalised mono power
        }

        // Phase correlation: normalised cross-correlation of L and R
        // r = sum(L*R) / sqrt(sum(L^2) * sum(R^2))
        let denom = (sum_ll * sum_rr).sqrt();
        let corr = if denom > 1e-10 {
            (sum_lr / denom) as f32
        } else {
            1.0 // silence is trivially in phase
        };
        phase_correlation.push(corr.clamp(-1.0, 1.0));

        // Stereo width: side/mid RMS ratio
        let mid_rms = (sum_mid_sq / n_fft as f64).sqrt();
        let side_rms = (sum_side_sq / n_fft as f64).sqrt();
        let width = if mid_rms > 1e-10 {
            (side_rms / mid_rms) as f32
        } else {
            0.0
        };
        stereo_width.push(width);

        mid_energy.push(mid_rms as f32);
        side_energy.push(side_rms as f32);

        // Balance: (R_energy - L_energy) / (R_energy + L_energy)
        let l_rms = sum_ll.sqrt();
        let r_rms = sum_rr.sqrt();
        let bal = if l_rms + r_rms > 1e-10 {
            ((r_rms - l_rms) / (r_rms + l_rms)) as f32
        } else {
            0.0
        };
        balance.push(bal);

        // Mono compatibility: energy preserved when summing to mono
        // = (L+R)^2 power / (2 * (L^2 + R^2) power)
        // If L and R are identical: (2L)^2 / (2 * 2 * L^2) = 4L^2 / 4L^2 = 1.0
        // If L and R cancel: 0 / (2 * 2 * L^2) = 0.0
        let stereo_power = sum_ll + sum_rr;
        let compat = if stereo_power > 1e-10 {
            // sum_mono_sq is sum((L+R)^2), stereo_power is sum(L^2 + R^2)
            // For perfectly correlated signals: sum((L+R)^2) = 4*sum(L^2) and
            // stereo_power = 2*sum(L^2), so ratio = 4/2 = 2. Normalise by /2.
            ((sum_mono_sq / (2.0 * stereo_power)) as f32).min(1.0)
        } else {
            1.0 // silence is trivially compatible
        };
        mono_compatibility.push(compat);
    }

    StereoAnalysis {
        phase_correlation,
        stereo_width,
        balance,
        mid_energy,
        side_energy,
        mono_compatibility,
        n_frames,
        source_channels,
    }
}

/// Summary statistics for stereo analysis.
#[derive(Debug)]
pub struct StereoSummary {
    pub avg_phase_correlation: f32,
    pub min_phase_correlation: f32,
    pub avg_stereo_width: f32,
    pub max_stereo_width: f32,
    pub avg_balance: f32,
    pub avg_mono_compatibility: f32,
    pub min_mono_compatibility: f32,
    /// Fraction of frames where phase correlation drops below 0 (phase issues).
    pub phase_warning_fraction: f32,
}

/// Compute summary statistics from a stereo analysis.
pub fn stereo_summary(analysis: &StereoAnalysis) -> StereoSummary {
    if analysis.n_frames == 0 {
        return StereoSummary {
            avg_phase_correlation: 1.0,
            min_phase_correlation: 1.0,
            avg_stereo_width: 0.0,
            max_stereo_width: 0.0,
            avg_balance: 0.0,
            avg_mono_compatibility: 1.0,
            min_mono_compatibility: 1.0,
            phase_warning_fraction: 0.0,
        };
    }

    let n = analysis.n_frames as f32;
    let avg = |v: &[f32]| v.iter().sum::<f32>() / n;
    let min = |v: &[f32]| v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = |v: &[f32]| v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let phase_warnings = analysis.phase_correlation.iter()
        .filter(|&&c| c < 0.0)
        .count();

    StereoSummary {
        avg_phase_correlation: avg(&analysis.phase_correlation),
        min_phase_correlation: min(&analysis.phase_correlation),
        avg_stereo_width: avg(&analysis.stereo_width),
        max_stereo_width: max(&analysis.stereo_width),
        avg_balance: avg(&analysis.balance),
        avg_mono_compatibility: avg(&analysis.mono_compatibility),
        min_mono_compatibility: min(&analysis.mono_compatibility),
        phase_warning_fraction: phase_warnings as f32 / n,
    }
}

/// Format stereo analysis as a human-readable summary string.
pub fn format_stereo_summary(summary: &StereoSummary, source_channels: u32) -> String {
    let mut out = String::new();

    if source_channels < 2 {
        out.push_str("Source: mono file (stereo analysis not applicable)\n");
        return out;
    }

    // Phase correlation
    let corr_desc = if summary.avg_phase_correlation > 0.9 {
        "excellent mono compatibility"
    } else if summary.avg_phase_correlation > 0.5 {
        "good mono compatibility"
    } else if summary.avg_phase_correlation > 0.0 {
        "some phase issues"
    } else {
        "severe phase problems"
    };
    out.push_str(&format!(
        "Phase correlation:   {:.3} avg, {:.3} min — {}\n",
        summary.avg_phase_correlation, summary.min_phase_correlation, corr_desc
    ));

    if summary.phase_warning_fraction > 0.0 {
        out.push_str(&format!(
            "Phase warnings:      {:.1}% of frames have negative correlation\n",
            summary.phase_warning_fraction * 100.0
        ));
    }

    // Stereo width
    let width_desc = if summary.avg_stereo_width < 0.1 {
        "nearly mono"
    } else if summary.avg_stereo_width < 0.3 {
        "narrow"
    } else if summary.avg_stereo_width < 0.6 {
        "moderate"
    } else if summary.avg_stereo_width < 1.0 {
        "wide"
    } else {
        "very wide (possible phase issues)"
    };
    out.push_str(&format!(
        "Stereo width:        {:.3} avg, {:.3} max — {}\n",
        summary.avg_stereo_width, summary.max_stereo_width, width_desc
    ));

    // Balance
    let bal_desc = if summary.avg_balance.abs() < 0.02 {
        "centred"
    } else if summary.avg_balance.abs() < 0.1 {
        if summary.avg_balance > 0.0 { "slightly right" } else { "slightly left" }
    } else {
        if summary.avg_balance > 0.0 { "right-heavy" } else { "left-heavy" }
    };
    out.push_str(&format!(
        "Balance:             {:.3} — {}\n",
        summary.avg_balance, bal_desc
    ));

    // Mono compatibility
    let compat_desc = if summary.avg_mono_compatibility > 0.9 {
        "minimal loss in mono"
    } else if summary.avg_mono_compatibility > 0.7 {
        "some loss in mono"
    } else if summary.avg_mono_compatibility > 0.5 {
        "significant mono loss"
    } else {
        "severe mono cancellation"
    };
    out.push_str(&format!(
        "Mono compatibility:  {:.3} avg, {:.3} min — {}\n",
        summary.avg_mono_compatibility, summary.min_mono_compatibility, compat_desc
    ));

    out
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_channels_perfect_mono() {
        // Identical L/R should give correlation=1, width=0, compat=1
        let samples: Vec<f32> = (0..4096)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        let result = analyse_stereo(&samples, &samples, 2, 2048, 512);
        let summary = stereo_summary(&result);

        assert!(summary.avg_phase_correlation > 0.99, "correlation should be ~1.0, got {}", summary.avg_phase_correlation);
        assert!(summary.avg_stereo_width < 0.01, "width should be ~0, got {}", summary.avg_stereo_width);
        assert!(summary.avg_mono_compatibility > 0.99, "mono compat should be ~1.0, got {}", summary.avg_mono_compatibility);
        assert!((summary.avg_balance).abs() < 0.01, "balance should be ~0, got {}", summary.avg_balance);
    }

    #[test]
    fn test_inverted_channels_cancel() {
        // L = -R should give correlation=-1, compat=0
        let left: Vec<f32> = (0..4096)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let right: Vec<f32> = left.iter().map(|s| -s).collect();

        let result = analyse_stereo(&left, &right, 2, 2048, 512);
        let summary = stereo_summary(&result);

        assert!(summary.avg_phase_correlation < -0.99, "correlation should be ~-1, got {}", summary.avg_phase_correlation);
        assert!(summary.avg_mono_compatibility < 0.01, "mono compat should be ~0, got {}", summary.avg_mono_compatibility);
    }

    #[test]
    fn test_uncorrelated_channels() {
        // Different frequencies should give low correlation
        let left: Vec<f32> = (0..4096)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let right: Vec<f32> = (0..4096)
            .map(|i| (i as f32 * 0.1537).sin()) // different frequency
            .collect();

        let result = analyse_stereo(&left, &right, 2, 2048, 512);
        let summary = stereo_summary(&result);

        // Should be somewhere between -1 and 1, but not near the extremes
        assert!(summary.avg_phase_correlation.abs() < 0.5,
            "correlation should be low for uncorrelated signals, got {}", summary.avg_phase_correlation);
        assert!(summary.avg_stereo_width > 0.3,
            "width should be moderate+ for uncorrelated signals, got {}", summary.avg_stereo_width);
    }

    #[test]
    fn test_left_only() {
        // Signal only in left channel — balance should be negative (left-heavy)
        let left: Vec<f32> = (0..4096)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let right = vec![0.0_f32; 4096];

        let result = analyse_stereo(&left, &right, 2, 2048, 512);
        let summary = stereo_summary(&result);

        assert!(summary.avg_balance < -0.8, "should be left-heavy, got {}", summary.avg_balance);
    }

    #[test]
    fn test_right_only() {
        // Signal only in right channel — balance should be positive (right-heavy)
        let left = vec![0.0_f32; 4096];
        let right: Vec<f32> = (0..4096)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        let result = analyse_stereo(&left, &right, 2, 2048, 512);
        let summary = stereo_summary(&result);

        assert!(summary.avg_balance > 0.8, "should be right-heavy, got {}", summary.avg_balance);
    }

    #[test]
    fn test_mono_source_trivial() {
        // Mono source file — identical channels, should be trivially perfect
        let samples: Vec<f32> = (0..4096)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        let result = analyse_stereo(&samples, &samples, 1, 2048, 512);
        assert_eq!(result.source_channels, 1);

        let summary = stereo_summary(&result);
        assert!(summary.avg_phase_correlation > 0.99);
        assert!(summary.avg_stereo_width < 0.01);
    }

    #[test]
    fn test_frame_count() {
        let samples = vec![0.0_f32; 8192];
        let result = analyse_stereo(&samples, &samples, 2, 2048, 512);
        // (8192 - 2048) / 512 + 1 = 6144 / 512 + 1 = 12 + 1 = 13
        assert_eq!(result.n_frames, 13);
    }

    #[test]
    fn test_silence() {
        let samples = vec![0.0_f32; 4096];
        let result = analyse_stereo(&samples, &samples, 2, 2048, 512);
        let summary = stereo_summary(&result);

        // Silence should be trivially in phase and compatible
        assert_eq!(summary.avg_phase_correlation, 1.0);
        assert_eq!(summary.avg_mono_compatibility, 1.0);
        assert_eq!(summary.phase_warning_fraction, 0.0);
    }
}
