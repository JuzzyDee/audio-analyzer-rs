// analysis/downsample.rs — Time-series downsampling and formatting
//
// When Claude asks for time-series data, we don't want to dump 94 frames
// per second of raw data — that would eat tokens alive. Instead, we
// downsample to a human/AI-readable rate and format as compact TSV.
//
// Three presets calibrated to match what you can extract from a visual
// spectrogram image (~0.5 points/sec), not raw DSP resolution:
//   "low"    ~0.5 data points per second (broad overview, image-equivalent)
//   "medium" ~1 data point per second   (good for most tasks)
//   "high"   ~4 data points per second  (detailed, for zooming in)
//
// The downsampling uses bin-averaging: group N consecutive frames into
// one output point and average them. This preserves the character of
// statistical features (centroid, flatness, etc.) without aliasing.

use std::fmt::Write;

/// Convert a resolution name to a target frames-per-second.
///
/// Accepts "low", "medium", "high", or a numeric string like "20".
pub fn resolution_to_fps(resolution: &str) -> Result<f32, String> {
    match resolution.trim().to_lowercase().as_str() {
        "low" => Ok(0.5),
        "medium" | "med" => Ok(1.0),
        "high" => Ok(4.0),
        other => other
            .parse::<f32>()
            .map_err(|_| format!("Invalid resolution '{}'. Use low, medium, high, or a number.", other))
            .and_then(|v| {
                if v > 0.0 {
                    Ok(v)
                } else {
                    Err("Resolution must be positive.".to_string())
                }
            }),
    }
}

/// Compute the native frames-per-second from sample rate and hop length.
pub fn native_fps(sample_rate: u32, hop_length: usize) -> f32 {
    sample_rate as f32 / hop_length as f32
}

/// Downsample a slice of f32 values by bin-averaging.
///
/// Returns (time_seconds, averaged_value) pairs.
/// Time is the midpoint of each bin.
pub fn downsample_f32(
    data: &[f32],
    native_fps: f32,
    target_fps: f32,
) -> Vec<(f32, f32)> {
    if data.is_empty() {
        return Vec::new();
    }

    // If target fps >= native fps, just return everything with timestamps
    if target_fps >= native_fps {
        return data
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as f32 / native_fps, v))
            .collect();
    }

    let bin_size = native_fps / target_fps;
    let n_bins = (data.len() as f32 / bin_size).ceil() as usize;
    let mut result = Vec::with_capacity(n_bins);

    let mut pos = 0.0_f32;
    while (pos as usize) < data.len() {
        let start = pos as usize;
        let end = ((pos + bin_size) as usize).min(data.len());

        if start >= end {
            break;
        }

        let count = (end - start) as f32;
        let sum: f32 = data[start..end].iter().sum();
        let midpoint = (start as f32 + end as f32) / 2.0 / native_fps;

        result.push((midpoint, sum / count));
        pos += bin_size;
    }

    result
}

/// Downsample a slice of fixed-size arrays by bin-averaging.
///
/// Returns (time_seconds, averaged_array) pairs.
pub fn downsample_array<const N: usize>(
    data: &[[f32; N]],
    native_fps: f32,
    target_fps: f32,
) -> Vec<(f32, [f32; N])> {
    if data.is_empty() {
        return Vec::new();
    }

    if target_fps >= native_fps {
        return data
            .iter()
            .enumerate()
            .map(|(i, arr)| (i as f32 / native_fps, *arr))
            .collect();
    }

    let bin_size = native_fps / target_fps;
    let n_bins = (data.len() as f32 / bin_size).ceil() as usize;
    let mut result = Vec::with_capacity(n_bins);

    let mut pos = 0.0_f32;
    while (pos as usize) < data.len() {
        let start = pos as usize;
        let end = ((pos + bin_size) as usize).min(data.len());

        if start >= end {
            break;
        }

        let count = (end - start) as f32;
        let midpoint = (start as f32 + end as f32) / 2.0 / native_fps;

        let mut avg = [0.0_f32; N];
        for frame in &data[start..end] {
            for (i, &val) in frame.iter().enumerate() {
                avg[i] += val;
            }
        }
        for val in &mut avg {
            *val /= count;
        }

        result.push((midpoint, avg));
        pos += bin_size;
    }

    result
}

// ---- Formatting helpers ----
// These produce compact TSV text that's easy for an LLM to parse.

/// Format multiple parallel f32 time-series as a TSV section.
///
/// `columns` names each series. `series` holds the downsampled data for each.
/// All series must share the same time axis (same length, same timestamps).
pub fn format_f32_series(
    section_name: &str,
    columns: &[&str],
    series: &[&[(f32, f32)]],
) -> String {
    if series.is_empty() || series[0].is_empty() {
        return String::new();
    }

    let n_points = series[0].len();
    // Pre-allocate: rough estimate of ~40 chars per line
    let mut out = String::with_capacity(n_points * 40 + 100);

    // Section header
    writeln!(out, "\n── {} ({} points) ──", section_name, n_points).unwrap();

    // Column header
    out.push_str("# time_s");
    for col in columns {
        out.push('\t');
        out.push_str(col);
    }
    out.push('\n');

    // Data rows
    for i in 0..n_points {
        let time = series[0][i].0;
        write!(out, "{:.2}", time).unwrap();
        for s in series {
            out.push('\t');
            write!(out, "{:.3}", s[i].1).unwrap();
        }
        out.push('\n');
    }

    out
}

/// Format a downsampled array time-series (e.g. chroma [f32; 12]) as TSV.
pub fn format_array_series<const N: usize>(
    section_name: &str,
    columns: &[&str],
    data: &[(f32, [f32; N])],
) -> String {
    if data.is_empty() {
        return String::new();
    }

    let mut out = String::with_capacity(data.len() * (N * 8 + 20) + 100);

    writeln!(out, "\n── {} ({} points) ──", section_name, data.len()).unwrap();

    // Column header
    out.push_str("# time_s");
    for col in columns {
        out.push('\t');
        out.push_str(col);
    }
    out.push('\n');

    // Data rows
    for (time, values) in data {
        write!(out, "{:.2}", time).unwrap();
        for val in values {
            out.push('\t');
            write!(out, "{:.3}", val).unwrap();
        }
        out.push('\n');
    }

    out
}

/// Format a unified time-series table combining all analysis data on a shared time axis.
///
/// Takes multiple f32 series and array series, all pre-downsampled to the same
/// time axis, and produces a single compact TSV table. This avoids repeating the
/// time column for each section, significantly reducing token count.
pub fn format_unified_timeseries(
    n_points: usize,
    // f32 series: (column_names, downsampled_data)
    f32_series: &[(&[&str], &[(f32, f32)])],
    // Array series: (column_names, downsampled_data) — we take raw slices of (time, values)
    chroma_data: Option<(&[(f32, [f32; 12])], &[&str])>,
    tonnetz_data: Option<(&[(f32, [f32; 6])], &[&str])>,
) -> String {
    if n_points == 0 {
        return String::new();
    }

    // Count total columns for capacity estimate
    let total_cols: usize = f32_series.iter().map(|(cols, _)| cols.len()).sum::<usize>()
        + if chroma_data.is_some() { 12 } else { 0 }
        + if tonnetz_data.is_some() { 6 } else { 0 };
    let mut out = String::with_capacity(n_points * (total_cols * 8 + 10) + 200);

    writeln!(out, "\n── Time-Series Data ({} points) ──", n_points).unwrap();

    // Build header
    out.push_str("# time_s");
    for (cols, _) in f32_series {
        for col in *cols {
            out.push('\t');
            out.push_str(col);
        }
    }
    if let Some((_, cols)) = chroma_data {
        for col in cols {
            out.push('\t');
            out.push_str(col);
        }
    }
    if let Some((_, cols)) = tonnetz_data {
        for col in cols {
            out.push('\t');
            out.push_str(col);
        }
    }
    out.push('\n');

    // Build data rows
    for i in 0..n_points {
        // Time from the first f32 series (they all share the same axis)
        let time = f32_series.first().map(|(_, data)| data[i].0).unwrap_or(0.0);
        write!(out, "{:.2}", time).unwrap();

        // f32 columns
        for (_, data) in f32_series {
            out.push('\t');
            write!(out, "{:.3}", data[i].1).unwrap();
        }

        // Chroma columns (12)
        if let Some((data, _)) = chroma_data {
            for val in &data[i].1 {
                out.push('\t');
                write!(out, "{:.3}", val).unwrap();
            }
        }

        // Tonnetz columns (6)
        if let Some((data, _)) = tonnetz_data {
            for val in &data[i].1 {
                out.push('\t');
                write!(out, "{:.3}", val).unwrap();
            }
        }

        out.push('\n');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_to_fps_presets() {
        assert_eq!(resolution_to_fps("low").unwrap(), 0.5);
        assert_eq!(resolution_to_fps("medium").unwrap(), 1.0);
        assert_eq!(resolution_to_fps("med").unwrap(), 1.0);
        assert_eq!(resolution_to_fps("high").unwrap(), 4.0);
        assert_eq!(resolution_to_fps("HIGH").unwrap(), 4.0);
    }

    #[test]
    fn test_resolution_to_fps_numeric() {
        assert_eq!(resolution_to_fps("20").unwrap(), 20.0);
        assert_eq!(resolution_to_fps("0.5").unwrap(), 0.5);
        assert!(resolution_to_fps("0").is_err());
        assert!(resolution_to_fps("-1").is_err());
        assert!(resolution_to_fps("banana").is_err());
    }

    #[test]
    fn test_downsample_f32_empty() {
        let result = downsample_f32(&[], 94.0, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_downsample_f32_passthrough() {
        // target >= native should return all points
        let data = vec![1.0, 2.0, 3.0];
        let result = downsample_f32(&data, 5.0, 10.0);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].1, 1.0);
        assert_eq!(result[1].1, 2.0);
        assert_eq!(result[2].1, 3.0);
    }

    #[test]
    fn test_downsample_f32_averaging() {
        // 10 frames at 10fps, downsampled to 1fps = 1 bin of 10 frames
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let result = downsample_f32(&data, 10.0, 1.0);
        assert_eq!(result.len(), 1);
        // Average of 0..9 = 4.5
        assert!((result[0].1 - 4.5).abs() < 0.01);
        // Midpoint should be at 0.5 seconds
        assert!((result[0].0 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_downsample_f32_multiple_bins() {
        // 20 frames at 10fps = 2 seconds, downsample to 1fps = 2 bins
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let result = downsample_f32(&data, 10.0, 1.0);
        assert_eq!(result.len(), 2);
        // First bin: avg of 0..9 = 4.5 at ~0.5s
        assert!((result[0].1 - 4.5).abs() < 0.01);
        // Second bin: avg of 10..19 = 14.5 at ~1.5s
        assert!((result[1].1 - 14.5).abs() < 0.01);
    }

    #[test]
    fn test_downsample_f32_partial_bin() {
        // 15 frames at 10fps, downsample to 1fps = 1 full bin + 1 partial
        let data: Vec<f32> = vec![1.0; 15];
        let result = downsample_f32(&data, 10.0, 1.0);
        assert_eq!(result.len(), 2);
        assert!((result[0].1 - 1.0).abs() < 0.01);
        assert!((result[1].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_downsample_array() {
        // 10 frames of [f32; 2], downsample 10fps -> 1fps
        let data: Vec<[f32; 2]> = (0..10)
            .map(|i| [i as f32, (i * 2) as f32])
            .collect();
        let result = downsample_array(&data, 10.0, 1.0);
        assert_eq!(result.len(), 1);
        assert!((result[0].1[0] - 4.5).abs() < 0.01);
        assert!((result[0].1[1] - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_format_f32_series() {
        let series = vec![(0.5, 100.0), (1.5, 200.0)];
        let output = format_f32_series("Test", &["value"], &[&series]);
        assert!(output.contains("# time_s\tvalue"));
        assert!(output.contains("0.50\t100.000"));
        assert!(output.contains("1.50\t200.000"));
        assert!(output.contains("2 points"));
    }

    #[test]
    fn test_format_array_series() {
        let data = vec![(0.5, [1.0, 2.0]), (1.5, [3.0, 4.0])];
        let output = format_array_series("Test", &["a", "b"], &data);
        assert!(output.contains("# time_s\ta\tb"));
        assert!(output.contains("0.50\t1.000\t2.000"));
    }

    #[test]
    fn test_format_unified_timeseries() {
        let centroid = vec![(0.5, 100.0), (1.5, 200.0)];
        let flatness = vec![(0.5, 0.05), (1.5, 0.08)];
        let chroma: Vec<(f32, [f32; 12])> = vec![
            (0.5, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (1.5, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        let chroma_cols = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
        let chroma_col_refs: Vec<&str> = chroma_cols.iter().copied().collect();

        let output = format_unified_timeseries(
            2,
            &[(&["centroid"], &centroid), (&["flatness"], &flatness)],
            Some((&chroma, &chroma_col_refs)),
            None,
        );

        // Single header with all columns
        assert!(output.contains("# time_s\tcentroid\tflatness\tC\tC#"));
        // Time appears only once per row
        assert!(output.contains("0.50\t100.000\t0.050\t1.000"));
        // Only 2 data lines
        let data_lines: Vec<&str> = output.lines().filter(|l| !l.starts_with('#') && !l.starts_with('─') && !l.is_empty()).collect();
        assert_eq!(data_lines.len(), 2);
    }
}
