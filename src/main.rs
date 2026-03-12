// main.rs — binary entry point

use std::env;
use std::time::Instant;

use audio_visualizer_rs::analysis::compare;
use audio_visualizer_rs::analysis::harmonic;
use audio_visualizer_rs::analysis::percussive;
use audio_visualizer_rs::analysis::rhythm;
use audio_visualizer_rs::analysis::sections;
use audio_visualizer_rs::analysis::spectral;
use audio_visualizer_rs::analysis::stereo;
use audio_visualizer_rs::analysis::temporal;
use audio_visualizer_rs::{load_audio, load_audio_stereo};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage: cli <audio_file>");
        println!("       cli compare <file_a> <file_b>");
        std::process::exit(1);
    }

    // A/B comparison mode
    if args[1] == "compare" {
        if args.len() < 4 {
            println!("Usage: cli compare <file_a> <file_b>");
            std::process::exit(1);
        }
        println!("{}", compare::compare_tracks(&args[2], &args[3]));
        return;
    }

    let path = &args[1];

    println!("Audio Visualizer (Rust)");
    println!("=======================\n");

    // --- Step 1: Decode audio ---
    let start = Instant::now();
    let audio = match load_audio(path) {
        Ok(audio) => audio,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    let decode_time = start.elapsed();

    println!("File:        {}", path);
    println!("Sample rate: {} Hz", audio.sample_rate);
    println!("Samples:     {}", audio.len());
    println!("Duration:    {:.2} seconds", audio.duration);
    println!("Decoded in:  {:.2?}\n", decode_time);

    // --- Step 2: Compute spectrogram ---
    println!("Computing spectrogram...");
    let start = Instant::now();

    // In Rust, we can shadow variables with a new `let` — `start` here
    // is a new variable that shadows the previous one. The old value is
    // dropped. This is idiomatic for "reset the timer" patterns.

    let spectrogram = spectral::compute_spectrogram(
        &audio.samples, // & borrows the samples — doesn't copy them
        audio.sample_rate,
        None, // use default n_fft (2048)
        None, // use default hop_length (512)
    );
    let spec_time = start.elapsed();

    println!("  Frames:     {}", spectrogram.n_frames);
    println!("  Freq bins:  {}", spectrogram.n_freq_bins);
    println!("  Duration:   {:.2} sec", spectrogram.duration());
    println!("  Computed in: {:.2?}\n", spec_time);

    // --- Step 3: Compute spectral features ---
    println!("Computing spectral features...");
    let start = Instant::now();

    let centroid = spectral::spectral_centroid(&spectrogram);
    let bandwidth = spectral::spectral_bandwidth(&spectrogram);
    let rolloff = spectral::spectral_rolloff(&spectrogram, None);
    let flatness = spectral::spectral_flatness(&spectrogram);

    let features_time = start.elapsed();

    // Show a summary of the features (average values across the track).
    // In a real MCP tool, we'd return this data directly to the AI.
    let avg = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    // This is a closure — Rust's lambda/anonymous function.
    // `|args| body` is like Python's `lambda args: body` but more powerful
    // (closures can capture variables from their environment and span
    // multiple lines).

    println!(
        "  Spectral Centroid:   avg {:.0} Hz (brightness)",
        avg(&centroid)
    );
    println!(
        "  Spectral Bandwidth:  avg {:.0} Hz (richness)",
        avg(&bandwidth)
    );
    println!(
        "  Spectral Rolloff:    avg {:.0} Hz (energy concentration)",
        avg(&rolloff)
    );
    println!(
        "  Spectral Flatness:   avg {:.4} (0=tonal, 1=noisy)",
        avg(&flatness)
    );

    let bands = spectral::frequency_band_energy(&spectrogram);
    let band_names = spectral::FREQUENCY_BANDS;
    let mut avg_bands = [0.0_f32; 7];
    for frame in &bands.band_energies {
        for (i, &val) in frame.iter().enumerate() {
            avg_bands[i] += val;
        }
    }
    for val in &mut avg_bands {
        *val /= bands.n_frames as f32;
    }
    println!("  Frequency Band Energy:");
    for (i, &(name, lo, hi)) in band_names.iter().enumerate() {
        let bar_len = (avg_bands[i] * 500.0) as usize;
        let bar: String = "█".repeat(bar_len.min(40));
        println!(
            "    {:<12} ({:>5.0}–{:>5.0} Hz): {:.6} {}",
            name, lo, hi, avg_bands[i], bar
        );
    }

    let sc = spectral::spectral_contrast(&spectrogram, None);
    let mut avg_contrast = [0.0_f32; 7];
    for frame in &sc.contrast {
        for (i, &val) in frame.iter().enumerate() {
            avg_contrast[i] += val;
        }
    }
    for val in &mut avg_contrast {
        *val /= sc.n_frames as f32;
    }
    println!("  Spectral Contrast (peak–valley):");
    for (i, &(name, lo, hi)) in band_names.iter().enumerate() {
        let bar_len = (avg_contrast[i] / 2.0) as usize;
        let bar: String = "█".repeat(bar_len.min(40));
        println!(
            "    {:<12} ({:>5.0}–{:>5.0} Hz): {:>5.1} dB {}",
            name, lo, hi, avg_contrast[i], bar
        );
    }

    println!("  Computed in:         {:.2?}\n", features_time);

    // --- Step 4: Harmonic analysis ---
    println!("Computing harmonic analysis...");
    let start = Instant::now();

    let chromagram = harmonic::compute_chromagram(&audio.samples, audio.sample_rate, &spectrogram);
    let tonnetz = harmonic::compute_tonnetz(&chromagram);

    let harmonic_time = start.elapsed();

    // Key estimation
    let (key, mode, confidence) = chromagram.estimate_key();
    println!(
        "  Estimated key:  {} {} (confidence: {:.3})",
        key, mode, confidence
    );

    // Show the average chroma profile — which notes dominate across the track
    let mut avg_chroma = [0.0_f32; 12];
    for frame in &chromagram.chroma {
        for (i, &val) in frame.iter().enumerate() {
            avg_chroma[i] += val;
        }
    }
    let n = chromagram.n_frames as f32;
    for val in &mut avg_chroma {
        *val /= n;
    }

    // Sort pitch classes by energy to show which notes are most prominent
    let mut ranked: Vec<(usize, f32)> = avg_chroma.iter().copied().enumerate().collect();
    // `.copied()` converts &f32 to f32 — needed because we want to own the values,
    // not borrow them. (Similar to .cloned() but for Copy types like f32.)
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // sort descending

    let pitch_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    println!("  Pitch class ranking (strongest → weakest):");
    for (idx, (pc, energy)) in ranked.iter().enumerate() {
        // Create a simple bar visualisation
        let bar_len = (energy * 30.0) as usize;
        let bar: String = "█".repeat(bar_len);
        // `String` owns its data. `.repeat(n)` creates a new String with
        // the pattern repeated n times. We're building a crude bar chart.
        println!(
            "    {:>2}. {:<2} {:.3} {}",
            idx + 1,
            pitch_names[*pc],
            energy,
            bar
        );
    }

    println!("  Tonnetz frames:  {}", tonnetz.len());
    println!("  Computed in:     {:.2?}\n", harmonic_time);

    // --- Step 5: Rhythm analysis ---
    println!("Computing rhythm analysis...");
    let start = Instant::now();

    let rhythm_analysis = rhythm::analyse_rhythm(&spectrogram, None, None);
    let rhythm_time = start.elapsed();

    println!(
        "  Estimated tempo: {:.1} BPM (confidence: {:.3})",
        rhythm_analysis.tempo_bpm, rhythm_analysis.tempo_confidence
    );
    println!("  Detected beats:  {}", rhythm_analysis.beat_times.len());

    // Show beat statistics if we have enough beats
    if let Some(stats) = rhythm::beat_statistics(&rhythm_analysis.beat_times) {
        println!(
            "  Mean tempo:      {:.1} BPM (from inter-beat intervals)",
            stats.mean_bpm
        );
        println!("  Median tempo:    {:.1} BPM", stats.median_bpm);
        println!(
            "  Tempo stability: {:.3} (0=erratic, 1=metronomic)",
            stats.tempo_stability
        );
        println!("  IBI std dev:     {:.3} sec", stats.std_dev_ibi);
    }

    // Show the first few beat times so you can sanity-check against the track
    if !rhythm_analysis.beat_times.is_empty() {
        let show_count = rhythm_analysis.beat_times.len().min(10);
        let first_beats: Vec<String> = rhythm_analysis.beat_times[..show_count]
            .iter()
            .map(|t| format!("{:.2}s", t))
            .collect();
        println!("  First beats:     {}", first_beats.join(", "));
    }

    println!("  Computed in:     {:.2?}\n", rhythm_time);

    // --- Step 6: Percussive analysis (HPSS) ---
    println!("Computing percussive analysis (HPSS)...");
    let start = Instant::now();

    let hpss_result = percussive::hpss(&spectrogram, None);
    let perc_feats =
        percussive::percussive_features(&hpss_result, audio.sample_rate, spectrogram.hop_length);
    let perc_time = start.elapsed();

    let avg_perc_ratio = avg(&perc_feats.percussive_ratio);
    let avg_density = avg(&perc_feats.onset_density);
    let max_sharpness = perc_feats
        .attack_sharpness
        .iter()
        .cloned()
        .fold(0.0_f32, f32::max);

    println!(
        "  Percussive ratio:  {:.3} (0=harmonic, 1=percussive)",
        avg_perc_ratio
    );
    println!("  Onset density:     {:.1}/sec", avg_density);
    println!("  Peak attack sharp: {:.3}", max_sharpness);
    println!("  Computed in:       {:.2?}\n", perc_time);

    // --- Step 7: Dynamic range ---
    println!("Computing dynamic range...");
    let start = Instant::now();

    let dr = temporal::dynamic_range(&audio.samples, spectrogram.n_fft, spectrogram.hop_length);
    let dr_time = start.elapsed();

    println!("  Peak:            {:.2} dBFS", dr.peak_dbfs);
    println!(
        "  Crest factor:    {:.1} dB — {}",
        dr.overall_crest_db,
        if dr.overall_crest_db > 12.0 {
            "very dynamic"
        } else if dr.overall_crest_db > 6.0 {
            "healthy dynamics"
        } else {
            "compressed"
        }
    );
    println!(
        "  Loudness range:  {:.1} dB (95th–5th percentile) — {}",
        dr.loudness_range_db,
        if dr.loudness_range_db > 12.0 {
            "very dynamic"
        } else if dr.loudness_range_db > 6.0 {
            "moderate"
        } else if dr.loudness_range_db > 3.0 {
            "compressed"
        } else {
            "brick-walled"
        }
    );
    println!(
        "  Quiet sections:  {:.1} dBFS (5th percentile)",
        dr.rms_5th_db
    );
    println!(
        "  Loud sections:   {:.1} dBFS (95th percentile)",
        dr.rms_95th_db
    );
    println!("  Computed in:     {:.2?}\n", dr_time);

    // --- Step 8: Stereo load (needed for correct LUFS and stereo analysis) ---
    println!("Loading stereo channels...");
    let start = Instant::now();

    let stereo_audio = match load_audio_stereo(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  Stereo load failed: {}", e);
            std::process::exit(1);
        }
    };
    let stereo_load_time = start.elapsed();
    println!("  Loaded in:       {:.2?}\n", stereo_load_time);

    // --- Step 9: LUFS loudness (EBU R128) ---
    println!("Computing LUFS loudness (EBU R128)...");
    let start = Instant::now();

    // Per ITU-R BS.1770-4: K-weight each channel independently, sum channel powers
    let lufs =
        temporal::measure_lufs_stereo(&stereo_audio.left, &stereo_audio.right, audio.sample_rate);
    let lufs_time = start.elapsed();

    println!("  Integrated:      {:.1} LUFS", lufs.integrated);
    println!("  True peak:       {:.1} dBTP", lufs.true_peak_dbtp);
    println!("  Loudness range:  {:.1} LU", lufs.loudness_range);

    // Platform comparison
    println!("  Platform targets:");
    for &(name, target) in &[
        ("Spotify", -14.0_f32),
        ("Apple Music", -16.0),
        ("YouTube", -14.0),
    ] {
        let diff = lufs.integrated - target;
        if diff.abs() < 0.5 {
            println!("    {:<13} {:.0} LUFS → on target", name, target);
        } else if diff > 0.0 {
            println!(
                "    {:<13} {:.0} LUFS → turned DOWN {:.1} dB",
                name, target, diff
            );
        } else {
            println!(
                "    {:<13} {:.0} LUFS → turned UP {:.1} dB (will sound quieter)",
                name, target, -diff
            );
        }
    }

    if !lufs.short_term.is_empty() {
        let min_st = lufs
            .short_term
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let max_st = lufs
            .short_term
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        println!("  Short-term:      {:.1} to {:.1} LUFS", min_st, max_st);
    }

    println!("  Computed in:     {:.2?}\n", lufs_time);

    // --- Step 10: Stereo analysis ---
    println!("Computing stereo analysis...");
    let start = Instant::now();

    let stereo_result = stereo::analyse_stereo(
        &stereo_audio.left,
        &stereo_audio.right,
        stereo_audio.channels,
        spectrogram.n_fft,
        spectrogram.hop_length,
    );
    let stereo_sum = stereo::stereo_summary(&stereo_result);
    let stereo_time = start.elapsed();

    print!(
        "{}",
        stereo::format_stereo_summary(&stereo_sum, stereo_audio.channels)
    );
    println!("  Computed in:     {:.2?}\n", stereo_time);

    // --- Step 11: Section boundary detection ---
    println!("Detecting section boundaries...");
    let start = Instant::now();

    let section_result = sections::detect_sections(&spectrogram, Some(&chromagram), None);
    let section_time = start.elapsed();

    print!("{}", sections::format_sections(&section_result));
    println!("  Computed in:     {:.2?}\n", section_time);

    // --- Summary ---
    let total = decode_time
        + spec_time
        + features_time
        + harmonic_time
        + rhythm_time
        + perc_time
        + dr_time
        + stereo_load_time
        + lufs_time
        + stereo_time
        + section_time;
    println!("=== TOTAL: {:.2?} ===", total);
}
