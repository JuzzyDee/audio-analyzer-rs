// main.rs — binary entry point

use std::env;
use std::time::Instant;

use audio_visualizer_rs::load_audio;
use audio_visualizer_rs::analysis::spectral;
use audio_visualizer_rs::analysis::harmonic;
use audio_visualizer_rs::analysis::rhythm;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage: audio_visualizer_rs <audio_file>");
        println!("Example: audio_visualizer_rs song.mp3");
        std::process::exit(1);
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
        &audio.samples,  // & borrows the samples — doesn't copy them
        audio.sample_rate,
        None,  // use default n_fft (2048)
        None,  // use default hop_length (512)
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

    println!("  Spectral Centroid:   avg {:.0} Hz (brightness)", avg(&centroid));
    println!("  Spectral Bandwidth:  avg {:.0} Hz (richness)", avg(&bandwidth));
    println!("  Spectral Rolloff:    avg {:.0} Hz (energy concentration)", avg(&rolloff));
    println!("  Spectral Flatness:   avg {:.4} (0=tonal, 1=noisy)", avg(&flatness));
    println!("  Computed in:         {:.2?}\n", features_time);

    // --- Step 4: Harmonic analysis ---
    println!("Computing harmonic analysis...");
    let start = Instant::now();

    let chromagram = harmonic::compute_chromagram(&spectrogram);
    let tonnetz = harmonic::compute_tonnetz(&chromagram);

    let harmonic_time = start.elapsed();

    // Key estimation
    let (key, mode, confidence) = chromagram.estimate_key();
    println!("  Estimated key:  {} {} (confidence: {:.3})", key, mode, confidence);

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

    let pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    println!("  Pitch class ranking (strongest → weakest):");
    for (idx, (pc, energy)) in ranked.iter().enumerate() {
        // Create a simple bar visualisation
        let bar_len = (energy * 30.0) as usize;
        let bar: String = "█".repeat(bar_len);
        // `String` owns its data. `.repeat(n)` creates a new String with
        // the pattern repeated n times. We're building a crude bar chart.
        println!("    {:>2}. {:<2} {:.3} {}", idx + 1, pitch_names[*pc], energy, bar);
    }

    println!("  Tonnetz frames:  {}", tonnetz.len());
    println!("  Computed in:     {:.2?}\n", harmonic_time);

    // --- Step 5: Rhythm analysis ---
    println!("Computing rhythm analysis...");
    let start = Instant::now();

    let rhythm_analysis = rhythm::analyse_rhythm(&spectrogram, None, None);
    let rhythm_time = start.elapsed();

    println!("  Estimated tempo: {:.1} BPM (confidence: {:.3})",
        rhythm_analysis.tempo_bpm, rhythm_analysis.tempo_confidence);
    println!("  Detected beats:  {}", rhythm_analysis.beat_times.len());

    // Show beat statistics if we have enough beats
    if let Some(stats) = rhythm::beat_statistics(&rhythm_analysis.beat_times) {
        println!("  Mean tempo:      {:.1} BPM (from inter-beat intervals)", stats.mean_bpm);
        println!("  Median tempo:    {:.1} BPM", stats.median_bpm);
        println!("  Tempo stability: {:.3} (0=erratic, 1=metronomic)", stats.tempo_stability);
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

    // --- Summary ---
    let total = decode_time + spec_time + features_time + harmonic_time + rhythm_time;
    println!("=== TOTAL: {:.2?} ===", total);
}
