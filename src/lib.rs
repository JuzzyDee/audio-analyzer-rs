// lib.rs — library root
//
// Rust's module system: each file is a module. lib.rs is the root.
// We declare submodules here and re-export their public items.
// For now everything lives in this file; we'll split it up as it grows.

// DSP code often indexes into multiple parallel arrays at the same position.
// Iterator-based alternatives obscure the intent. Allow indexed loops crate-wide.
#![allow(clippy::needless_range_loop)]
// Generic type signatures for downsampled time-series are inherently nested.
#![allow(clippy::type_complexity)]

// ---- Modules ----
// `pub mod` declares a public submodule. Rust looks for either:
//   - src/analysis.rs (single file module), or
//   - src/analysis/mod.rs (directory module with sub-files)
// We use the directory approach since analysis will have multiple files.
pub mod analysis;

// ---- Imports ----
// `use` brings items into scope. The `::` is Rust's path separator (like `.` in Python).
// Symphonia is organised into sub-crates (core, default).

use std::fs::File; // File I/O from the standard library
use symphonia::core::audio::SampleBuffer; // Buffer to hold decoded audio samples
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream; // Wraps a file for streaming reads
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint; // Tells Symphonia what format to expect

// `pub struct` defines a public data structure — like a Python dataclass.
// In Rust, data and behaviour are separate: you define the struct, then
// attach methods via `impl` blocks. This is different from Python/Swift
// where classes bundle data + methods together.
//
// `#[derive(Debug)]` auto-generates a debug representation (like __repr__).
// Rust doesn't give you this for free — you opt in via "derive macros."
#[derive(Debug)]
pub struct AudioData {
    /// Raw audio samples, mono, normalised to -1.0..1.0
    pub samples: Vec<f32>,
    /// Sample rate in Hz (e.g., 44100)
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f64,
}

#[derive(Debug)]
pub struct StereoAudioData {
    /// Left channel samples, normalised to -1.0..1.0
    pub left: Vec<f32>,
    /// Right channel samples, normalised to -1.0..1.0
    pub right: Vec<f32>,
    /// Mono mixdown (L+R)/2 — same as what load_audio() produces
    pub mono: Vec<f32>,
    /// Number of channels in the original file (1 = mono source, 2 = stereo)
    pub channels: u32,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f64,
}

// `impl` block: attach methods to a struct. Like defining methods inside
// a class in Python, but the struct definition and its methods are separate.
impl AudioData {
    /// Number of samples in the audio data.
    pub fn len(&self) -> usize {
        // `&self` is like Python's `self`, but the `&` means we're *borrowing*
        // the struct, not taking ownership of it. This is Rust's big idea:
        //
        // OWNERSHIP: every value has exactly one owner. When the owner goes
        // out of scope, the value is dropped (freed). No garbage collector needed.
        //
        // BORROWING: `&self` lets you read the data without taking ownership.
        // `&mut self` lets you read AND modify. The compiler enforces that
        // you can have either ONE mutable reference OR any number of immutable
        // references at a time — never both. This prevents data races at
        // compile time, which is why Rust is called "fearless concurrency."
        //
        // You'll develop intuition for this as we go. For now: & = borrow, no & = move.
        self.samples.len()
    }

    /// Whether the audio data is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Load an audio file from disk, decode it to mono f32 samples.
///
/// # Arguments
/// * `path` - Path to the audio file (mp3, wav, flac, ogg, aac)
///
/// # Returns
/// `Result<AudioData, String>` — either the decoded audio or an error message.
///
/// In Rust, there's no exceptions / try-except. Instead, functions that can
/// fail return `Result<OkType, ErrType>`. The caller MUST handle both cases.
/// The compiler won't let you ignore an error — if you try, it warns you.
pub fn load_audio(path: &str) -> Result<AudioData, String> {
    // `let` declares a variable. Variables are immutable by default!
    // Use `let mut` to make them mutable. This is the opposite of Python
    // where everything is mutable. Immutable-by-default prevents accidental
    // mutation and makes code easier to reason about.

    // Open the file. `map_err` converts the IO error into our String error type.
    // This pattern — calling a function that returns Result, then transforming
    // the error — is extremely common in Rust.
    let file = File::open(path).map_err(|e| format!("Cannot open file '{}': {}", path, e))?;
    //                                                                 ^
    // The `?` operator: Rust's equivalent of "if this failed, return the error
    // immediately." It's syntactic sugar for a match statement that either
    // unwraps Ok(value) or returns Err(e) from the current function.
    // You'll see `?` everywhere — it keeps error handling concise.

    // Create a media source stream from the file.
    // MediaSourceStream wraps our File for efficient buffered reading.
    // `Box::new(file)` puts the file on the heap (like Python does by default).
    // In Rust, values live on the stack by default; Box moves them to the heap.
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Give Symphonia a hint about the file format based on extension.
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path).extension()
        && let Some(ext_str) = ext.to_str()
    {
        hint.with_extension(ext_str);
    }

    // Probe the file to detect the format and find audio tracks.
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("Failed to probe audio format: {}", e))?;

    let mut format = probed.format;

    // Get the default (first) audio track.
    // `.ok_or(...)` converts an Option (Rust's Maybe/Optional type) into a Result.
    // Option<T> is either Some(value) or None — Rust has no null/nil.
    // This eliminates null pointer exceptions entirely.
    let track = format
        .default_track()
        .ok_or("No audio tracks found in file")?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("Unknown sample rate")?;

    let codec_params = track.codec_params.clone();

    // Create a decoder for this track's codec.
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Failed to create decoder: {}", e))?;

    // Decode all packets and collect samples.
    // `Vec::new()` creates an empty growable array (like Python's []).
    let mut all_samples: Vec<f32> = Vec::new();

    // Decode packets until end-of-stream.
    while let Ok(packet) = format.next_packet() {
        // Decode the packet into audio samples.
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(_) => continue, // Skip bad packets
        };

        // Get the audio spec (channels, sample rate, etc.)
        let spec = *decoded.spec();
        let num_channels = spec.channels.count();

        // Create a sample buffer and copy decoded audio into it.
        let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        // Convert to mono by averaging channels.
        // If stereo (2 channels), samples are interleaved: [L, R, L, R, ...]
        // We average each pair to get mono: [(L+R)/2, (L+R)/2, ...]
        if num_channels == 1 {
            all_samples.extend_from_slice(samples);
        } else {
            // `.chunks(num_channels)` splits the slice into groups of N.
            // `.map(|chunk| ...)` transforms each group — like Python's map().
            // `.sum::<f32>()` sums the chunk — the `::<f32>` is a "turbofish"
            // telling Rust what type to use for the sum (it can't always infer).
            let mono: Vec<f32> = samples
                .chunks(num_channels)
                .map(|chunk| chunk.iter().sum::<f32>() / num_channels as f32)
                .collect(); // .collect() turns an iterator into a Vec
            all_samples.extend(mono);
        }
    }

    if all_samples.is_empty() {
        return Err("No audio samples decoded".to_string());
        // `.to_string()` converts a string literal (&str) into an owned String.
        // &str = borrowed, fixed-size string slice (like a view)
        // String = owned, growable string (like Python's str)
        // You'll get used to this distinction — it's the ownership system at work.
    }

    let duration = all_samples.len() as f64 / sample_rate as f64;

    // Build and return the AudioData struct.
    // No `return` keyword needed — the last expression in a function is
    // implicitly returned (no semicolon!). This is like Ruby/Kotlin.
    Ok(AudioData {
        samples: all_samples,
        sample_rate, // shorthand: field name matches variable name
        duration,
    })
}

/// Load an audio file preserving stereo channels.
///
/// Returns left, right, and mono channels. For mono source files,
/// left and right will be identical copies of the single channel.
pub fn load_audio_stereo(path: &str) -> Result<StereoAudioData, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open file '{}': {}", path, e))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path).extension()
        && let Some(ext_str) = ext.to_str()
    {
        hint.with_extension(ext_str);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("Failed to probe audio format: {}", e))?;

    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or("No audio tracks found in file")?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("Unknown sample rate")?;

    let codec_params = track.codec_params.clone();

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Failed to create decoder: {}", e))?;

    let mut left_samples: Vec<f32> = Vec::new();
    let mut right_samples: Vec<f32> = Vec::new();
    let mut file_channels: u32 = 1;

    while let Ok(packet) = format.next_packet() {
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(_) => continue,
        };

        let spec = *decoded.spec();
        let num_channels = spec.channels.count();
        file_channels = num_channels as u32;

        let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        if num_channels == 1 {
            left_samples.extend_from_slice(samples);
            right_samples.extend_from_slice(samples);
        } else {
            // Interleaved: [L, R, L, R, ...] (or more channels — we take first two)
            for chunk in samples.chunks(num_channels) {
                left_samples.push(chunk[0]);
                right_samples.push(if chunk.len() > 1 { chunk[1] } else { chunk[0] });
            }
        }
    }

    if left_samples.is_empty() {
        return Err("No audio samples decoded".to_string());
    }

    // Build mono mixdown
    let mono: Vec<f32> = left_samples
        .iter()
        .zip(right_samples.iter())
        .map(|(l, r)| (l + r) / 2.0)
        .collect();

    let duration = mono.len() as f64 / sample_rate as f64;

    Ok(StereoAudioData {
        left: left_samples,
        right: right_samples,
        mono,
        channels: file_channels,
        sample_rate,
        duration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_nonexistent_file() {
        let result = load_audio("nonexistent.mp3");
        assert!(result.is_err());
    }

    #[test]
    fn test_audio_data_len() {
        // We can construct AudioData directly for unit testing.
        let audio = AudioData {
            samples: vec![0.0, 0.1, 0.2, 0.3], // vec![] is a macro to create a Vec
            sample_rate: 44100,
            duration: 4.0 / 44100.0,
        };
        assert_eq!(audio.len(), 4);
        assert!(!audio.is_empty());
    }
}
