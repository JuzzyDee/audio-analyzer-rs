# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rust-based audio analysis toolkit that decodes audio files (mp3, wav, flac, ogg, aac) and extracts spectral, harmonic, and rhythm features. Exposes analysis as both a CLI tool and an MCP (Model Context Protocol) server for use with Claude Code.

Uses Rust 2024 edition. No external linter or formatter config — use standard `cargo fmt` and `cargo clippy`.

## Build & Test Commands

```bash
cargo build                    # debug build
cargo build --release          # release build
cargo test                     # run all tests
cargo test spectral            # run tests in a specific module (by name filter)
cargo test -- --nocapture      # run tests with println! output visible
cargo run --bin cli -- <file>  # run CLI on an audio file
cargo run --bin mcp-server     # run the MCP server (stdio JSON-RPC)
```

## Architecture

Two binaries defined in `Cargo.toml`:
- **`cli`** (`src/main.rs`) — CLI that runs all analyses on an audio file and prints results
- **`mcp-server`** (`src/mcp_server.rs`) — MCP server exposing 5 tools: `audio_info`, `spectral_features`, `harmonic_analysis`, `rhythm_analysis`, `full_analysis`

Library code (`src/lib.rs`):
- `AudioData` struct holds decoded mono f32 samples + sample rate + duration
- `load_audio()` decodes any supported format to mono f32 via Symphonia

Analysis modules (`src/analysis/`):
- **`spectral.rs`** — STFT/spectrogram computation, spectral centroid/bandwidth/rolloff/flatness, mel filterbank, MFCCs. `Spectrogram` struct is the foundation for all other analysis.
- **`harmonic.rs`** — Chromagram (pitch class energy over time), key estimation (Krumhansl-Schmuckler), Tonnetz (tonal centroid features). Depends on `Spectrogram`.
- **`rhythm.rs`** — Onset detection (spectral flux), tempo estimation (autocorrelation), beat tracking (peak picking), beat statistics. Depends on `Spectrogram`.
- **`temporal.rs`** — RMS energy (loudness) and zero crossing rate (texture). Operates on raw audio samples with windowed frames matching spectrogram timing.
- **`downsample.rs`** — Time-series downsampling (bin-averaging) and TSV formatting. Converts frame-rate data to token-efficient output at selectable resolution. Presets: low (0.5/sec), medium (1/sec), high (4/sec).

Data flow: audio file → `load_audio()` → `AudioData` → `compute_spectrogram()` → `Spectrogram` → spectral/harmonic/rhythm analysis. Temporal features (RMS, ZCR) run on raw samples in parallel. All time-series share the same frame axis (n_fft=2048, hop=512).

## Time-Series Resolution

MCP tools accept an optional `resolution` parameter. When omitted, only summary stats are returned. When set, downsampled time-series data is appended as compact TSV. The `full_analysis` tool uses a unified table format (single shared time axis, all columns in one table) to minimize token usage.

## Key Dependencies

- **symphonia** — pure-Rust audio decoding
- **rustfft** — FFT computation
- **rmcp** — MCP server SDK (uses `schemars` for JSON Schema generation of tool params)
- **tokio** — async runtime (only for MCP server transport layer; analysis is synchronous)

## MCP Server Pattern

Tool methods are defined in `#[tool_router] impl AudioAnalyzerServer` blocks with `#[tool(description = "...")]` attributes. Parameter structs derive `Deserialize` + `schemars::JsonSchema`. The `#[tool_handler]` on the `ServerHandler` impl wires routing. Server logs to stderr to avoid interfering with stdio JSON-RPC.
