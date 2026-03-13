# audio_visualizer_rs

**An MCP server that gives Claude the ability to hear music.**

Point Claude at any audio file and it can tell you the key, tempo, dynamics, timbre, percussive character, stereo field, structural sections, and how the music evolves over time -- all from raw audio analysis, no images, no guessing, under 1% context window usage.

Compare two tracks side-by-side. Detect where the music changes structurally — intro, verse, chorus, bridge — and zoom into the moments that matter.

## What is this?

LLMs can see (vision) and read (text), but they can't hear. This project bridges that gap by running real audio analysis -- the same DSP techniques used in music information retrieval research -- and returning structured numerical data that Claude can reason about.

It's an [MCP server](https://modelcontextprotocol.io/) that exposes audio analysis as tools Claude can call on demand. Ask Claude to analyze a song and it will decode the audio, run spectral/harmonic/rhythm/percussive analysis, and return the results as compact text. No spectrograms, no images, no wasted tokens.

Full analysis of a 60-second track completes in under 2 seconds (including source separation). Pure Rust. No Python, no FFmpeg, no system dependencies.

## Features

- **Audio decoding** -- mp3, wav, flac, ogg, aac via Symphonia (pure Rust)
- **Spectral analysis** -- centroid (brightness), bandwidth (richness), rolloff, flatness (tonality)
- **Frequency band energy** -- RMS energy across 7 standard producer bands (sub-bass through brilliance) for mix diagnosis
- **Spectral contrast** -- peak vs valley per band in dB, reveals clarity vs muddiness
- **Dynamic range** -- crest factor, loudness range (95th-5th percentile), peak dBFS
- **LUFS loudness** -- EBU R128 integrated loudness (ITU-R BS.1770-4 stereo channel summing), true peak (dBTP), loudness range (LRA), streaming platform targets. Validated to 0.0 dB of FabFilter Pro-L 2.
- **Stereo field analysis** -- phase correlation (mono compatibility), stereo width (mid/side ratio), L/R balance, mono compatibility score. Per-frame time-series to pinpoint where phase issues occur.
- **Temporal features** -- RMS energy (loudness), zero crossing rate (texture)
- **Timbre** -- 13 MFCCs (Mel-frequency cepstral coefficients)
- **Harmonic analysis** -- chromagram, key detection (Krumhansl-Schmuckler algorithm), tonnetz
- **Rhythm analysis** -- tempo estimation, beat tracking, onset detection, tempo stability
- **Percussive character** -- harmonic/percussive source separation (HPSS), attack sharpness, onset density
- **Section boundary detection** -- multi-feature novelty analysis detects structural changes (energy, spectral, harmonic, texture). Enables a summary→zoom workflow: get the map, then dive into interesting moments
- **A/B comparison** -- analyse two tracks side-by-side, get a compact diff table highlighting differences in loudness, dynamics, spectral balance, stereo field, key, and tempo
- **Time-series data** -- track how every feature evolves over time at selectable resolution
- **Token-efficient** -- downsampled output calibrated to fit comfortably in the context window

## See it in action

[Watch a fresh Claude analyse four tracks in one session](https://claude.ai/share/e40ea498-fe3e-4b22-9a70-81edf6637514) — an original piano/brass composition, Bohemian Rhapsody's full structure mapped with section boundaries, an A/B mix comparison with production advice, and a thunderstorm field recording where it triages music metrics as "meaningless" vs "physically real." All four analyses fit in a single context window.

## Installation

### Claude Desktop — one-click install

Download the `.mcpb` bundle for your platform from [GitHub Releases](https://github.com/JuzzyDee/audio-analyzer-rs/releases) and open it. Claude Desktop will handle the rest — no config files, no terminal, no setup.

| Platform | File |
|----------|------|
| macOS (Apple Silicon) | `audio-analyzer-darwin-arm64.mcpb` |
| macOS (Intel) | `audio-analyzer-darwin-x64.mcpb` |
| Windows | `audio-analyzer-win32-x64.mcpb` |
| Linux | `audio-analyzer-linux-x64.mcpb` |

### Claude Code — Homebrew (macOS)

```bash
brew tap JuzzyDee/tap
brew install audio-analyzer
claude mcp add --scope user audio-analyzer -- $(which audio-analyzer-mcp)
```

### Claude Code — manual (all platforms)

Download the `mcp-server` binary for your platform from [GitHub Releases](https://github.com/JuzzyDee/audio-analyzer-rs/releases), then:

```bash
claude mcp add --scope user audio-analyzer -- /path/to/mcp-server
```

### Build from source

```bash
git clone https://github.com/JuzzyDee/audio-analyzer-rs.git
cd audio-analyzer-rs
cargo build --release
claude mcp add --scope user audio-analyzer -- target/release/mcp-server
```

Restart Claude Desktop. The audio analysis tools will be available in your conversations.

**Note**: This is a local MCP server using stdio transport, so it requires Claude Code or Claude Desktop. It does not work with claude.ai in the browser or mobile apps.

## Usage

**Important**: This tool analyses files on your local machine. Give Claude the full file path (e.g., `/Users/you/Music/song.mp3`) -- don't try to upload or attach files to the chat. Claude will read the file directly from disk.

### CLI (standalone)

```bash
cargo run --bin cli -- /path/to/song.mp3
cargo run --bin cli -- compare /path/to/mix_v1.mp3 /path/to/mix_v2.mp3
```

### MCP tools

Once configured, Claude can call these tools directly:

| Tool | What it does |
|------|-------------|
| `audio_info` | Basic file info: duration, sample rate, sample count |
| `spectral_features` | Brightness, richness, loudness, texture, timbre (MFCCs), frequency band energy, spectral contrast, dynamic range, LUFS loudness, stereo field |
| `harmonic_analysis` | Key detection, pitch class distribution, tonnetz |
| `rhythm_analysis` | Tempo (BPM), beat positions, tempo stability |
| `full_analysis` | Everything above in one call, plus percussive character (HPSS), stereo field, and section boundaries. Recommended workflow: call without resolution first to get summary + section map, then zoom into interesting sections with `start_time`/`end_time` at high resolution |
| `compare` | A/B two tracks -- analyses both and returns a compact diff table |

### Example: full_analysis output

Here's what `full_analysis` returns for a 60-second jazz trio track:

```
═══ Full Audio Analysis ═══
File: /music/jazz_trio.mp3
Duration: 60.62 sec | Sample rate: 48000 Hz | Samples: 2909952
Analysis completed in: 2.02s

── Spectral/Temporal Features ──
Centroid (brightness):  2812 Hz — moderate
Bandwidth (richness):   3933 Hz — complex
Rolloff (energy focus): 6489 Hz
Flatness (tonality):    0.0824 — strongly tonal
RMS Energy (loudness):  0.1160
Zero Crossing Rate:     0.0402 — mixed
MFCCs (timbre):         [-141.3, 13.7, 0.9, 7.7, -2.5, 3.1, -1.7, 2.3, -0.9, 1.0, -0.5, 0.9, 0.6]

── Frequency Band Energy ──
Sub bass  (20–60 Hz):     0.007803
Bass      (60–250 Hz):    0.013043
Low-mid   (250–500 Hz):   0.004906
Mid       (500–2k Hz):    0.002165
Upper-mid (2k–4k Hz):     0.000203
Presence  (4k–6k Hz):     0.000166
Brilliance(6k–20k Hz):    0.000089

── Spectral Contrast (peak–valley dB) ──
Sub bass  (20–60 Hz):     10.4
Bass      (60–250 Hz):    20.4
Low-mid   (250–500 Hz):   26.0
Mid       (500–2k Hz):    30.9
Upper-mid (2k–4k Hz):     19.4
Presence  (4k–6k Hz):     16.6
Brilliance(6k–20k Hz):    55.2

── Harmonic Content ──
Estimated key: E minor (confidence: 0.538)
Top pitch classes:
   1. G  0.627 ███████████████
   2. E  0.561 ██████████████
   3. C  0.512 ████████████
   4. D# 0.495 ████████████
   5. F# 0.487 ████████████
   6. C# 0.459 ███████████

── Rhythm ──
Tempo: 84.0 BPM (confidence: 0.316)
Beats detected: 69
Mean tempo: 84.3 BPM | Median: 84.0 BPM
Stability: 0.951 (0=free, 1=locked)

── Percussive Character ──
Percussive ratio:    0.277 — harmony-dominated
Onset density:       7.0/sec — very dense
Peak attack sharp:   1.000

── Dynamic Range ──
Peak:            -0.44 dBFS
Crest factor:    16.2 dB — very dynamic
Loudness range:  76.4 dB — very dynamic
Quiet sections:  -87.5 dBFS | Loud sections: -11.1 dBFS

── Loudness (EBU R128) ──
Integrated:      -12.1 LUFS
True peak:       0.0 dBTP
Loudness range:  3.7 LU
Spotify (-14):   turned DOWN 1.9 dB | Apple (-16): turned DOWN 3.9 dB | YouTube (-14): turned DOWN 1.9 dB

── Stereo Field ──
Phase correlation:   0.257 avg, -0.822 min — some phase issues
Phase warnings:      20.6% of frames have negative correlation
Stereo width:        0.812 avg, 2.747 max — wide
Balance:             -0.095 — slightly left
Mono compatibility:  0.620 avg, 0.117 min — significant mono loss

── Section Boundaries ──
  Working BPM:     84 (confidence: 0.32)
  Boundaries:      2
     0:18.3s  energy+spectral+texture (confidence: 0.85)
     0:42.1s  energy+harmonic (confidence: 0.67)
```

When you add `resolution: "medium"`, the output also includes a time-series table showing how every feature changes over the track's duration -- letting Claude see the intro build, the dynamic solo section, and the quiet outro.

## Time-series resolution

All analysis tools accept an optional `resolution` parameter that controls time-series output:

| Preset | Data points/sec | Use case |
|--------|----------------|----------|
| `"low"` | ~0.5/sec | Broad overview, equivalent to what you'd eyeball from a spectrogram image |
| `"medium"` | ~1/sec | Good default for most analysis tasks |
| `"high"` | ~4/sec | Detailed view for short passages or zooming in on transitions |

You can also pass a numeric string (e.g., `"20"`) for custom rates.

Without `resolution`, tools return summary statistics only (averages across the whole track). With it, you get a compact TSV table showing how features evolve over time -- centroid, RMS, dynamic range, chroma, onset strength, percussive ratio, band energy, spectral contrast, and more, all aligned to the same time axis.

The presets are calibrated for token efficiency. A 3-minute track at `"medium"` resolution produces roughly 180 rows of data -- enough to track musical structure without blowing up the context window.

## Architecture

```
audio file
    |
    +---> load_audio()          -- Symphonia decodes to mono f32 samples
    |         |
    |         v
    |     compute_spectrogram() -- STFT via rustfft, time-frequency matrix
    |         |
    |         +---> spectral.rs    -- centroid, bandwidth, rolloff, flatness, MFCCs, band energy, contrast
    |         +---> temporal.rs    -- RMS energy, zero crossing rate, dynamic range
    |         +---> harmonic.rs    -- chromagram, key detection, tonnetz
    |         +---> rhythm.rs      -- onset detection, tempo, beat tracking
    |         +---> percussive.rs  -- HPSS (source separation), attack sharpness, onset density
    |         +---> sections.rs    -- section boundary detection (multi-feature novelty)
    |
    +---> load_audio_stereo()   -- preserves L/R channels
              |
              +---> stereo.rs      -- phase correlation, width, balance, mono compatibility
              +---> temporal.rs    -- LUFS loudness (ITU-R BS.1770-4 stereo channel summing)
    |
    v
downsample.rs                   -- bin-average to target resolution, format as TSV
```

Two binaries share the same analysis library:
- **`cli`** (`src/main.rs`) -- runs all analyses and prints results
- **`mcp-server`** (`src/mcp_server.rs`) -- exposes tools over stdio JSON-RPC via rmcp

Key dependencies: [symphonia](https://github.com/pdeljanov/Symphonia) (audio decoding), [rustfft](https://github.com/ejmahler/RustFFT) (FFT), [rmcp](https://github.com/anthropics/rmcp) (MCP SDK).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

MIT License. See [LICENSE](LICENSE) for details.
