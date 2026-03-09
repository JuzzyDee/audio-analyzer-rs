# audio_visualizer_rs

**An MCP server that gives Claude the ability to hear music.**

Point Claude at any audio file and it can tell you the key, tempo, dynamics, timbre, and how the music evolves over time -- all from raw audio analysis, no images, no guessing, under 1% context window usage.

## What is this?

LLMs can see (vision) and read (text), but they can't hear. This project bridges that gap by running real audio analysis -- the same DSP techniques used in music information retrieval research -- and returning structured numerical data that Claude can reason about.

It's an [MCP server](https://modelcontextprotocol.io/) that exposes audio analysis as tools Claude can call on demand. Ask Claude to analyze a song and it will decode the audio, run spectral/harmonic/rhythm analysis, and return the results as compact text. No spectrograms, no images, no wasted tokens.

Full analysis of a 60-second track completes in ~150-230ms. Pure Rust. No Python, no FFmpeg, no system dependencies.

## Features

- **Audio decoding** -- mp3, wav, flac, ogg, aac via Symphonia (pure Rust)
- **Spectral analysis** -- centroid (brightness), bandwidth (richness), rolloff, flatness (tonality)
- **Temporal features** -- RMS energy (loudness), zero crossing rate (texture)
- **Timbre** -- 13 MFCCs (Mel-frequency cepstral coefficients)
- **Harmonic analysis** -- chromagram, key detection (Krumhansl-Schmuckler algorithm), tonnetz
- **Rhythm analysis** -- tempo estimation, beat tracking, onset detection, tempo stability
- **Time-series data** -- track how every feature evolves over time at selectable resolution
- **Token-efficient** -- downsampled output calibrated to fit comfortably in the context window

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
claude mcp add audio-analyzer $(which audio-analyzer-mcp)
```

### Claude Code — manual (all platforms)

Download the `mcp-server` binary for your platform from [GitHub Releases](https://github.com/JuzzyDee/audio-analyzer-rs/releases), then:

```bash
claude mcp add audio-analyzer /path/to/mcp-server
```

### Build from source

```bash
git clone https://github.com/JuzzyDee/audio-analyzer-rs.git
cd audio_visualizer_rs
cargo build --release
claude mcp add audio-analyzer target/release/mcp-server
```

Restart Claude Desktop. The audio analysis tools will be available in your conversations.

**Note**: This is a local MCP server using stdio transport, so it requires Claude Code or Claude Desktop. It does not work with claude.ai in the browser or mobile apps.

## Usage

### CLI (standalone)

```bash
cargo run --bin cli -- /path/to/song.mp3
```

### MCP tools

Once configured, Claude can call these tools directly:

| Tool | What it does |
|------|-------------|
| `audio_info` | Basic file info: duration, sample rate, sample count |
| `spectral_features` | Brightness, richness, loudness, texture, timbre (MFCCs) |
| `harmonic_analysis` | Key detection, pitch class distribution, tonnetz |
| `rhythm_analysis` | Tempo (BPM), beat positions, tempo stability |
| `full_analysis` | Everything above in one call |

### Example: full_analysis output

Here's what `full_analysis` returns for a 58-second jazz piano trio track:

```
═══ Full Audio Analysis ═══
File: /music/bill_evans_waltz_for_debby.mp3
Duration: 58.24 sec | Sample rate: 44100 Hz | Samples: 2568264
Analysis completed in: 187.43ms

── Spectral/Temporal Features ──
Centroid (brightness):  1847 Hz — moderate
Bandwidth (richness):   1923 Hz — moderate
Rolloff (energy focus): 4102 Hz
Flatness (tonality):    0.0312 — strongly tonal
RMS Energy (loudness):  0.0873
Zero Crossing Rate:     0.0421 — mixed
MFCCs (timbre):         [-312.4, 78.2, -15.7, 22.1, -8.4, 5.9, -3.2, 1.8, -2.1, 0.9, -1.4, 0.6, -0.3]

── Harmonic Content ──
Estimated key: F major (confidence: 0.782)
Top pitch classes:
   1. F  0.847 █████████████████████
   2. A  0.713 █████████████████
   3. C  0.698 █████████████████
   4. A# 0.524 █████████████
   5. D  0.481 ████████████
   6. G  0.439 ██████████

── Rhythm ──
Tempo: 138.2 BPM (confidence: 0.614)
Beats detected: 134
Mean tempo: 137.8 BPM | Median: 138.1 BPM
Stability: 0.721 (0=free, 1=locked)
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

Without `resolution`, tools return summary statistics only (averages across the whole track). With it, you get a compact TSV table showing how features evolve over time -- centroid, RMS, chroma, onset strength, and more, all aligned to the same time axis.

The presets are calibrated for token efficiency. A 3-minute track at `"medium"` resolution produces roughly 180 rows of data -- enough to track musical structure without blowing up the context window.

## Architecture

```
audio file
    |
    v
load_audio()          -- Symphonia decodes to mono f32 samples
    |
    v
compute_spectrogram() -- STFT via rustfft, produces time-frequency matrix
    |
    +---> spectral.rs  -- centroid, bandwidth, rolloff, flatness, MFCCs
    +---> temporal.rs  -- RMS energy, zero crossing rate
    +---> harmonic.rs  -- chromagram, key detection, tonnetz
    +---> rhythm.rs    -- onset detection, tempo, beat tracking
    |
    v
downsample.rs         -- bin-average to target resolution, format as TSV
```

Two binaries share the same analysis library:
- **`cli`** (`src/main.rs`) -- runs all analyses and prints results
- **`mcp-server`** (`src/mcp_server.rs`) -- exposes tools over stdio JSON-RPC via rmcp

Key dependencies: [symphonia](https://github.com/pdeljanov/Symphonia) (audio decoding), [rustfft](https://github.com/ejmahler/RustFFT) (FFT), [rmcp](https://github.com/anthropics/rmcp) (MCP SDK).

## License

Non-commercial use only. Free for personal, educational, and research purposes. Commercial use requires a separate license -- see [LICENSE](LICENSE) for details.
