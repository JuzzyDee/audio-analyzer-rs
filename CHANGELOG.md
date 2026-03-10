# Changelog

## [0.3.2] - 2026-03-10

### Fixed
- MCP server hang on `full_analysis` with resolution parameter. Root cause: chromagram
  (n_fft=8192) produces fewer frames than the spectrogram (n_fft=2048), causing an
  out-of-bounds panic on the tokio task that silently swallowed the MCP response.

### Added
- Time slicing on all MCP tools (`start_time`/`end_time` parameters) for zooming into
  specific sections without re-analysing the entire file.
- Lenient numeric deserialization — tool calls now accept `"110"` or `110` for numeric
  parameters, preventing type mismatch failures from LLM tool use.
- Improved tool descriptions with workflow guidance (summary → low overview → high zoom).

## [0.3.1] - 2026-03-10

### Fixed
- Key estimation off by one semitone (two bugs: FFT frequency resolution calculation
  and chromagram rotation direction).

## [0.3.0] - 2026-03-10

### Added
- Harmonic/percussive source separation (HPSS) via median filtering with soft masking.
- Percussive features: percussive ratio, attack sharpness, onset density.
- Percussive columns in the unified time-series table.

## [0.2.0] - 2026-03-09

### Added
- Non-commercial license.
- MCPB bundle support for one-click Claude Desktop install.
- Homebrew tap (`brew tap JuzzyDee/tap && brew install audio-analyzer`).
- `audio-analyzer-setup` script for automatic Claude Code/Desktop configuration.

## [0.1.0] - 2026-03-09

### Added
- Initial release: MCP audio analysis server for Claude.
- Audio decoding (mp3, wav, flac, ogg, aac) via Symphonia.
- Spectral features: centroid, bandwidth, rolloff, flatness, MFCCs.
- Harmonic analysis: chromagram, key detection (Krumhansl-Schmuckler), tonnetz.
- Rhythm analysis: tempo estimation, beat tracking, beat statistics.
- Temporal features: RMS energy, zero crossing rate.
- Time-series downsampling with resolution presets (low/medium/high).
- Unified table format for `full_analysis` — single shared time axis, all columns.
- CLI tool for standalone analysis.
