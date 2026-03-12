// analysis/mod.rs — Module root for audio analysis
//
// In Rust, when you have a directory of related files, you need a `mod.rs`
// file that acts as the module's entry point (similar to Python's __init__.py).
//
// This file declares which submodules exist and what's publicly visible.
// Think of it as the table of contents for the analysis module.

// Declare submodules. Each `pub mod X;` tells Rust:
// "there's a file called X.rs in this directory, make it publicly accessible"
pub mod spectral;
pub mod harmonic;
pub mod rhythm;
pub mod temporal;
pub mod percussive;
pub mod stereo;
pub mod downsample;
