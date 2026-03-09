#!/bin/bash
# Build .mcpb bundles for each platform from release binaries.
#
# Usage:
#   ./mcpb/build-mcpb.sh <version>
#
# Expects release tarballs/zips in the current directory (as produced by CI):
#   audio-analyzer-aarch64-apple-darwin.tar.gz
#   audio-analyzer-x86_64-apple-darwin.tar.gz
#   audio-analyzer-x86_64-unknown-linux-gnu.tar.gz
#   audio-analyzer-x86_64-pc-windows-msvc.zip
#
# Produces:
#   audio-analyzer-darwin-arm64.mcpb
#   audio-analyzer-darwin-x64.mcpb
#   audio-analyzer-linux-x64.mcpb
#   audio-analyzer-win32-x64.mcpb

set -euo pipefail

VERSION="${1:?Usage: build-mcpb.sh <version>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$(pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

build_bundle() {
    local archive="$1"
    local output_name="$2"
    local binary_name="$3"  # mcp-server or mcp-server.exe
    local staging="$WORK_DIR/$output_name"

    echo "Building $output_name.mcpb ..."

    rm -rf "$staging"
    mkdir -p "$staging/server"

    # Copy manifest and update version
    sed "s/\"0.2.0\"/\"$VERSION\"/" "$SCRIPT_DIR/manifest.json" > "$staging/manifest.json"

    # Extract binary from archive
    if [[ "$archive" == *.zip ]]; then
        unzip -q -o "$archive" "$binary_name" -d "$staging/server/"
    else
        tar -xzf "$archive" -C "$staging/server/" "$binary_name"
    fi

    chmod +x "$staging/server/$binary_name" 2>/dev/null || true

    # Create .mcpb (zip archive) in the output directory
    (cd "$staging" && zip -q -r "$OUTPUT_DIR/$output_name.mcpb" .)

    echo "  -> $output_name.mcpb"
}

build_bundle "audio-analyzer-aarch64-apple-darwin.tar.gz"   "audio-analyzer-darwin-arm64" "mcp-server"
build_bundle "audio-analyzer-x86_64-apple-darwin.tar.gz"    "audio-analyzer-darwin-x64"   "mcp-server"
build_bundle "audio-analyzer-x86_64-unknown-linux-gnu.tar.gz" "audio-analyzer-linux-x64"  "mcp-server"
build_bundle "audio-analyzer-x86_64-pc-windows-msvc.zip"    "audio-analyzer-win32-x64"    "mcp-server.exe"

echo ""
echo "Done! Built $(ls -1 *.mcpb 2>/dev/null | wc -l | tr -d ' ') bundles."
