#!/bin/bash

# Script to download and extract OpenSLR SLR81 audio samples for testing
# License: CC BY 4.0
# Source: https://www.openslr.org/81/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES_DIR="$SCRIPT_DIR/samples"
ARCHIVE="$SCRIPT_DIR/samples.tar.gz"
DOWNLOAD_URL="https://openslr.trmal.net/resources/81/samples.tar.gz"

# Speech-only files to extract (excluding piano music files)
SPEECH_FILES=(
    "samples/bmaher0.wav"    # Bill Maher "In Your Time"
    "samples/bmaher1.wav"
    "samples/joliver0.wav"   # John Oliver "Last Week Tonight"
    "samples/joliver1.wav"
    "samples/joliver2.wav"
    "samples/simons0.wav"    # Simons Institute
    "samples/simons1.wav"
    "samples/simons2.wav"
)

echo "Downloading OpenSLR SLR81 audio samples..."
echo "URL: $DOWNLOAD_URL"

# Download the archive
curl -L -o "$ARCHIVE" "$DOWNLOAD_URL" --progress-bar

echo ""
echo "Extracting speech-only samples to $SAMPLES_DIR/..."

# Create samples directory
mkdir -p "$SAMPLES_DIR"

# Extract only the speech files (not piano music)
for file in "${SPEECH_FILES[@]}"; do
    tar -xzf "$ARCHIVE" "$file"
    echo "  ✓ $file"
done

echo ""
echo "Cleaning up archive..."
rm "$ARCHIVE"

echo ""
echo "Done! Extracted ${#SPEECH_FILES[@]} speech samples:"
ls -la "$SAMPLES_DIR/"

echo ""
echo "Files are ready for testing in: $SAMPLES_DIR"
