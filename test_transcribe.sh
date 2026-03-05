#!/bin/bash

# Test script for /v1/audio/transcriptions endpoint
# Usage: ./test_transcribe.sh <path_to_audio_file>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_audio_file>"
    exit 1
fi

FILE_PATH="$1"

if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File not found: $FILE_PATH"
    exit 1
fi

curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer test-key" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@${FILE_PATH}" \
  -F "model=whisper-1"
