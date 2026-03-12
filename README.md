# parakeet-server

A self-hosted, OpenAI-compatible speech-to-text API powered by NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model.

Point Whisper-compatible clients at this server and keep everything local.

[![GHCR](https://img.shields.io/badge/GHCR-ghcr.io%2Fchand1012%2Fparakeet--server-2ea44f?logo=github&style=flat-square)](https://ghcr.io/chand1012/parakeet-server)
[![Rocket](https://img.shields.io/badge/Rocket-0.5.1-blue?style=flat-square)](https://docs.rs/rocket/0.5.1/rocket/)
[![License](https://img.shields.io/github/license/chand1012/parakeet-server?color=blue&style=flat-square)](LICENSE)

## Why this project

- OpenAI Whisper API compatible endpoint: `POST /v1/audio/transcriptions`
- Automatic model download and local caching on first use
- Multiple response formats: `json`, `text`, `srt`, `vtt`, `verbose_json`
- Built-in browser UI at `http://localhost:8000`
- Docker-friendly deployment and Rust-native runtime

## Quick start

### Option 1: Run from GHCR image

```bash
docker pull ghcr.io/chand1012/parakeet-server:latest
docker run --rm -it \
  -p 8000:8000 \
  -v "$(pwd)/models:/home/parakeet/models" \
  ghcr.io/chand1012/parakeet-server:latest
```

### Option 2: Build and run locally

Prerequisites: Rust 1.93+, `cmake`, `protobuf-compiler`, `pkg-config`, OpenSSL dev libs.

```bash
cargo build --release
./target/release/parakeet-server
```

Server starts on `http://localhost:8000`.

## Docker Compose (modern)

```yaml
name: parakeet

services:
  app:
    image: ghcr.io/chand1012/parakeet-server:latest
    pull_policy: always
    init: true
    container_name: parakeet-server
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      ROCKET_PORT: "8000"
      RUST_LOG: info
    volumes:
      - ./models:/home/parakeet/models
```

Then run:

```bash
docker compose up -d
```

## API usage

### Basic transcription

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F model=whisper-1
```

### Optional form fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file | Yes | Audio upload (max 100 MB) |
| `model` | string | No | Default: `whisper-1` |
| `response_format` | string | No | `json` (default), `text`, `srt`, `vtt`, `verbose_json` |
| `language` | string | No | Default: `en` |

### Response examples

`json`:

```json
{ "text": "Hello, world." }
```

`verbose_json`:

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 1.5,
  "text": "Hello, world.",
  "segments": [
    { "id": 0, "seek": 0, "start": 0.0, "end": 1.5, "text": "Hello, world." }
  ]
}
```

`srt`:

```text
1
00:00:00,000 --> 00:00:01,500
Hello, world.
```

## OpenAI client compatibility

Works with standard OpenAI SDKs by overriding `base_url`:

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1",
)

with open("audio.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
    )

print(result.text)
```

## Browser UI

Open `http://localhost:8000` and upload an audio file to test transcription in the built-in web interface.

## Model and processing details

- Model: **NVIDIA Parakeet TDT 0.6B v3 (INT8)**
- Runtime: ONNX Runtime via `transcribe-rs`
- Audio decoding: Symphonia and ffmpeg fallback path
- Target sample rate: 16 kHz mono
- Model cache path: `models/parakeet-tdt-0.6b-v3-int8/`

On first use, the model archive is fetched automatically from:

- `https://blob.handy.computer/parakeet-v3-int8.tar.gz`

## Development

```bash
# build
cargo build
cargo build --release

# test
cargo test
cargo test <test_function_name>

# lint / format
cargo fmt --check
cargo fmt
cargo clippy
```

There is also a helper script for manual endpoint testing:

```bash
bash test_transcribe.sh path/to/audio.mp3
```

## License

MIT. See `LICENSE`.
