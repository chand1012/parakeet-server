# parakeet-server

A self-hosted, OpenAI-compatible speech-to-text API server powered by NVIDIA's [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model. Drop-in replacement for the Whisper API — point any Whisper-compatible client at it and start transcribing.

## Features

- **OpenAI Whisper API compatible** — works with existing clients (e.g. `openai-python`, `whisper.cpp` clients, etc.)
- **Multiple output formats** — `json`, `text`, `srt`, `vtt`, `verbose_json`
- **Automatic model download** — model archive fetched and extracted automatically on first run, no manual setup
- **Browser UI** — built-in web interface for quick testing at `http://localhost:8000`
- **Efficient inference** — INT8 quantized ONNX model via ONNX Runtime
- **Supports many audio formats** — MP3, WAV, FLAC, OGG, and more (via Symphonia)
- **Docker-ready** — multi-stage build with non-root user, minimal runtime image

## Quick Start

### Docker (recommended)

```bash
docker build -t parakeet-server:latest .
bash run.sh
```

The server will start on `http://localhost:8000`. Model files (~300 MB) are downloaded automatically on first transcription and cached in `./models/`.

### From Source

**Prerequisites:** Rust 1.93+, `cmake`, `protobuf-compiler`, `libssl-dev`, `pkg-config`

```bash
cargo build --release
./target/release/parakeet-server
```

## API Usage

### Transcribe Audio

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Audio file (≤ 100 MB) |
| `model` | string | No | Model name (default: `whisper-1`) |
| `response_format` | string | No | `json` (default), `text`, `srt`, `vtt`, `verbose_json` |
| `language` | string | No | Language code (default: `en`) |

**Example with curl:**

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F model=whisper-1
```

**JSON response:**
```json
{ "text": "Hello, world." }
```

**Verbose JSON response** (includes segment timestamps):
```json
{
  "text": "Hello, world.",
  "segments": [
    { "text": "Hello, world.", "start": 0.0, "end": 1.5 }
  ]
}
```

**SRT response:**
```
1
00:00:00,000 --> 00:00:01,500
Hello, world.
```

### Test Script

A convenience script is included for quick testing:

```bash
bash test_transcribe.sh path/to/audio.mp3
```

### Browser UI

Open `http://localhost:8000` in your browser for the built-in transcription interface.

## Using with OpenAI-Compatible Clients

Point any Whisper API client at your server by overriding the base URL:

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

with open("audio.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
    )

print(result.text)
```

## Model

Uses the **NVIDIA Parakeet TDT 0.6B v3** model (INT8 quantized):

| Property | Value |
|----------|-------|
| Architecture | NeMo Conformer with Transducer Decoder (TDT) |
| Parameters | ~600M (INT8 quantized) |
| Input | 16 kHz mono audio (auto-resampled) |
| Source | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |

Model files are stored in `models/parakeet-tdt-0.6b-v3-int8/` and extracted automatically from `https://blob.handy.computer/parakeet-v3-int8.tar.gz` on first use.

## Configuration

| Setting | Default | Notes |
|---------|---------|-------|
| Port | `8000` | Set by Rocket / exposed in Docker |
| Model directory | `models/parakeet-tdt-0.6b-v3-int8/` | Relative to working directory |
| Max file size | 100 MB | Enforced on upload |
| Sample rate | 16 kHz | Input audio is resampled automatically |

## Docker Details

The image uses a two-stage build to keep the runtime image small:

1. **Builder** (`rust:1.93`) — compiles the binary and ONNX Runtime from source
2. **Runtime** (`debian:bookworm-slim`) — minimal image with only the required shared libraries

The container runs as a non-root user (`parakeet`, UID 1000). The `run.sh` script mounts `./models` into the container so downloaded model files persist between runs:

```bash
docker run --rm -it \
  -p 8000:8000 \
  -v "$(pwd)/models:/home/parakeet/models" \
  parakeet-server:latest
```

## License

See [LICENSE](LICENSE).
