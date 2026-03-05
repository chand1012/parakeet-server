# OpenAI Whisper API Interface Documentation

## Endpoint

**URL:** `POST https://api.openai.com/v1/audio/transcriptions`

**Authentication:** Bearer token via `Authorization: Bearer $OPENAI_API_KEY` header

**Content-Type:** `multipart/form-data`

---

## Request Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | File (binary) | The audio file object (not file name) to transcribe. Supported formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm |
| `model` | String | The model to use for transcription. Common values: `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-transcribe-diarize`, `gpt-4o-mini-transcribe` |

### Optional Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | String | Optional text to guide the model's style or continue a previous audio segment. Helps improve transcription accuracy by providing context. |
| `response_format` | String | The format of the output. Options: `json`, `text`, `srt`, `verbose_json`, `vtt`, `diarized_json`. Default: `json`. For `gpt-4o-transcribe` and `gpt-4o-mini-transcribe`, only `json` is supported. |
| `temperature` | Number | The sampling temperature, between 0 and 1. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make it more focused and deterministic. Default: 0 |
| `language` | String | The language of the input audio. Pass the audio language code (e.g., `en`, `es`, `fr`) to improve transcription quality (auto-detection by default). |
| `timestamp_granularities` | Array of strings | For `verbose_json` format only. Specify which granularities to include. Options: `segment` or `word`. Can include both: `["segment", "word"]`. Must pass `segment` to get segment-level timestamps. |
| `stream` | Boolean | If set to `true`, the model response data will be streamed to the client as it is generated. Only works with `response_format=diarized_json`. |
| `include` | Array of strings | Additional information to include in the response. Use `logprobs` to get log probabilities of tokens (works with `json` format and models `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`). |
| `chunking_strategy` | String | For diarization models. How to chunk the audio before processing. Options: `auto`, `fixed_duration`, `speaker_aware`. |
| `known_speaker_names` | Array of strings | For diarization. Names of known speakers to identify in the audio. |
| `known_speaker_references` | Array of strings | For diarization. Audio samples for known speakers, in format `data:audio/wav;base64,{base64_encoded_audio}`. |

---

## Response Formats

### Standard JSON Response (default)

```json
{
  "text": "This is the transcribed text from the audio file."
}
```

### Text Response

```
This is the transcribed text from the audio file.
```

### SRT Response for Subtitle Files

```srt
1
00:00:00,000 --> 00:00:03,000
This is the first subtitle segment.

2
00:00:03,000 --> 00:00:06,000
This is the second subtitle segment.
```

### VTT Response for Web Video Time Text

```vtt
WEBVTT

00:00:00.000 --> 00:00:03.000
This is the first subtitle segment.

00:00:03.000 --> 00:00:06.000
This is the second subtitle segment.
```

### Verbose JSON Response (includes metadata)

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 123.45,
  "text": "This is the complete transcribed text.",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.0,
      "text": "This is the first segment.",
      "tokens": [1001, 234, 567, 890],
      "temperature": 0.0,
      "avg_logprob": -0.5,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.01
    },
    {
      "id": 1,
      "seek": 300,
      "start": 3.0,
      "end": 6.5,
      "text": "This is the second segment.",
      "tokens": [891, 234, 567, 1002],
      "temperature": 0.0,
      "avg_logprob": -0.4,
      "compression_ratio": 1.1,
      "no_speech_prob": 0.02
    }
  ]
}
```

### Verbose JSON with Word Timestamps

When `timestamp_granularities=["segment", "word"]` and `response_format=verbose_json`:

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 123.45,
  "text": "This is the complete transcribed text.",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.0,
      "text": "This is the first segment.",
      "tokens": [1001, 234, 567, 890],
      "temperature": 0.0,
      "avg_logprob": -0.5,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.01,
      "words": [
        {
          "word": "This",
          "start": 0.0,
          "end": 0.5,
          "probability": 0.95
        },
        {
          "word": "is",
          "start": 0.5,
          "end": 0.8,
          "probability": 0.98
        }
      ]
    }
  ],
  "words": [
    {
      "word": "This",
      "start": 0.0,
      "end": 0.5,
      "probability": 0.95
    }
  ]
}
```

### JSON with Logprobs

When `include=["logprobs"]` and `response_format=json`:

```json
{
  "text": "This is the transcribed text.",
  "logprobs": {
    "content": [
      {
        "token": "This",
        "logprob": -0.15,
        "bytes": [84, 104, 105, 115],
        "top_logprobs": [
          {"token": "This", "logprob": -0.15, "bytes": [84, 104, 105, 115]},
          {"token": "That", "logprob": -1.25, "bytes": [84, 104, 97, 116]},
          {"token": "It", "logprob": -2.45, "bytes": [73, 116]}
        ]
      }
    ]
  }
}
```

### Diarized JSON Response (multi-speaker)

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 123.45,
  "text": "Speaker 1: Hello. Speaker 2: Hi there.",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.0,
      "text": "Hello.",
      "words": [
        {
          "word": "Hello.",
          "start": 0.0,
          "end": 2.8
        }
      ],
      "speaker": "SPEAKER_00"
    },
    {
      "id": 1,
      "seek": 300,
      "start": 3.5,
      "end": 6.0,
      "text": "Hi there.",
      "words": [
        {
          "word": "Hi",
          "start": 3.5,
          "end": 3.8
        },
        {
          "word": "there.",
          "start": 3.9,
          "end": 5.5
        }
      ],
      "speaker": "SPEAKER_01"
    }
  ],
  "speaker_count": 2
}
```

---

## Example API Calls

### Basic Transcription

```bash
curl -X POST https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/audio.mp3" \
  -F model="whisper-1"
```

### Transcription with Verbose JSON Response

```bash
curl -X POST https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/audio.mp3" \
  -F model="whisper-1" \
  -F response_format="verbose_json"
```

### Transcription with Word Timestamps

```bash
curl -X POST https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/audio.mp3" \
  -F model="whisper-1" \
  -F response_format="verbose_json" \
  -F "timestamp_granularities[]=segment" \
  -F "timestamp_granularities[]=word"
```

### Transcription with Prompt for Style Guidance

```bash
curl -X POST https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/audio.mp3" \
  -F model="whisper-1" \
  -F response_format="json" \
  -F prompt="This is a technical discussion about artificial intelligence." \
  -F language="en"
```

### Transcription for SRT Subtitles

```bash
curl -X POST https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/audio.mp3" \
  -F model="whisper-1" \
  -F response_format="srt"
```

### Transcription with Speaker Diarization

```bash
curl -X POST https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/meeting.wav" \
  -F model="gpt-4o-transcribe-diarize" \
  -F response_format="diarized_json" \
  -F chunking_strategy="auto" \
  -F "known_speaker_names[]=agent" \
  -F "known_speaker_references[]=data:audio/wav;base64,AAA..."
```

---

## Supported Languages

The Whisper model supports transcription in over 90 languages including:

Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Malay, Marathi, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh

---

## Notes

1. **File Size Limits**: Audio files should not exceed the maximum file size limit (typically 25MB for standard API access, check current limits).
2. **Language Detection**: When `language` is not specified, the model attempts to auto-detect the language.
3. **Temperature**: For more consistent and less creative transcriptions, use lower temperatures (0.0-0.3). Higher temperatures introduce more randomness.
4. **Verbose JSON**: To get word-level timestamps, you must include both `"segment"` and `"word"` in `timestamp_granularities`.
5. **Supported Audio Formats**: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
6. **Model-Specific Features**:
   - `gpt-4o-transcribe` and `gpt-4o-mini-transcribe`: Only support `json` response format
   - `gpt-4o-transcribe-diarize`: Supports `json`, `text`, and `diarized_json` (required for speaker annotations)
