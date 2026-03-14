use parakeet_server::model_archive::{ensure_model_present, model_dir};
use std::io::Cursor;
use std::path::Path;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Instant;

use once_cell::sync::Lazy;
use rocket::catch;
use rocket::data::Data;
use rocket::fairing::AdHoc;
use rocket::fs::NamedFile;
use rocket::http::ContentType;
use rocket::serde::json::Json;
use rocket::serde::Serialize;
use rocket::{get, launch, post, routes, Build, Rocket};
use rocket_multipart_form_data::{
    MultipartFormData, MultipartFormDataField, MultipartFormDataOptions,
};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};
use tokio::io::AsyncWriteExt;
use transcribe_rs::engines::parakeet::{
    ParakeetEngine, ParakeetInferenceParams, ParakeetModelParams, TimestampGranularity,
};
use transcribe_rs::{TranscriptionEngine, TranscriptionSegment};

const TARGET_SAMPLE_RATE: u32 = 16_000;

static ENGINE_STATE: Lazy<Mutex<EngineState>> = Lazy::new(|| {
    Mutex::new(EngineState {
        engine: ParakeetEngine::new(),
        model_loaded: false,
    })
});

struct EngineState {
    engine: ParakeetEngine,
    model_loaded: bool,
}

#[derive(Serialize)]
struct SimpleTranscriptionResponse {
    text: String,
}

#[derive(Serialize)]
struct VerboseJsonResponse {
    task: String,
    language: String,
    duration: f64,
    text: String,
    segments: Vec<VerboseSegment>,
}

#[derive(Serialize)]
struct VerboseSegment {
    id: usize,
    seek: usize,
    start: f32,
    end: f32,
    text: String,
}

#[derive(Serialize)]
struct OpenAiErrorBody {
    message: String,
    r#type: String,
    param: Option<String>,
    code: Option<String>,
}

#[derive(Serialize)]
struct OpenAiErrorResponse {
    error: OpenAiErrorBody,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
}

#[post("/v1/audio/transcriptions", data = "<data>")]
async fn transcribe(
    content_type: &ContentType,
    data: Data<'_>,
) -> Result<(ContentType, String), rocket::http::Status> {
    let request_started = Instant::now();
    log::info!(
        "received transcription request: content_type={}",
        content_type
    );

    let mut options = MultipartFormDataOptions::new();
    options
        .allowed_fields
        .push(MultipartFormDataField::raw("file").size_limit(100 * 1024 * 1024));
    options
        .allowed_fields
        .push(MultipartFormDataField::text("model"));
    options
        .allowed_fields
        .push(MultipartFormDataField::text("response_format"));
    options
        .allowed_fields
        .push(MultipartFormDataField::text("language"));

    let multipart_form = MultipartFormData::parse(content_type, data, options)
        .await
        .map_err(|err| {
            log::error!("multipart parse failed: {err}");
            rocket::http::Status::BadRequest
        })?;

    let file_field = multipart_form
        .raw
        .get("file")
        .and_then(|fields| fields.first())
        .ok_or(rocket::http::Status::BadRequest)?;

    let response_format = multipart_form
        .texts
        .get("response_format")
        .and_then(|v| v.first())
        .map(|t| t.text.trim())
        .unwrap_or("json");

    let language = multipart_form
        .texts
        .get("language")
        .and_then(|v| v.first())
        .map(|t| t.text.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or("en")
        .to_string();

    let file_name = file_field
        .file_name
        .clone()
        .unwrap_or_else(|| "upload.bin".to_string());
    let upload_size = file_field.raw.len();
    log::info!(
        "parsed transcription request: file_name={}, bytes={}, response_format={}, language={}",
        file_name,
        upload_size,
        response_format,
        language
    );

    let model_dir = model_dir();
    log::info!("model files ready at {}", model_dir.display());

    let extension = file_field
        .file_name
        .as_deref()
        .and_then(|name| Path::new(name).extension())
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase());
    log::info!(
        "preparing audio input: file_name={}, extension={}",
        file_name,
        extension.as_deref().unwrap_or("unknown")
    );

    let decoded = if extension.as_deref() == Some("wav") {
        log::info!("decoding wav input directly with symphonia");
        decode_audio_to_mono_f32(&file_field.raw, extension.as_deref()).map_err(|err| {
            log::error!("audio decode failed: {err}");
            rocket::http::Status::BadRequest
        })?
    } else {
        log::info!("converting input to mono 16kHz wav via ffmpeg");
        convert_via_ffmpeg(&file_field.raw).await.map_err(|err| {
            log::error!("ffmpeg conversion failed: {err}");
            rocket::http::Status::BadRequest
        })?
    };
    log::info!(
        "decoded audio: sample_rate={}, samples={}",
        decoded.sample_rate,
        decoded.samples.len()
    );

    let samples = if decoded.sample_rate == TARGET_SAMPLE_RATE {
        log::info!("input already at target sample rate {}", TARGET_SAMPLE_RATE);
        decoded.samples
    } else {
        log::info!(
            "resampling audio from {}Hz to {}Hz",
            decoded.sample_rate,
            TARGET_SAMPLE_RATE
        );
        resample_linear(&decoded.samples, decoded.sample_rate, TARGET_SAMPLE_RATE)
    };

    let duration = samples.len() as f64 / TARGET_SAMPLE_RATE as f64;
    log::info!(
        "running transcription: duration_seconds={:.2}, sample_count={}",
        duration,
        samples.len()
    );

    let transcription = tokio::task::spawn_blocking(move || {
        let mut state = ENGINE_STATE
            .lock()
            .map_err(|_| "engine lock poisoned".to_string())?;

        if !state.model_loaded {
            log::info!("loading parakeet model into inference engine");
            state
                .engine
                .load_model_with_params(&model_dir, ParakeetModelParams::int8())
                .map_err(|err| format!("failed to load parakeet model: {err}"))?;
            state.model_loaded = true;
            log::info!("parakeet model loaded successfully");
        } else {
            log::info!("reusing already loaded parakeet model");
        }

        let params = ParakeetInferenceParams {
            timestamp_granularity: TimestampGranularity::Segment,
        };

        state
            .engine
            .transcribe_samples(samples, Some(params))
            .map_err(|err| format!("transcription failed: {err}"))
    })
    .await
    .map_err(|err| {
        log::error!("transcription task failed: {err}");
        rocket::http::Status::InternalServerError
    })?
    .map_err(|err| {
        log::error!("transcription error: {err}");
        rocket::http::Status::InternalServerError
    })?;

    log::info!(
        "transcription complete: text_bytes={}, segments={}, elapsed_ms={}",
        transcription.text.len(),
        transcription
            .segments
            .as_ref()
            .map(|segments| segments.len())
            .unwrap_or(0),
        request_started.elapsed().as_millis()
    );

    Ok(format_openai_response(
        response_format,
        &language,
        duration,
        transcription.text,
        transcription.segments,
    ))
}

#[get("/")]
async fn index() -> Option<NamedFile> {
    NamedFile::open("static/index.html").await.ok()
}

#[get("/health")]
async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

struct DecodedAudio {
    samples: Vec<f32>,
    sample_rate: u32,
}

async fn convert_via_ffmpeg(bytes: &[u8]) -> Result<DecodedAudio, String> {
    let mut child = tokio::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("failed to spawn ffmpeg: {err}"))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "failed to acquire ffmpeg stdin".to_string())?;
    let input = bytes.to_vec();
    let stdin_task = tokio::spawn(async move {
        stdin
            .write_all(&input)
            .await
            .map_err(|err| format!("write to ffmpeg stdin failed: {err}"))?;
        stdin
            .shutdown()
            .await
            .map_err(|err| format!("failed to close ffmpeg stdin: {err}"))
    });

    let output = child
        .wait_with_output()
        .await
        .map_err(|err| format!("ffmpeg wait failed: {err}"))?;
    let stdin_result = stdin_task
        .await
        .map_err(|err| format!("ffmpeg stdin task failed: {err}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return match stdin_result {
            Ok(()) => Err(format!("ffmpeg failed: {stderr}")),
            Err(stdin_err) if stderr.trim().is_empty() => {
                Err(format!("ffmpeg failed and stdin write failed: {stdin_err}"))
            }
            Err(stdin_err) => Err(format!(
                "ffmpeg failed: {stderr}; stdin write also failed: {stdin_err}"
            )),
        };
    }

    stdin_result?;

    let raw = output.stdout;
    if raw.len() % 4 != 0 {
        return Err("ffmpeg output byte length is not a multiple of 4".to_string());
    }

    let samples: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    if samples.is_empty() {
        return Err("ffmpeg produced no audio samples".to_string());
    }

    Ok(DecodedAudio {
        samples,
        sample_rate: TARGET_SAMPLE_RATE,
    })
}

fn decode_audio_to_mono_f32(bytes: &[u8], extension: Option<&str>) -> Result<DecodedAudio, String> {
    let mut hint = Hint::new();
    if let Some(ext) = extension {
        hint.with_extension(ext);
    }

    let media = MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
    let probed = get_probe()
        .format(
            &hint,
            media,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|err| format!("failed to probe audio: {err}"))?;

    let mut format = probed.format;
    let (track_id, codec_params) = {
        let track = format
            .default_track()
            .ok_or_else(|| "no default audio track found".to_string())?;
        (track.id, track.codec_params.clone())
    };

    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| "missing sample rate in codec params".to_string())?;
    let channels = codec_params
        .channels
        .ok_or_else(|| "missing channel info in codec params".to_string())?;
    let channel_count = channels.count();

    let mut decoder = get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|err| format!("failed to create decoder: {err}"))?;

    let mut mono = Vec::<f32>::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::ResetRequired) => {
                return Err("decoder reset required, unsupported stream".to_string())
            }
            Err(err) => return Err(format!("failed to read packet: {err}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder
            .decode(&packet)
            .map_err(|err| format!("decode failed: {err}"))?;

        match decoded {
            AudioBufferRef::F32(buffer) => {
                append_mono_f32(&mut mono, buffer.chan(0), channel_count, Some(&buffer));
            }
            _ => {
                let spec = *decoded.spec();
                let mut interleaved = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                interleaved.copy_interleaved_ref(decoded);
                append_mono_interleaved(&mut mono, interleaved.samples(), channel_count);
            }
        }
    }

    if mono.is_empty() {
        return Err("audio stream decoded to empty sample set".to_string());
    }

    Ok(DecodedAudio {
        samples: mono,
        sample_rate,
    })
}

fn append_mono_f32(
    out: &mut Vec<f32>,
    first_channel: &[f32],
    channel_count: usize,
    original: Option<&symphonia::core::audio::AudioBuffer<f32>>,
) {
    if channel_count <= 1 {
        out.extend_from_slice(first_channel);
        return;
    }

    if let Some(buffer) = original {
        for frame_index in 0..buffer.frames() {
            let mut sum = 0.0;
            for chan in 0..channel_count {
                sum += buffer.chan(chan)[frame_index];
            }
            out.push(sum / channel_count as f32);
        }
    }
}

fn append_mono_interleaved(out: &mut Vec<f32>, interleaved: &[f32], channel_count: usize) {
    if channel_count <= 1 {
        out.extend_from_slice(interleaved);
        return;
    }

    for frame in interleaved.chunks(channel_count) {
        let sum: f32 = frame.iter().copied().sum();
        out.push(sum / channel_count as f32);
    }
}

fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if samples.is_empty() || from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src = i as f64 / ratio;
        let left = src.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let frac = (src - left as f64) as f32;
        let interpolated = samples[left] * (1.0 - frac) + samples[right] * frac;
        out.push(interpolated);
    }

    out
}

fn format_openai_response(
    response_format: &str,
    language: &str,
    duration: f64,
    text: String,
    segments: Option<Vec<TranscriptionSegment>>,
) -> (ContentType, String) {
    match response_format {
        "text" => (ContentType::Plain, text),
        "srt" => (
            ContentType::new("application", "x-subrip"),
            render_srt(segments.as_deref(), &text),
        ),
        "vtt" => (
            ContentType::new("text", "vtt"),
            render_vtt(segments.as_deref(), &text),
        ),
        "verbose_json" => {
            let verbose_segments = segments
                .unwrap_or_default()
                .into_iter()
                .enumerate()
                .map(|(idx, seg)| VerboseSegment {
                    id: idx,
                    seek: (seg.start * 100.0) as usize,
                    start: seg.start,
                    end: seg.end,
                    text: seg.text,
                })
                .collect::<Vec<_>>();

            let body = serde_json::to_string(&VerboseJsonResponse {
                task: "transcribe".to_string(),
                language: language.to_string(),
                duration,
                text,
                segments: verbose_segments,
            })
            .unwrap_or_else(|_| "{\"text\":\"serialization failed\"}".to_string());

            (ContentType::JSON, body)
        }
        _ => {
            let body = serde_json::to_string(&SimpleTranscriptionResponse { text })
                .unwrap_or_else(|_| "{\"text\":\"serialization failed\"}".to_string());
            (ContentType::JSON, body)
        }
    }
}

fn render_srt(segments: Option<&[TranscriptionSegment]>, fallback_text: &str) -> String {
    let Some(segments) = segments else {
        return format!(
            "1\n00:00:00,000 --> 00:00:10,000\n{}\n",
            fallback_text.trim()
        );
    };

    if segments.is_empty() {
        return format!(
            "1\n00:00:00,000 --> 00:00:10,000\n{}\n",
            fallback_text.trim()
        );
    }

    let mut out = String::new();
    for (idx, segment) in segments.iter().enumerate() {
        out.push_str(&(idx + 1).to_string());
        out.push('\n');
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            format_srt_timestamp(segment.start as f64),
            format_srt_timestamp(segment.end as f64),
            segment.text.trim()
        ));
    }
    out
}

fn render_vtt(segments: Option<&[TranscriptionSegment]>, fallback_text: &str) -> String {
    let mut out = String::from("WEBVTT\n\n");
    let Some(segments) = segments else {
        out.push_str("00:00:00.000 --> 00:00:10.000\n");
        out.push_str(fallback_text.trim());
        out.push('\n');
        return out;
    };

    if segments.is_empty() {
        out.push_str("00:00:00.000 --> 00:00:10.000\n");
        out.push_str(fallback_text.trim());
        out.push('\n');
        return out;
    }

    for segment in segments {
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            format_vtt_timestamp(segment.start as f64),
            format_vtt_timestamp(segment.end as f64),
            segment.text.trim()
        ));
    }

    out
}

fn format_srt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

fn format_vtt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
}

#[catch(400)]
fn bad_request() -> Json<OpenAiErrorResponse> {
    Json(OpenAiErrorResponse {
        error: OpenAiErrorBody {
            message: "Invalid multipart transcription request".to_string(),
            r#type: "invalid_request_error".to_string(),
            param: None,
            code: None,
        },
    })
}

#[catch(500)]
fn internal_error() -> Json<OpenAiErrorResponse> {
    Json(OpenAiErrorResponse {
        error: OpenAiErrorBody {
            message: "Internal transcription error".to_string(),
            r#type: "server_error".to_string(),
            param: None,
            code: None,
        },
    })
}

#[launch]
fn rocket() -> Rocket<Build> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("starting parakeet-server with default log level info");
    rocket::build()
        .attach(AdHoc::try_on_ignite(
            "Prepare model archive",
            |rocket| async move {
                log::info!("ensuring parakeet model is present during startup");
                match ensure_model_present().await {
                    Ok(model_dir) => {
                        log::info!(
                            "startup model preparation complete: {}",
                            model_dir.display()
                        );
                        Ok(rocket)
                    }
                    Err(err) => {
                        log::error!("startup model preparation failed: {err}");
                        Err(rocket)
                    }
                }
            },
        ))
        .mount("/", routes![index, transcribe, health])
        .register("/", rocket::catchers![bad_request, internal_error])
}
