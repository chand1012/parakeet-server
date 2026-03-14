use parakeet_server::audio::{
    convert_via_ffmpeg, decode_audio_to_mono_f32, resample_linear, TARGET_SAMPLE_RATE,
};
use parakeet_server::model_archive::{ensure_model_present, model_dir};
use std::path::Path;
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
use transcribe_rs::engines::parakeet::{
    ParakeetEngine, ParakeetInferenceParams, ParakeetModelParams, TimestampGranularity,
};
use transcribe_rs::{TranscriptionEngine, TranscriptionSegment};

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
