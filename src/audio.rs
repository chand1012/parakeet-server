use std::io::Cursor;
use std::process::Stdio;

use tokio::io::AsyncWriteExt;

use symphonia::core::audio::{AudioBufferRef, SampleBuffer, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub async fn convert_via_ffmpeg(bytes: &[u8]) -> Result<DecodedAudio, String> {
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

pub fn decode_audio_to_mono_f32(
    bytes: &[u8],
    extension: Option<&str>,
) -> Result<DecodedAudio, String> {
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

pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
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
