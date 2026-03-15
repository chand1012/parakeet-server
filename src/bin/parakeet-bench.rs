use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use sysinfo::System;
use transcribe_rs::engines::parakeet::{
    ParakeetEngine, ParakeetInferenceParams, ParakeetModelParams, TimestampGranularity,
};
use transcribe_rs::TranscriptionEngine;

use parakeet_server::audio::{decode_audio_to_mono_f32, resample_linear, TARGET_SAMPLE_RATE};
use parakeet_server::model_archive::ensure_model_present;

const DEFAULT_WAV_DIR: &str = "./wav_files";

struct BenchmarkResult {
    filename: String,
    duration_secs: f64,
    wall_clock_secs: f64,
    realtime_factor: f64,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let wav_dir = args.get(1).map(|s| s.as_str()).unwrap_or(DEFAULT_WAV_DIR);

    println!("parakeet-bench: Transcription Performance Benchmark");
    println!("===================================================");
    println!();

    let mut sys = System::new_all();
    sys.refresh_cpu_all();
    let cpu_info = sys.cpus()[0].brand().to_string();
    let core_count = sys.cpus().len();
    let threads = sys.cpus().len();
    println!(
        "CPU: {} ({} cores, {} threads)",
        cpu_info, core_count, threads
    );
    println!("Directory: {}", wav_dir);

    let wav_path = Path::new(wav_dir);
    if !wav_path.exists() {
        eprintln!("Error: Directory '{}' does not exist", wav_dir);
        std::process::exit(1);
    }

    let wav_files = collect_wav_files(wav_path);
    if wav_files.is_empty() {
        eprintln!("Error: No .wav files found in '{}'", wav_dir);
        std::process::exit(1);
    }

    println!("Files found: {}", wav_files.len());
    println!();

    let results = tokio::runtime::Runtime::new()
        .expect("Failed to create Tokio runtime")
        .block_on(run_benchmark(wav_files));

    print_results(&results);
}

fn collect_wav_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_wav_files_recursive(dir, &mut files);
    files.sort();
    files
}

fn collect_wav_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!("Error reading directory '{}': {}", dir.display(), err);
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        if path.is_dir() {
            collect_wav_files_recursive(&path, files);
        } else if let Some(ext) = path.extension() {
            if ext.to_string_lossy().to_lowercase() == "wav" {
                files.push(path);
            }
        }
    }
}

async fn run_benchmark(wav_files: Vec<PathBuf>) -> Vec<BenchmarkResult> {
    let model_dir = ensure_model_present()
        .await
        .expect("Failed to ensure model is present");
    println!("Model ready at: {}", model_dir.display());
    println!();

    let mut engine = ParakeetEngine::new();
    engine
        .load_model_with_params(&model_dir, ParakeetModelParams::int8())
        .expect("Failed to load Parakeet model");

    let params = Some(ParakeetInferenceParams {
        timestamp_granularity: TimestampGranularity::Segment,
    });

    let mut results = Vec::new();

    for wav_path in wav_files {
        let filename = wav_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let file_start = Instant::now();

        let file_bytes = fs::read(&wav_path).expect("Failed to read WAV file");
        let extension = wav_path.extension().and_then(|e| e.to_str());

        let decoded =
            decode_audio_to_mono_f32(&file_bytes, extension).expect("Failed to decode audio");

        let samples = if decoded.sample_rate == TARGET_SAMPLE_RATE {
            decoded.samples
        } else {
            resample_linear(&decoded.samples, decoded.sample_rate, TARGET_SAMPLE_RATE)
        };

        let duration_secs = samples.len() as f64 / TARGET_SAMPLE_RATE as f64;

        let _transcription = engine
            .transcribe_samples(samples, params.clone())
            .expect("Transcription failed");

        let wall_clock_secs = file_start.elapsed().as_secs_f64();
        let realtime_factor = duration_secs / wall_clock_secs;

        results.push(BenchmarkResult {
            filename,
            duration_secs,
            wall_clock_secs,
            realtime_factor,
        });
    }

    results
}

fn print_results(results: &[BenchmarkResult]) {
    let mut total_duration = 0.0;
    let mut total_wall_clock = 0.0;

    for r in results {
        total_duration += r.duration_secs;
        total_wall_clock += r.wall_clock_secs;
    }

    let avg_realtime_factor = total_duration / total_wall_clock;

    let max_filename_len = results
        .iter()
        .map(|r| r.filename.len())
        .max()
        .unwrap_or(10)
        .min(25);

    let header_row = format!(
        "┌{0}─┬{1:>8}┬{2:>10}┬{3:>11}┐",
        "─".repeat(max_filename_len + 2),
        "─".repeat(8),
        "─".repeat(10),
        "─".repeat(11)
    );

    let separator_row = format!(
        "├{0}─┼{1:>8}┼{2:>10}┼{3:>11}┤",
        "─".repeat(max_filename_len + 2),
        "─".repeat(8),
        "─".repeat(10),
        "─".repeat(11)
    );

    let footer_row = format!(
        "└{0}─┴{1:>8}┴{2:>10}┴{3:>11}┘",
        "─".repeat(max_filename_len + 2),
        "─".repeat(8),
        "─".repeat(10),
        "─".repeat(11)
    );

    let header = format!(
        "│ {0:<width$} │ {1:>8} │ {2:>10} │ {3:>11} │",
        "File",
        "Duration",
        "Wall Clock",
        "Realtime",
        width = max_filename_len
    );

    println!("{}", header_row);
    println!("{}", header);
    println!("{}", separator_row);

    for r in results {
        let filename_display = if r.filename.len() > max_filename_len {
            format!(
                ".{}/{}",
                "..",
                &r.filename[r.filename.len() - max_filename_len + 2..]
            )
        } else {
            r.filename.clone()
        };
        println!(
            "│ {0:<width$} │ {1:>8.2} │ {2:>10.2} │ {3:>10.1}x │",
            filename_display,
            r.duration_secs,
            r.wall_clock_secs,
            r.realtime_factor,
            width = max_filename_len
        );
    }

    println!("{}", separator_row);
    println!(
        "│ {0:<width$} │ {1:>8.2} │ {2:>10.2} │ {3:>10.1}x │",
        "TOTAL",
        total_duration,
        total_wall_clock,
        avg_realtime_factor,
        width = max_filename_len
    );
    println!("{}", footer_row);
}
