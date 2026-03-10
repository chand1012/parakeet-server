use flate2::read::GzDecoder;
use futures_util::StreamExt;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

const MODEL_DIR: &str = "models/parakeet-tdt-0.6b-v3-int8";
const MODEL_ARCHIVE_URL: &str = "https://blob.handy.computer/parakeet-v3-int8.tar.gz";
const MODEL_FILES: [&str; 5] = [
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "nemo128.onnx",
    "vocab.txt",
    "config.json",
];

pub async fn ensure_model_present() -> Result<PathBuf, String> {
    let model_dir = PathBuf::from(MODEL_DIR);
    if model_files_present(&model_dir).await? {
        return Ok(model_dir);
    }

    let parent_dir = model_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    tokio::fs::create_dir_all(&parent_dir)
        .await
        .map_err(|err| format!("failed to create model parent dir: {err}"))?;

    let client = reqwest::Client::builder()
        .build()
        .map_err(|err| format!("failed to create http client: {err}"))?;

    let archive_path = parent_dir.join("parakeet-v3-int8.tar.gz.part");
    let extract_dir = parent_dir.join("parakeet-v3-int8.extracting");

    log::info!("downloading model archive from {}", MODEL_ARCHIVE_URL);

    let response = client
        .get(MODEL_ARCHIVE_URL)
        .send()
        .await
        .map_err(|err| format!("download failed for model archive: {err}"))?
        .error_for_status()
        .map_err(|err| format!("bad response for model archive: {err}"))?;

    let mut out = tokio::fs::File::create(&archive_path)
        .await
        .map_err(|err| format!("failed to create {}: {err}", archive_path.display()))?;

    let mut stream = response.bytes_stream();
    while let Some(chunk_result) = stream.next().await {
        let chunk =
            chunk_result.map_err(|err| format!("stream read failed for model archive: {err}"))?;
        out.write_all(&chunk)
            .await
            .map_err(|err| format!("write failed for {}: {err}", archive_path.display()))?;
    }

    out.flush()
        .await
        .map_err(|err| format!("flush failed for {}: {err}", archive_path.display()))?;

    if tokio::fs::try_exists(&extract_dir).await.map_err(|err| {
        format!(
            "failed to stat extract dir {}: {err}",
            extract_dir.display()
        )
    })? {
        tokio::fs::remove_dir_all(&extract_dir)
            .await
            .map_err(|err| {
                format!(
                    "failed to clear extract dir {}: {err}",
                    extract_dir.display()
                )
            })?;
    }
    tokio::fs::create_dir_all(&extract_dir)
        .await
        .map_err(|err| {
            format!(
                "failed to create extract dir {}: {err}",
                extract_dir.display()
            )
        })?;

    let archive_path_for_extract = archive_path.clone();
    let extract_dir_for_extract = extract_dir.clone();
    let model_dir_for_extract = model_dir.clone();
    tokio::task::spawn_blocking(move || {
        extract_model_archive(&archive_path_for_extract, &extract_dir_for_extract)?;
        install_model_files(&extract_dir_for_extract, &model_dir_for_extract)
    })
    .await
    .map_err(|err| format!("model extraction task failed: {err}"))??;

    let _ = tokio::fs::remove_file(&archive_path).await;
    let _ = tokio::fs::remove_dir_all(&extract_dir).await;

    Ok(model_dir)
}

async fn model_files_present(model_dir: &Path) -> Result<bool, String> {
    for file_name in MODEL_FILES {
        let file_path = model_dir.join(file_name);
        if !tokio::fs::try_exists(&file_path)
            .await
            .map_err(|err| format!("failed to stat model file {}: {err}", file_path.display()))?
        {
            return Ok(false);
        }
    }

    Ok(true)
}

fn extract_model_archive(archive_path: &Path, extract_dir: &Path) -> Result<(), String> {
    let archive = std::fs::File::open(archive_path)
        .map_err(|err| format!("failed to open {}: {err}", archive_path.display()))?;
    let decoder = GzDecoder::new(archive);
    let mut tar = tar::Archive::new(decoder);
    tar.unpack(extract_dir).map_err(|err| {
        format!(
            "failed to extract archive into {}: {err}",
            extract_dir.display()
        )
    })
}

fn install_model_files(extract_dir: &Path, model_dir: &Path) -> Result<(), String> {
    let source_dir = find_model_root(extract_dir)?;
    std::fs::create_dir_all(model_dir)
        .map_err(|err| format!("failed to create model dir {}: {err}", model_dir.display()))?;

    for file_name in MODEL_FILES {
        let source_path = source_dir.join(file_name);
        let destination_path = model_dir.join(file_name);
        let tmp_path = destination_path.with_extension("part");
        std::fs::copy(&source_path, &tmp_path).map_err(|err| {
            format!(
                "failed to copy {} to {}: {err}",
                source_path.display(),
                tmp_path.display()
            )
        })?;
        std::fs::rename(&tmp_path, &destination_path).map_err(|err| {
            format!(
                "failed to rename {} to {}: {err}",
                tmp_path.display(),
                destination_path.display()
            )
        })?;
    }

    Ok(())
}

fn find_model_root(root: &Path) -> Result<PathBuf, String> {
    let mut pending = vec![root.to_path_buf()];

    while let Some(path) = pending.pop() {
        if MODEL_FILES
            .iter()
            .all(|file_name| path.join(file_name).is_file())
        {
            return Ok(path);
        }

        for entry in std::fs::read_dir(&path)
            .map_err(|err| format!("failed to read extracted dir {}: {err}", path.display()))?
        {
            let entry = entry.map_err(|err| format!("failed to read extracted entry: {err}"))?;
            let child = entry.path();
            if child.is_dir() {
                pending.push(child);
            }
        }
    }

    Err(format!(
        "model archive did not contain expected files: {}",
        MODEL_FILES.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::{extract_model_archive, install_model_files, MODEL_FILES};
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn installs_model_files_from_nested_archive_directory() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("parakeet-server-test-{unique}"));
        let archive_path = root.join("model.tar.gz");
        let extract_dir = root.join("extract");
        let model_dir = root.join("model");

        fs::create_dir_all(&root).unwrap();

        let archive_file = fs::File::create(&archive_path).unwrap();
        let encoder = GzEncoder::new(archive_file, Compression::default());
        let mut tar = tar::Builder::new(encoder);

        for file_name in MODEL_FILES {
            let path = format!("parakeet-v3-int8/{file_name}");
            let contents = format!("test contents for {file_name}");
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, path, contents.as_bytes())
                .unwrap();
        }

        tar.into_inner().unwrap().finish().unwrap();

        extract_model_archive(&archive_path, &extract_dir).unwrap();
        install_model_files(&extract_dir, &model_dir).unwrap();

        for file_name in MODEL_FILES {
            let installed = fs::read_to_string(model_dir.join(file_name)).unwrap();
            assert_eq!(installed, format!("test contents for {file_name}"));
        }

        fs::remove_dir_all(&root).unwrap();
    }
}
