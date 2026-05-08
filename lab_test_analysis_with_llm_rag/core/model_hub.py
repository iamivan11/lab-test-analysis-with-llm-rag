"""Approved HuggingFace model downloader."""

import socket
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from config import (
    ALLOW_ALL_MODEL_DOWNLOADS,
    ALLOWED_DOWNLOAD_MODEL_IDS,
    APPROVED_MODELS,
    MODELS_DIR,
    approved_model_file_path,
    get_default_model,
)
from core.logger import log

HF_BASE = "https://huggingface.co"

# Socket timeout for the HF download connection. urllib reuses this for
# both connect and per-read, so a TCP stall (no bytes received for this
# many seconds) raises socket.timeout instead of hanging forever. 60s is
# generous enough for slow links to receive at least one chunk.
_HF_SOCKET_TIMEOUT = 60


def _approved_filenames(model_id: str) -> set[str]:
    filenames: set[str] = set()
    for model in APPROVED_MODELS.values():
        if model["repo_id"] != model_id:
            continue
        filenames.add(model["model_file"])
        if mmproj_file := model.get("mmproj_file"):
            filenames.add(mmproj_file)
    return filenames


def download_model(
    model_id: str,
    filename: str,
    on_progress: Callable[[int, int], None] | None = None,
    stop_event=None,
    download_dir: Path | None = None,
) -> Path:
    """Download a GGUF file from HuggingFace to MODELS_DIR.

    Args:
        model_id: HuggingFace model ID (e.g. "bartowski/Qwen-7B-GGUF").
        filename: Name of the .gguf file to download.
        on_progress: Callback receiving (downloaded_bytes, total_bytes).
        stop_event: threading.Event; if set, cancels the download.

    Returns:
        Path to the downloaded file.
    """
    if not filename.lower().endswith(".gguf"):
        raise ValueError(f"Only .gguf files can be downloaded, got: {filename}")

    if not ALLOW_ALL_MODEL_DOWNLOADS and model_id not in ALLOWED_DOWNLOAD_MODEL_IDS:
        raise PermissionError(
            f"Model {model_id!r} is not on the approved download allowlist."
        )
    if not ALLOW_ALL_MODEL_DOWNLOADS and filename not in _approved_filenames(model_id):
        raise PermissionError(
            f"File {filename!r} is not approved for model {model_id!r}."
        )

    url = f"{HF_BASE}/{model_id}/resolve/main/{urllib.parse.quote(filename)}"
    dest_dir = download_dir or MODELS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    tmp = dest.with_suffix(".gguf.part")

    try:
        with urllib.request.urlopen(url, timeout=_HF_SOCKET_TIMEOUT) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0

            with tmp.open("wb") as f:
                while True:
                    if stop_event and stop_event.is_set():
                        raise InterruptedError("Download cancelled")

                    try:
                        chunk = resp.read(1024 * 1024)
                    except (TimeoutError, socket.timeout) as e:
                        raise OSError(
                            f"Download of {filename} timed out after "
                            f"{_HF_SOCKET_TIMEOUT}s of no data — please retry."
                        ) from e
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress:
                        on_progress(downloaded, total)

        # Integrity check: a dropped connection ends the read loop with
        # `chunk = b""` and exits silently. Without this guard, a partial
        # file would get renamed to the final destination — and llama.cpp
        # later fails with "failed to seek for tensor X" when the truncated
        # GGUF is loaded. Verify byte count matches Content-Length.
        if total > 0 and downloaded < total:
            raise OSError(
                f"Download of {filename} truncated: got {downloaded} of "
                f"{total} bytes. The connection likely dropped — please "
                f"retry the download."
            )

        tmp.rename(dest)
        return dest

    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def ensure_default_model(
    on_progress: Callable[[str], None] | None = None,
    stop_event=None,
) -> str:
    """Ensure the default model and its mmproj file are downloaded.

    Returns the path to the main model file.
    """
    model = get_default_model()
    model_path = approved_model_file_path(model)
    mmproj_file = model["mmproj_file"]
    mmproj_local = model["mmproj_local"]
    mmproj_path = approved_model_file_path(model, mmproj_local)

    files_to_download = []
    if not model_path.exists():
        files_to_download.append((model["model_file"], model["model_file"]))
    if not mmproj_path.exists():
        files_to_download.append((mmproj_file, mmproj_local))

    if not files_to_download:
        log("HUB", "Default model files already present")
        return str(model_path)

    for hf_name, local_name in files_to_download:
        dest = approved_model_file_path(model, local_name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if on_progress:
            on_progress(f"Downloading {local_name}...")

        def _report(downloaded, total, _name=local_name):
            if on_progress and total > 0:
                pct = downloaded * 100 // total
                on_progress(f"Downloading {_name}... {pct}%")

        log("HUB", f"Downloading {hf_name} -> {local_name}")
        download_model(
            model["repo_id"],
            hf_name,
            on_progress=_report,
            stop_event=stop_event,
            download_dir=dest.parent,
        )
        # Rename if HF filename differs from local name
        downloaded_path = dest.parent / hf_name
        if downloaded_path.exists() and downloaded_path != dest:
            downloaded_path.rename(dest)

    log("HUB", "Default model files ready")
    return str(model_path)
