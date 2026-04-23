"""HuggingFace model hub client for browsing and downloading GGUF models."""

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.config import (
    DEFAULT_MMPROJ_FILE,
    DEFAULT_MMPROJ_LOCAL,
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_REPO,
    MODELS_DIR,
)
from app.core.logger import log

HF_API = "https://huggingface.co/api"
HF_BASE = "https://huggingface.co"


def search_models(query: str, limit: int = 20) -> list[dict]:
    """Search HuggingFace for models that actually contain GGUF files.

    The HF `filter=gguf` tag is loose — some tagged repos hold only
    safetensors. We post-filter by listing each repo's tree in parallel and
    dropping any that has zero `.gguf` files, so non-runnable repos never
    reach the UI.
    """
    params = urllib.parse.urlencode(
        {
            "search": query,
            "filter": "gguf",
            "sort": "downloads",
            "direction": "-1",
            "limit": str(limit),
        }
    )
    url = f"{HF_API}/models?{params}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read().decode())

    candidates = [
        {
            "id": m.get("id", ""),
            "author": m.get("author", ""),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "last_modified": m.get("lastModified", ""),
        }
        for m in data
    ]

    def _has_gguf(model_id: str) -> bool:
        try:
            return bool(list_gguf_files(model_id))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            log("HUB", f"Dropping {model_id} (tree fetch failed: {e})")
            return False

    with ThreadPoolExecutor(max_workers=10) as pool:
        flags = list(pool.map(_has_gguf, [c["id"] for c in candidates]))

    filtered = [c for c, ok in zip(candidates, flags, strict=True) if ok]
    log("HUB", f"search_models: {len(candidates)} candidates -> {len(filtered)} with GGUF files")
    return filtered


def list_gguf_files(model_id: str) -> list[dict]:
    """List .gguf files in a model repo with their sizes."""
    url = f"{HF_API}/models/{model_id}/tree/main"
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read().decode())

    files = [
        {"name": item.get("path", ""), "size": item.get("size", 0)}
        for item in data
        if item.get("path", "").endswith(".gguf")
    ]
    return sorted(files, key=lambda f: f["size"])


def download_model(
    model_id: str,
    filename: str,
    on_progress: Callable[[int, int], None] | None = None,
    stop_event=None,
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

    url = f"{HF_BASE}/{model_id}/resolve/main/{urllib.parse.quote(filename)}"
    dest = MODELS_DIR / filename
    tmp = dest.with_suffix(".gguf.part")

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0

            with tmp.open("wb") as f:
                while True:
                    if stop_event and stop_event.is_set():
                        raise InterruptedError("Download cancelled")

                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress:
                        on_progress(downloaded, total)

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
    model_path = MODELS_DIR / DEFAULT_MODEL_FILE
    mmproj_path = MODELS_DIR / DEFAULT_MMPROJ_LOCAL

    files_to_download = []
    if not model_path.exists():
        files_to_download.append((DEFAULT_MODEL_FILE, DEFAULT_MODEL_FILE))
    if not mmproj_path.exists():
        files_to_download.append((DEFAULT_MMPROJ_FILE, DEFAULT_MMPROJ_LOCAL))

    if not files_to_download:
        log("HUB", "Default model files already present")
        return str(model_path)

    for hf_name, local_name in files_to_download:
        dest = MODELS_DIR / local_name
        if on_progress:
            on_progress(f"Downloading {local_name}...")

        def _report(downloaded, total, _name=local_name):
            if on_progress and total > 0:
                pct = downloaded * 100 // total
                on_progress(f"Downloading {_name}... {pct}%")

        log("HUB", f"Downloading {hf_name} -> {local_name}")
        download_model(
            DEFAULT_MODEL_REPO,
            hf_name,
            on_progress=_report,
            stop_event=stop_event,
        )
        # Rename if HF filename differs from local name
        if hf_name != local_name:
            downloaded_path = MODELS_DIR / hf_name
            if downloaded_path.exists():
                downloaded_path.rename(dest)

    log("HUB", "Default model files ready")
    return str(model_path)
