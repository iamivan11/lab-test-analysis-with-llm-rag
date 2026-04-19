import json
import os
import tempfile
from pathlib import Path

from app.core.logger import log

APP_NAME = "Lab Test Analyzer"
APP_VERSION = "0.1.0"

DATA_DIR = Path.home() / "Library" / "Application Support" / "Lab Test Analyzer"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DOCS_DIR = DATA_DIR / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = DATA_DIR / "settings.json"

# Default model — auto-downloaded on first launch
DEFAULT_MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"
DEFAULT_MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_MMPROJ_FILE = "mmproj-BF16.gguf"
DEFAULT_MMPROJ_LOCAL = "mmproj-Qwen3.5-9B-BF16.gguf"


def _load_settings() -> dict:
    """Load settings from disk, returning {} on missing or corrupt file."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log("CONFIG", f"Failed to read settings: {e}")
        return {}


def _save_settings(data: dict) -> None:
    """Atomic write: write to temp file then rename, so a crash can't corrupt settings."""
    try:
        fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".json.tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        Path(tmp_path).replace(SETTINGS_FILE)
    except OSError as e:
        log("CONFIG", f"Failed to save settings: {e}")


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    return f"{size_bytes / 1024**3:.2f} GB"


def load_model_path() -> str | None:
    data = _load_settings()
    path = data.get("model_path", "")
    if path and Path(path).exists():
        return path
    return None


def save_model_path(path: str) -> None:
    data = _load_settings()
    data["model_path"] = path
    _save_settings(data)


def load_ctx_size() -> int | None:
    """Return the user's preferred context window size, or None if not set."""
    return _load_settings().get("ctx_size")


def save_ctx_size(size: int) -> None:
    data = _load_settings()
    data["ctx_size"] = size
    _save_settings(data)


def load_max_tokens() -> int | None:
    """Return the user's preferred max output tokens, or None if not set."""
    return _load_settings().get("max_tokens")


def save_max_tokens(value: int) -> None:
    data = _load_settings()
    data["max_tokens"] = value
    _save_settings(data)


def load_model_meta(model_path: str) -> dict | None:
    """Return cached metadata for a model, or None if not yet read."""
    return _load_settings().get("model_meta", {}).get(model_path)


def save_model_meta(model_path: str, meta: dict) -> None:
    """Persist model metadata keyed by absolute model path."""
    data = _load_settings()
    data.setdefault("model_meta", {})[model_path] = meta
    _save_settings(data)


def load_profile() -> dict:
    return _load_settings().get("profile", {})


def save_profile(profile: dict) -> None:
    data = _load_settings()
    data["profile"] = profile
    _save_settings(data)
