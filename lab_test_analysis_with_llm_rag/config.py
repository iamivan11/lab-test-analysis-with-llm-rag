import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

APP_NAME = "Lab Analyzer"
APP_VERSION = "0.1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = PROJECT_ROOT / "tmp"
ICONS_DIR = PROJECT_ROOT / "assets" / "icons"

# Data folder name kept as "Lab Test Analyzer" to preserve existing user
# data (models, chats, settings, documents) after the rename.
DATA_DIR = Path.home() / "Library" / "Application Support" / "Lab Test Analyzer"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DOCS_DIR = DATA_DIR / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = DATA_DIR / "settings.json"
APP_LOG_FILE = DATA_DIR / "logs" / "app.log"
RAG_DEBUG_DIR = TMP_DIR / "rug_chunks"
PARSING_OUTPUT_DIR = TMP_DIR / "parsing_output"
FILTERING_OUTPUT_DIR = TMP_DIR / "filtering_output"

# Default model — auto-downloaded on first launch
DEFAULT_MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"
DEFAULT_MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_MMPROJ_FILE = "mmproj-BF16.gguf"
DEFAULT_MMPROJ_LOCAL = "mmproj-Qwen3.5-9B-BF16.gguf"

# Curated allowlist of HuggingFace repo IDs that are permitted to be
# downloaded through the Models screen. The Browse / Download tab filters
# search results against this list, and download_model() refuses anything
# not on it. Add new IDs here as they get tested and approved.
ALLOWED_DOWNLOAD_MODEL_IDS: list[str] = [
    DEFAULT_MODEL_REPO,
]

# Dev override: when True, the allowlist above is bypassed and ANY GGUF
# repo can be searched and downloaded. Flip back to False to restore the
# curated-allowlist behavior.
ALLOW_ALL_MODEL_DOWNLOADS = True

# Runtime tunables
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
KB_COLLECTION_NAME = "lab_documents"
EMBEDDER_BGE_M3 = "BAAI/bge-m3"
EMBEDDER_SNOWFLAKE_ARCTIC_L_V2 = "Snowflake/snowflake-arctic-embed-l-v2.0"
EMBEDDER_JINA_V5_TEXT_SMALL = "jinaai/jina-embeddings-v5-text-small"
EMBEDDER_JINA_V5_TEXT_NANO = "jinaai/jina-embeddings-v5-text-nano"
KB_EMBEDDING_MODEL = EMBEDDER_JINA_V5_TEXT_SMALL
EMBEDDER_CONFIGS = {
    EMBEDDER_BGE_M3: {
        "loader_kwargs": {},
        "document_encode_kwargs": {},
        "query_encode_kwargs": {},
    },
    EMBEDDER_SNOWFLAKE_ARCTIC_L_V2: {
        "loader_kwargs": {},
        "document_encode_kwargs": {},
        "query_encode_kwargs": {},
    },
    EMBEDDER_JINA_V5_TEXT_SMALL: {
        "loader_kwargs": {"trust_remote_code": True},
        "document_encode_kwargs": {"task": "retrieval", "prompt_name": "document"},
        "query_encode_kwargs": {"task": "retrieval", "prompt_name": "query"},
    },
    EMBEDDER_JINA_V5_TEXT_NANO: {
        "loader_kwargs": {"trust_remote_code": True},
        "document_encode_kwargs": {"task": "retrieval", "prompt_name": "document"},
        "query_encode_kwargs": {"task": "retrieval", "prompt_name": "query"},
    },
}
KB_CHUNK_SIZE = 500
KB_CHUNK_OVERLAP = 100
KB_TOP_K = 15
LOGGER_LEVEL = "INFO"
LOGGER_MAX_BYTES = 1_000_000
LOGGER_BACKUP_COUNT = 5
PARSER_MAX_OUTPUT_TOKENS = 8192
PARSER_PDF_DPI = 200
PARSER_MAX_PARALLEL_PAGES = 2
PARSER_VISION_TIMEOUT_SECONDS = 600
PARSER_SANITIZE_TIMEOUT_SECONDS = 300
PARSER_METADATA_TIMEOUT_SECONDS = 120
LLM_SERVER_READY_TIMEOUT_SECONDS = 120
LLM_HEALTH_TIMEOUT_SECONDS = 2
LLM_TOKENIZE_TIMEOUT_SECONDS = 10
LLM_QUERY_MODIFICATION_TIMEOUT_SECONDS = 60
LLM_HYDE_TIMEOUT_SECONDS = 60
LLM_RAG_COMPRESSION_TIMEOUT_SECONDS = 120

CURRENT_SETTINGS_SCHEMA_VERSION = 1


class ModelMetaSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    context_length: int | None = None


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = CURRENT_SETTINGS_SCHEMA_VERSION
    model_path: str | None = None
    ctx_size: int | None = None
    max_tokens: int | None = None
    model_meta: dict[str, ModelMetaSettings] = Field(default_factory=dict)
    profile: dict[str, str] = Field(default_factory=dict)
    onboarding_complete: bool = False


def _migrate_settings(raw: object) -> dict:
    """Normalize legacy settings dicts into the current schema."""
    if not isinstance(raw, dict):
        return {}

    version = raw.get("schema_version")
    if version is None:
        return {
            "schema_version": CURRENT_SETTINGS_SCHEMA_VERSION,
            **{key: value for key, value in raw.items() if key != "schema_version"},
        }

    if version == CURRENT_SETTINGS_SCHEMA_VERSION:
        return raw

    from core.logger import log

    log("CONFIG", f"Unknown settings schema_version={version}, attempting best-effort load")
    return {
        "schema_version": CURRENT_SETTINGS_SCHEMA_VERSION,
        **{key: value for key, value in raw.items() if key != "schema_version"},
    }


def _load_app_settings() -> AppSettings:
    """Load, migrate, and validate settings from disk."""
    if not SETTINGS_FILE.exists():
        return AppSettings()
    try:
        raw = json.loads(SETTINGS_FILE.read_text())
        return AppSettings.model_validate(_migrate_settings(raw))
    except (json.JSONDecodeError, OSError, ValueError) as e:
        from core.logger import log

        log("CONFIG", f"Failed to read settings: {e}")
        return AppSettings()


def _load_settings() -> dict:
    """Load settings from disk, returning {} on missing or corrupt file."""
    return _load_app_settings().model_dump(exclude_none=True)


def _save_settings(data: dict) -> None:
    """Atomic write: write to temp file then rename, so a crash can't corrupt settings."""
    try:
        settings = AppSettings.model_validate(_migrate_settings(data))
        fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".json.tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(settings.model_dump(exclude_none=True), f, ensure_ascii=False, indent=2)
        Path(tmp_path).replace(SETTINGS_FILE)
    except (OSError, ValueError) as e:
        from core.logger import log

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


def is_onboarding_complete() -> bool:
    return bool(_load_settings().get("onboarding_complete", False))


def set_onboarding_complete(value: bool) -> None:
    data = _load_settings()
    data["onboarding_complete"] = value
    _save_settings(data)
