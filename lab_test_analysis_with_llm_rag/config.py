import json
import shutil
import sys
from contextlib import suppress
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core.file_io import atomic_write_json
from core.macos_compat import application_support_dir

APP_NAME = "Lab Analyzer"
APP_VERSION = "0.1.0"
MIN_MACOS_VERSION = "13.0"


def _project_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def _assets_root(project_root: Path) -> Path:
    if getattr(sys, "frozen", False):
        resources_root = project_root.parent / "Resources"
        if resources_root.exists():
            return resources_root
    return project_root


PROJECT_ROOT = _project_root()
ASSETS_ROOT = _assets_root(PROJECT_ROOT)
PROJECT_TMP_DIR = PROJECT_ROOT / "tmp"
ICONS_DIR = ASSETS_ROOT / "assets" / "icons"

LEGACY_DATA_DIR = application_support_dir("Lab Test Analyzer")
DATA_DIR = application_support_dir(APP_NAME)


def _migrate_legacy_data_dir() -> None:
    if not LEGACY_DATA_DIR.exists() or DATA_DIR.exists():
        return
    try:
        shutil.move(str(LEGACY_DATA_DIR), str(DATA_DIR))
    except OSError as e:
        # Surface to stderr (logger may not be configured yet at import
        # time): a silent failure leaves the user with two data dirs and
        # no idea their old chats/profile didn't carry over.
        print(
            f"WARNING: Failed to migrate legacy data dir from {LEGACY_DATA_DIR} "
            f"to {DATA_DIR}: {e}. Existing data may be inaccessible until "
            f"resolved manually.",
            file=sys.stderr,
        )


_migrate_legacy_data_dir()

MODELS_DIR = DATA_DIR / "models"
DOCS_DIR = DATA_DIR / "documents"
SETTINGS_FILE = DATA_DIR / "settings.json"
PROFILE_FILE = DATA_DIR / "profile.json"
APP_LOG_FILE = DATA_DIR / "logs" / "app.log"
CACHE_DIR = DATA_DIR / "cache"
PARSING_STAGING_DIR = CACHE_DIR / "parsing_staging"
PARSING_OUTPUT_DIR = DATA_DIR / "parsing_output"
FILTERING_OUTPUT_DIR = DATA_DIR / "filtering_output"
REPORTS_DIR = DATA_DIR / "reports"
RAG_DEBUG_ENABLED = False
RAG_DEBUG_DIR = DATA_DIR / "debug" / "rug_chunks"


def _ensure_private_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError:
        pass


for _dir in (
    DATA_DIR,
    MODELS_DIR,
    DOCS_DIR,
    APP_LOG_FILE.parent,
    CACHE_DIR,
    PARSING_STAGING_DIR,
    PARSING_OUTPUT_DIR,
    FILTERING_OUTPUT_DIR,
    REPORTS_DIR,
):
    _ensure_private_dir(_dir)


def _migrate_legacy_runtime_dir(old_path: Path, new_path: Path) -> None:
    """Copy legacy project-tmp runtime files into app data without deleting originals."""
    try:
        if not old_path.exists() or (new_path.exists() and any(new_path.iterdir())):
            return
        shutil.copytree(old_path, new_path, dirs_exist_ok=True)
    except OSError as e:
        print(
            f"WARNING: Failed to migrate legacy runtime dir from {old_path} "
            f"to {new_path}: {e}",
            file=sys.stderr,
        )


_migrate_legacy_runtime_dir(PROJECT_TMP_DIR / "parsing_output", PARSING_OUTPUT_DIR)
_migrate_legacy_runtime_dir(PROJECT_TMP_DIR / "filtering_output", FILTERING_OUTPUT_DIR)

# Curated app models. These are the only models the app should present as
# first-class supported choices; repo/file names stay internal.
APPROVED_MODELS = {
    "qwen35_9b_vision": {
        "display_name": "Qwen3.5-9B",
        "repo_id": "unsloth/Qwen3.5-9B-GGUF",
        "model_file": "Qwen3.5-9B-Q4_K_M.gguf",
        "mmproj_file": "mmproj-BF16.gguf",
        "mmproj_local": "mmproj-Qwen3.5-9B-BF16.gguf",
        "download_size_bytes": 6_602_227_488,
    },
    "qwen36_35b_a3b_vision": {
        "display_name": "Qwen3.6-35B-A3B",
        "repo_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "model_file": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "mmproj_local": "mmproj-Qwen3.6-35B-A3B-F16.gguf",
        "download_size_bytes": 23_033_528_992,
    },
    "gpt_oss_20b": {
        "display_name": "GPT-OSS-20B",
        "repo_id": "unsloth/gpt-oss-20b-GGUF",
        "model_file": "gpt-oss-20b-Q4_K_M.gguf",
        "download_size_bytes": 11_624_759_488,
    },
    "medgemma_4b_it": {
        "display_name": "MedGemma-4B-IT",
        "repo_id": "unsloth/medgemma-4b-it-GGUF",
        "model_file": "medgemma-4b-it-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "mmproj_local": "mmproj-MedGemma-4B-IT-F16.gguf",
        "download_size_bytes": 2_490_000_000,
    },
}


def approved_model_dir(model: dict) -> Path:
    return MODELS_DIR / model["display_name"]


def approved_model_file_path(model: dict, local_name: str | None = None) -> Path:
    return approved_model_dir(model) / (local_name or model["model_file"])


def approved_model_for_file(path: str | Path) -> dict | None:
    path = Path(path)
    for model in APPROVED_MODELS.values():
        if path.name != model["model_file"]:
            continue
        if path.parent.name == model["display_name"]:
            return model
        # Legacy flat model location from older app versions.
        if path.parent == MODELS_DIR:
            return model
    return None


def approved_main_model_paths() -> list[Path]:
    paths = []
    for model in APPROVED_MODELS.values():
        nested = approved_model_file_path(model)
        if nested.exists():
            paths.append(nested)
            continue
        legacy = MODELS_DIR / model["model_file"]
        if legacy.exists():
            paths.append(legacy)
    return paths


def _migrate_approved_model_files() -> None:
    for model in APPROVED_MODELS.values():
        for local_name in (model["model_file"], model.get("mmproj_local")):
            if not local_name:
                continue
            legacy = MODELS_DIR / local_name
            target = approved_model_file_path(model, local_name)
            if not legacy.exists() or target.exists():
                continue
            with suppress(OSError):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(legacy), str(target))


SYSTEM_MODEL_ID = "qwen35_9b_vision"
SYSTEM_MODEL = APPROVED_MODELS[SYSTEM_MODEL_ID]
MULTIMODAL_MODEL_IDS = tuple(
    model_id for model_id, model in APPROVED_MODELS.items() if model.get("mmproj_file")
)
# Models the user is allowed to pick as their default in Settings.
# Intentionally narrower than MULTIMODAL_MODEL_IDS — we only ship the two
# Qwen vision models as first-class default choices. Other approved models
# (e.g. MedGemma) remain installable for document-parsing use, but can't
# become the chat default. Append new IDs here as they get cleared.
DEFAULT_MODEL_CHOICE_IDS = ("qwen35_9b_vision", "qwen36_35b_a3b_vision")

# Compatibility aliases for existing code paths.
DEFAULT_MODEL_DISPLAY_NAME = SYSTEM_MODEL["display_name"]
DEFAULT_MODEL_REPO = SYSTEM_MODEL["repo_id"]
DEFAULT_MODEL_FILE = SYSTEM_MODEL["model_file"]
DEFAULT_MMPROJ_FILE = SYSTEM_MODEL["mmproj_file"]
DEFAULT_MMPROJ_LOCAL = SYSTEM_MODEL["mmproj_local"]

_migrate_approved_model_files()

# Curated allowlist of HuggingFace repo IDs that are permitted to be
# downloaded through the Models screen. The Download tab is built from
# APPROVED_MODELS, and download_model() refuses anything not on it.
# Add new IDs here as they get tested and approved.
ALLOWED_DOWNLOAD_MODEL_IDS: list[str] = [
    model["repo_id"] for model in APPROVED_MODELS.values()
]

# Dev override: when True, download_model() bypasses the allowlist and can
# download any GGUF repo/file. Keep False for released curated-list behavior.
ALLOW_ALL_MODEL_DOWNLOADS = False

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

ANSWER_DETAIL_SHORT = "short"
ANSWER_DETAIL_BALANCED = "balanced"
ANSWER_DETAIL_DETAILED = "detailed"
ANSWER_DETAIL_DEFAULT = ANSWER_DETAIL_BALANCED
ANSWER_DETAIL_MAX_TOKENS = {
    ANSWER_DETAIL_SHORT: 2048,
    ANSWER_DETAIL_BALANCED: 4096,
    ANSWER_DETAIL_DETAILED: 8192,
}

CURRENT_SETTINGS_SCHEMA_VERSION = 1


class ModelMetaSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    context_length: int | None = None


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = CURRENT_SETTINGS_SCHEMA_VERSION
    model_path: str | None = None
    # Legacy global ctx_size kept for backwards-compat deserialization;
    # all reads now go through ctx_size_by_model.
    ctx_size: int | None = None
    # Per-model ctx_size overrides, keyed by absolute model file path.
    ctx_size_by_model: dict[str, int] = Field(default_factory=dict)
    max_tokens: int | None = None
    answer_detail: str = ANSWER_DETAIL_DEFAULT
    default_model_id: str = SYSTEM_MODEL_ID
    model_meta: dict[str, ModelMetaSettings] = Field(default_factory=dict)
    profile: dict[str, str] = Field(default_factory=dict)
    onboarding_complete: bool = False


def _migrate_settings(raw: object) -> dict:
    """Normalize legacy settings dicts into the current schema."""
    if not isinstance(raw, dict):
        return {}

    migrated = dict(raw)
    if "answer_detail" not in migrated and isinstance(migrated.get("max_tokens"), int):
        tokens = migrated["max_tokens"]
        if tokens <= ANSWER_DETAIL_MAX_TOKENS[ANSWER_DETAIL_SHORT]:
            migrated["answer_detail"] = ANSWER_DETAIL_SHORT
        elif tokens >= ANSWER_DETAIL_MAX_TOKENS[ANSWER_DETAIL_DETAILED]:
            migrated["answer_detail"] = ANSWER_DETAIL_DETAILED
        else:
            migrated["answer_detail"] = ANSWER_DETAIL_BALANCED
    if migrated.get("answer_detail") not in ANSWER_DETAIL_MAX_TOKENS:
        migrated["answer_detail"] = ANSWER_DETAIL_DEFAULT
    if migrated.get("default_model_id") not in DEFAULT_MODEL_CHOICE_IDS:
        migrated["default_model_id"] = SYSTEM_MODEL_ID

    # Migrate the legacy global ctx_size to per-model storage. The legacy
    # value gets attributed to whichever model_path was active when it was
    # saved (best-effort). After migration the global field is cleared so
    # it doesn't override the per-model dict on subsequent reads.
    legacy_ctx = migrated.get("ctx_size")
    by_model = migrated.get("ctx_size_by_model") or {}
    if legacy_ctx is not None:
        path = migrated.get("model_path")
        if path and str(path) not in by_model:
            by_model = dict(by_model)
            by_model[str(path)] = legacy_ctx
        migrated["ctx_size_by_model"] = by_model
        migrated["ctx_size"] = None

    version = raw.get("schema_version")
    if version is None:
        return {
            "schema_version": CURRENT_SETTINGS_SCHEMA_VERSION,
            **{key: value for key, value in migrated.items() if key != "schema_version"},
        }

    if version == CURRENT_SETTINGS_SCHEMA_VERSION:
        return migrated

    from core.logger import log

    log("CONFIG", f"Unknown settings schema_version={version}, attempting best-effort load")
    return {
        "schema_version": CURRENT_SETTINGS_SCHEMA_VERSION,
        **{key: value for key, value in migrated.items() if key != "schema_version"},
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
        atomic_write_json(SETTINGS_FILE, settings.model_dump(exclude_none=True))
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


def load_ctx_size(model_path: str | None = None) -> int | None:
    """Saved context-window size for a specific model.

    Lookup is keyed by absolute model file path. When called without a
    path, falls back to the currently-saved active `model_path` (this
    matches the old global-setting semantics for callers that only care
    about "the model the app is running"). Returns None if no value is
    saved for that model — callers should default to `meta.context_length`
    (model max) on first load.
    """
    settings = _load_settings()
    by_model = settings.get("ctx_size_by_model") or {}
    if model_path is None:
        model_path = settings.get("model_path")
    if not model_path:
        return None
    return by_model.get(str(model_path))


def save_ctx_size(size: int, model_path: str | None = None) -> None:
    """Persist the context-window choice for a specific model."""
    data = _load_settings()
    if model_path is None:
        model_path = data.get("model_path")
    if not model_path:
        # Nothing to key the value on; ignore rather than overwrite the
        # legacy global field (which is no longer read).
        return
    by_model = dict(data.get("ctx_size_by_model") or {})
    by_model[str(model_path)] = int(size)
    data["ctx_size_by_model"] = by_model
    _save_settings(data)


def load_max_tokens() -> int | None:
    """Return the user's preferred max output tokens, or None if not set."""
    return _load_settings().get("max_tokens")


def save_max_tokens(value: int) -> None:
    data = _load_settings()
    data["max_tokens"] = value
    _save_settings(data)


def load_answer_detail() -> str:
    value = _load_settings().get("answer_detail", ANSWER_DETAIL_DEFAULT)
    return value if value in ANSWER_DETAIL_MAX_TOKENS else ANSWER_DETAIL_DEFAULT


def save_answer_detail(value: str) -> None:
    if value not in ANSWER_DETAIL_MAX_TOKENS:
        value = ANSWER_DETAIL_DEFAULT
    data = _load_settings()
    data["answer_detail"] = value
    data["max_tokens"] = ANSWER_DETAIL_MAX_TOKENS[value]
    _save_settings(data)


def answer_detail_max_tokens(value: str | None = None) -> int:
    key = value or load_answer_detail()
    return ANSWER_DETAIL_MAX_TOKENS.get(key, ANSWER_DETAIL_MAX_TOKENS[ANSWER_DETAIL_DEFAULT])


def load_default_model_id() -> str:
    model_id = _load_settings().get("default_model_id", SYSTEM_MODEL_ID)
    return model_id if model_id in DEFAULT_MODEL_CHOICE_IDS else SYSTEM_MODEL_ID


def save_default_model_id(model_id: str) -> None:
    if model_id not in DEFAULT_MODEL_CHOICE_IDS:
        model_id = SYSTEM_MODEL_ID
    data = _load_settings()
    data["default_model_id"] = model_id
    _save_settings(data)


def get_default_model() -> dict:
    return APPROVED_MODELS[load_default_model_id()]


def load_model_meta(model_path: str) -> dict | None:
    """Return cached metadata for a model, or None if not yet read."""
    return _load_settings().get("model_meta", {}).get(model_path)


def save_model_meta(model_path: str, meta: dict) -> None:
    """Persist model metadata keyed by absolute model path."""
    data = _load_settings()
    data.setdefault("model_meta", {})[model_path] = meta
    _save_settings(data)


def load_profile() -> dict:
    if PROFILE_FILE.exists():
        try:
            from core.security import read_protected_json

            profile = read_protected_json(PROFILE_FILE)
            return profile if isinstance(profile, dict) else {}
        except Exception as e:
            from core.logger import log

            log("CONFIG", f"Failed to read profile: {e}")
            return {}
    return _load_settings().get("profile", {})


def save_profile(profile: dict) -> None:
    try:
        from core.security import is_security_configured, write_protected_json

        if is_security_configured():
            write_protected_json(PROFILE_FILE, profile)
            data = _load_settings()
            if "profile" in data:
                data.pop("profile", None)
                _save_settings(data)
            return
    except Exception as e:
        from core.logger import log

        log("CONFIG", f"Failed to save protected profile: {e}")
    data = _load_settings()
    data["profile"] = profile
    _save_settings(data)


def migrate_profile_to_protected_file() -> None:
    """Move legacy profile data out of settings after password protection is enabled."""
    data = _load_settings()
    profile = data.get("profile")
    if not isinstance(profile, dict) or not profile:
        return
    try:
        from core.security import write_protected_json

        write_protected_json(PROFILE_FILE, profile)
    except Exception as e:
        from core.logger import log

        log("CONFIG", f"Failed to migrate profile: {e}")
        return
    data.pop("profile", None)
    _save_settings(data)


def is_onboarding_complete() -> bool:
    return bool(_load_settings().get("onboarding_complete", False))


def set_onboarding_complete(value: bool) -> None:
    data = _load_settings()
    data["onboarding_complete"] = value
    _save_settings(data)
