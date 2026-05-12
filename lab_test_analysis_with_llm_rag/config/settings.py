"""Persisted user settings + all per-setting accessor helpers.

The on-disk shape is described by AppSettings (Pydantic). All reads
and writes go through `_load_settings` / `_save_settings`, which take
the module-level `_SETTINGS_LOCK` so concurrent load-modify-save
cycles from different threads don't clobber each other.
"""

import json
import threading
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from config.models import (
    APPROVED_MODELS,
    DEFAULT_MODEL_CHOICE_IDS,
    SYSTEM_MODEL_ID,
)
from config.paths import PROFILE_FILE, SETTINGS_FILE
from core.file_io import atomic_write_json

# Answer-detail vocabulary lives with settings because that's the only
# code path that reads / writes it.
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
    hidden_biomarkers: list[str] = Field(default_factory=list)


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


# Serialises the load-modify-save cycle across threads. atomic_write_json
# protects against partial writes on crash, but not against concurrent
# load-modify-save sequences clobbering each other — e.g. the chat
# worker calling save_model_meta() at the same time as the UI saving
# the user's profile. Holding the lock for the whole cycle is the only
# safe way without inventing per-field merge logic.
_SETTINGS_LOCK = threading.RLock()


def _load_settings() -> dict:
    """Load settings from disk, returning {} on missing or corrupt file."""
    with _SETTINGS_LOCK:
        return _load_app_settings().model_dump(exclude_none=True)


def _save_settings(data: dict) -> None:
    """Atomic write: write to temp file then rename, so a crash can't corrupt settings."""
    with _SETTINGS_LOCK:
        try:
            settings = AppSettings.model_validate(_migrate_settings(data))
            atomic_write_json(SETTINGS_FILE, settings.model_dump(exclude_none=True))
        except (OSError, ValueError) as e:
            from core.logger import log

            log("CONFIG", f"Failed to save settings: {e}")


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


def load_hidden_biomarkers() -> set[str]:
    return set(_load_settings().get("hidden_biomarkers") or [])


def add_hidden_biomarker(name: str) -> None:
    data = _load_settings()
    hidden = list(data.get("hidden_biomarkers") or [])
    if name not in hidden:
        hidden.append(name)
        data["hidden_biomarkers"] = hidden
        _save_settings(data)


def clear_hidden_biomarkers() -> None:
    data = _load_settings()
    if data.get("hidden_biomarkers"):
        data["hidden_biomarkers"] = []
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
    from core.logger import log

    if PROFILE_FILE.exists():
        try:
            from core.security import read_protected_json

            profile = read_protected_json(PROFILE_FILE)
            result = profile if isinstance(profile, dict) else {}
            log("CONFIG", f"load_profile from PROFILE_FILE: {len(result)} keys")
            return result
        except Exception as e:
            log("CONFIG", f"Failed to read profile: {e}")
            return {}
    settings_profile = _load_settings().get("profile", {})
    log(
        "CONFIG",
        f"load_profile from settings.json: {len(settings_profile)} keys",
    )
    return settings_profile


def save_profile(profile: dict) -> None:
    from core.logger import log

    log("CONFIG", f"save_profile: {len(profile)} keys")
    try:
        from core.security import is_security_configured, write_protected_json

        if is_security_configured():
            write_protected_json(PROFILE_FILE, profile)
            log("CONFIG", "save_profile: wrote PROFILE_FILE (security on)")
            data = _load_settings()
            if "profile" in data:
                data.pop("profile", None)
                _save_settings(data)
            return
    except Exception as e:
        log("CONFIG", f"Failed to save protected profile: {e}")
    data = _load_settings()
    data["profile"] = profile
    _save_settings(data)
    log("CONFIG", "save_profile: wrote settings.json profile field")


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
