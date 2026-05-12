"""Curated model catalog and the helpers that locate model files.

The catalog is the only place repo IDs and exact filenames appear in
the code; downstream code refers to models by their `display_name`.
"""

import shutil
from contextlib import suppress
from pathlib import Path

from config.paths import MODELS_DIR


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
