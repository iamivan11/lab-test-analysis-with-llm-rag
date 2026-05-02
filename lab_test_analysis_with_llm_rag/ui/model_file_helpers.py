"""Helpers for displaying locally stored model files."""

from pathlib import Path

from config import load_model_meta
from core.model_meta import _clean_name, read_model_name


def is_main_model(path: Path) -> bool:
    return "mmproj" not in path.name.lower()


def model_display_name(path: Path, fallback: str = "") -> str:
    meta = load_model_meta(str(path))
    if meta and meta.get("name"):
        return _clean_name(meta["name"])
    return read_model_name(str(path)) or fallback
