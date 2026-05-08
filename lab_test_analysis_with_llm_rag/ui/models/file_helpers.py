"""Helpers for displaying locally stored model files."""

from pathlib import Path

from config import approved_model_for_file


def is_main_model(path: Path) -> bool:
    return "mmproj" not in path.name.lower()


def model_display_name(path: Path, fallback: str = "") -> str:
    if model := approved_model_for_file(path):
        return model["display_name"]
    return fallback or path.stem
