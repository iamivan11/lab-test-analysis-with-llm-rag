"""User-data cleanup helpers."""

import shutil
from pathlib import Path

from config import (
    DOCS_DIR,
    FILTERING_OUTPUT_DIR,
    PARSING_OUTPUT_DIR,
    PARSING_STAGING_DIR,
    PROFILE_FILE,
    RAG_DEBUG_DIR,
    REPORTS_DIR,
    _load_settings,
    _save_settings,
)
from core.logger import log


def _clear_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.unlink(missing_ok=True)


def clear_user_data() -> None:
    """Delete user-created data while keeping models, app settings, and security state."""
    from core.biomarkers import BIOMARKERS_FILE
    from core.chat_store import CHATS_DIR
    from core.knowledge_base import clear_index

    clear_index()
    for path in (
        DOCS_DIR,
        PARSING_STAGING_DIR,
        PARSING_OUTPUT_DIR,
        FILTERING_OUTPUT_DIR,
        REPORTS_DIR,
        RAG_DEBUG_DIR,
        CHATS_DIR,
        PROFILE_FILE,
        BIOMARKERS_FILE,
    ):
        _clear_path(path)

    settings = _load_settings()
    settings.pop("profile", None)
    _save_settings(settings)


def purge_document_artifacts(filename: str) -> None:
    """Remove every cross-subsystem trace of an uploaded document.

    Called whenever a document leaves the user's library — explicit
    delete, batch delete, parse-failure cleanup, cancellation cleanup.
    Keeps deletion holistic so stale chunks / cached biomarkers / old
    report-metadata entries don't outlive their source file.
    """
    from core.biomarkers.store import remove_from_cache
    from core.health_report import remove_from_metadata
    from core.knowledge_base import remove_document

    remove_document(filename)

    stem = Path(filename).stem
    for path in (
        PARSING_OUTPUT_DIR / f"{stem}.md",
        FILTERING_OUTPUT_DIR / f"{stem}.md",
        FILTERING_OUTPUT_DIR / f"{stem}.meta.json",
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            log("DOCS", f"Failed to remove {path}: {e}")

    remove_from_cache(filename)
    remove_from_metadata(filename)
