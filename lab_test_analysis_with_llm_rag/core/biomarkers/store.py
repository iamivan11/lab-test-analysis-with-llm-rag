"""Cache + document-lookup helpers for biomarker extraction."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import DATA_DIR, FILTERING_OUTPUT_DIR, list_uploaded_doc_paths
from core.logger import log
from core.security import read_protected_json, read_protected_text, write_protected_json

BIOMARKERS_FILE = DATA_DIR / "biomarkers.json"


def _doc_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def list_uploaded_docs() -> list[str]:
    return [p.name for p in list_uploaded_doc_paths()]


def _read_parsed(name: str) -> str:
    """Returns the post-sanitization (filtered) markdown for a document.

    We deliberately read from FILTERING_OUTPUT_DIR — the cleaned-up output
    of the parser's sanitization pass — rather than the raw PARSING_OUTPUT
    text. Filtered text has page-header/footer noise stripped and lab
    tables re-tabulated, which makes structured biomarker extraction
    materially more reliable.
    """
    md = FILTERING_OUTPUT_DIR / f"{Path(name).stem}.md"
    if md.exists():
        try:
            return read_protected_text(md)
        except OSError:
            return ""
    return ""


def load_cache() -> dict[str, Any]:
    if not BIOMARKERS_FILE.exists():
        return {"by_doc_hash": {}, "extracted_at": None}
    try:
        return read_protected_json(BIOMARKERS_FILE)
    except (OSError, json.JSONDecodeError) as e:
        log("BIO", f"cache read failed: {e}")
        return {"by_doc_hash": {}, "extracted_at": None}


def save_cache(by_doc_hash: dict[str, Any]) -> None:
    payload = {
        "extracted_at": datetime.now(UTC).isoformat(),
        "by_doc_hash": by_doc_hash,
    }
    write_protected_json(BIOMARKERS_FILE, payload)


def has_cache() -> bool:
    return BIOMARKERS_FILE.exists() and bool(load_cache().get("by_doc_hash"))


def clear_cache() -> None:
    BIOMARKERS_FILE.unlink(missing_ok=True)


def remove_from_cache(filename: str) -> None:
    """Drop all cache entries whose source_doc matches `filename`."""
    if not BIOMARKERS_FILE.exists():
        return
    cache = load_cache()
    by_doc_hash = cache.get("by_doc_hash") or {}
    kept = {h: entry for h, entry in by_doc_hash.items()
            if entry.get("source_doc") != filename}
    if len(kept) == len(by_doc_hash):
        return
    if not kept:
        BIOMARKERS_FILE.unlink(missing_ok=True)
        return
    save_cache(kept)


__all__ = [
    "BIOMARKERS_FILE",
    "clear_cache",
    "has_cache",
    "list_uploaded_docs",
    "load_cache",
    "remove_from_cache",
    "save_cache",
]
