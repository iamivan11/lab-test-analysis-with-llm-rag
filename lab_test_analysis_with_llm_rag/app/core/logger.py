"""Lightweight console logger with timestamps for debugging."""

import contextlib
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

_file_handle: TextIO | None = None


def enable_file_logging(path: Path) -> None:
    """Mirror log output to `path`, truncating any previous file.

    Call once at app startup. stderr output is unaffected. If opening fails
    (e.g. read-only filesystem) logging continues to stderr only.
    """
    global _file_handle
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _file_handle = path.open("w", buffering=1, encoding="utf-8")
    except OSError as e:
        print(
            f"[LOGGER] Failed to open log file {path}: {e}",
            file=sys.stderr,
            flush=True,
        )
        _file_handle = None


def log(tag: str, msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{tag}] {msg}"
    print(line, file=sys.stderr, flush=True)
    if _file_handle is not None:
        with contextlib.suppress(OSError):
            _file_handle.write(line + "\n")
