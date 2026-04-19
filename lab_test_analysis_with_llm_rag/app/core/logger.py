"""Lightweight console logger with timestamps for debugging."""

import sys
from datetime import datetime


def log(tag: str, msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}", file=sys.stderr, flush=True)
