"""HTTP helpers for local llama-server calls."""

import time
from collections.abc import Mapping
from typing import Any

import httpx

from core.logger import log

_RETRYABLE_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)


def post_with_retries(
    url: str,
    *,
    json: Mapping[str, Any],
    timeout: float,
    attempts: int = 3,
) -> httpx.Response:
    """Retry transient local-server failures, not HTTP application errors."""
    delay = 0.4
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return httpx.post(url, json=json, timeout=timeout)
        except _RETRYABLE_ERRORS as e:
            last_error = e
            if attempt == attempts:
                break
            log("HTTP", f"Retrying POST after transient error ({attempt}/{attempts}): {e}")
            time.sleep(delay)
            delay *= 2
    raise last_error or RuntimeError("POST failed without a captured error")
