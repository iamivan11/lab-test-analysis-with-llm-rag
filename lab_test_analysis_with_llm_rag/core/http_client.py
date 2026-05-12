"""HTTP helpers for local llama-server calls."""

import threading
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

# Poll cadence for the cancellation watcher. Smaller = faster cancel
# response, more CPU; 0.05 s gives sub-100 ms cancellation latency
# at negligible overhead.
_CANCEL_POLL_SECONDS = 0.05


def _post_cancellable(
    url: str,
    json: Mapping[str, Any],
    timeout: float,
    stop_event,
) -> httpx.Response:
    """POST that returns instantly when `stop_event` fires.

    The naive `httpx.post(...)` blocks for the whole generation —
    up to `timeout` seconds — so a user-initiated cancel waits that
    long before propagating. We sidestep httpx's version-dependent
    cancellation semantics by running the POST in a daemon thread
    and polling `stop_event` from the caller; on cancel we raise
    `InterruptedError` immediately and **abandon** the daemon
    (Python cannot interrupt a blocked native socket read from
    another thread). The abandoned thread eventually completes when
    the server responds or the per-request timeout fires; its
    response is discarded.
    """
    if stop_event is None:
        return httpx.post(url, json=json, timeout=timeout)

    result: dict[str, Any] = {}

    def _do_post() -> None:
        try:
            result["response"] = httpx.post(url, json=json, timeout=timeout)
        except BaseException as e:
            result["error"] = e

    worker = threading.Thread(target=_do_post, daemon=True, name="http-post")
    worker.start()
    while worker.is_alive():
        if stop_event.wait(_CANCEL_POLL_SECONDS):
            raise InterruptedError("HTTP request cancelled")
    if "error" in result:
        err = result["error"]
        if isinstance(err, BaseException):
            raise err
    return result["response"]


def post_with_retries(
    url: str,
    *,
    json: Mapping[str, Any],
    timeout: float,
    attempts: int = 3,
    stop_event=None,
) -> httpx.Response:
    """Retry transient local-server failures, not HTTP application errors.

    `stop_event`: optional `threading.Event`; checked before each attempt
    AND used by an out-of-band watcher to abort the in-flight POST the
    moment cancellation fires, so a user-pressed cancel doesn't wait for
    llama-server to finish generating.
    """
    delay = 0.4
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("HTTP request cancelled")
        try:
            return _post_cancellable(url, json, timeout, stop_event)
        except _RETRYABLE_ERRORS as e:
            last_error = e
            if attempt == attempts:
                break
            log("HTTP", f"Retrying POST after transient error ({attempt}/{attempts}): {e}")
            # Sleep in small slices so a cancel during backoff is fast.
            slept = 0.0
            while slept < delay:
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError("HTTP request cancelled") from e
                step = min(0.1, delay - slept)
                time.sleep(step)
                slept += step
            delay *= 2
    raise last_error or RuntimeError("POST failed without a captured error")
