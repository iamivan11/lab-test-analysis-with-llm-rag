import contextlib
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import httpx

from config import (
    DEFAULT_MMPROJ_LOCAL,
    DEFAULT_MODEL_FILE,
    LLM_HEALTH_TIMEOUT_SECONDS,
    LLM_SERVER_READY_TIMEOUT_SECONDS,
    MODELS_DIR,
    PROJECT_ROOT,
    SERVER_HOST,
    SERVER_PORT,
)
from core.logger import log

SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
SERVER_BINARY = PROJECT_ROOT / "bin" / "llama-server"

_PROGRESS_MAP = [
    ("loading model", "Reading model file..."),
    ("loaded meta data", "Parsing model metadata..."),
    ("load_tensors: loading model tensors", "Loading model tensors..."),
    ("offloading output layer to GPU", "Offloading layers to GPU..."),
    ("offloaded", "Offloaded layers to GPU"),
    ("constructing llama_context", "Initializing context..."),
    ("warming up the model", "Warming up (first run takes a moment)..."),
    ("model loaded", "Model loaded, starting server..."),
    ("server is listening", "Server ready"),
]

_STDERR_RING_SIZE = 500


def _kill_port() -> None:
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{SERVER_PORT}"],
            capture_output=True,
            text=True,
        )
        for pid in result.stdout.strip().splitlines():
            subprocess.run(["kill", "-9", pid], capture_output=True)
    except Exception:
        pass


def _parse_progress(line: str) -> str | None:
    line_lower = line.lower()
    for keyword, message in _PROGRESS_MAP:
        if keyword in line_lower:
            return message
    return None


class LlamaServer:
    """Manages the llama-server subprocess lifecycle."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._model_path: str | None = None
        self._stderr_buffer: deque[str] = deque(maxlen=_STDERR_RING_SIZE)
        self._stderr_thread: threading.Thread | None = None
        self._progress_cb: Callable[[str], None] | None = None

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def current_model_path(self) -> str | None:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return None
            return self._model_path

    def start(
        self,
        model_path: str,
        n_ctx: int = 32768,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        with self._lock:
            log("SERVER", f"start called, model={model_path}, n_ctx={n_ctx}")

            from core.llama_setup import ensure_server

            ensure_server(on_progress=on_progress)

            self._stop_locked()
            _kill_port()

            cmd = [
                str(SERVER_BINARY),
                "--model",
                model_path,
                "--host",
                SERVER_HOST,
                "--port",
                str(SERVER_PORT),
                "--ctx-size",
                str(n_ctx),
                "-ngl",
                "99",
                "--no-webui",
                "--parallel",
                "2",
            ]
            if Path(model_path).name == DEFAULT_MODEL_FILE:
                mmproj_path = MODELS_DIR / DEFAULT_MMPROJ_LOCAL
                if mmproj_path.exists():
                    cmd.extend(["--mmproj", str(mmproj_path)])
            log("SERVER", f"Launching llama-server: {' '.join(cmd)}")

            self._stderr_buffer.clear()
            self._progress_cb = on_progress
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            log("SERVER", f"Process started, pid={self._process.pid}")

            self._stderr_thread = threading.Thread(
                target=self._pump_stderr, daemon=True, name="llama-stderr-pump"
            )
            self._stderr_thread.start()

            try:
                self._wait_for_ready()
            except Exception:
                self._dump_stderr_buffer()
                self._stop_locked()
                raise
            self._model_path = model_path
            log("SERVER", "Server is ready")

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        status = "alive" if self._process and self._process.poll() is None else "none"
        log("SERVER", f"stop called, process={status}")
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._stderr_thread:
            self._stderr_thread.join(timeout=2)
        self._process = None
        self._model_path = None
        self._stderr_thread = None
        self._progress_cb = None

    def _pump_stderr(self) -> None:
        proc = self._process
        if proc is None or proc.stderr is None:
            return
        with contextlib.suppress(Exception):
            for raw in proc.stderr:
                line = raw.rstrip()
                if not line:
                    continue
                self._stderr_buffer.append(line)
                cb = self._progress_cb
                if cb is not None:
                    msg = _parse_progress(line)
                    if msg:
                        with contextlib.suppress(Exception):
                            cb(msg)

    def _wait_for_ready(self, timeout: int = LLM_SERVER_READY_TIMEOUT_SECONDS) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            proc = self._process
            if proc is not None and proc.poll() is not None:
                raise RuntimeError("llama-server process exited unexpectedly")
            try:
                r = httpx.get(f"{SERVER_URL}/health", timeout=1)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return
            except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout):
                pass
            time.sleep(0.2)
        raise RuntimeError("llama-server did not become ready in time.")

    def _dump_stderr_buffer(self) -> None:
        if not self._stderr_buffer:
            return
        lines = list(self._stderr_buffer)
        log("LLM_STDERR", f"--- server stderr ({len(lines)} lines) ---")
        for line in lines:
            log("LLM_STDERR", line)
        log("LLM_STDERR", "--- end ---")


_server = LlamaServer()


def start_server(
    model_path: str,
    n_ctx: int = 32768,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    _server.start(model_path, n_ctx=n_ctx, on_progress=on_progress)


def stop_server() -> None:
    _server.stop()


def get_current_model_path() -> str | None:
    return _server.current_model_path()


def is_server_running() -> bool:
    if not _server.is_running():
        return False
    try:
        r = httpx.get(f"{SERVER_URL}/health", timeout=LLM_HEALTH_TIMEOUT_SECONDS)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout):
        return False
