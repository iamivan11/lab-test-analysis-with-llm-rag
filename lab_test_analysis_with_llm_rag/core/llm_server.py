import contextlib
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable

import httpx

from config import (
    LLM_HEALTH_TIMEOUT_SECONDS,
    LLM_SERVER_READY_TIMEOUT_SECONDS,
    PROJECT_ROOT,
    SERVER_HOST,
    SERVER_PORT,
    approved_model_file_path,
    approved_model_for_file,
)
from core.device_compat import current_device_capabilities, llama_gpu_layer_args
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
        self._start_generation = 0

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def current_model_path(self) -> str | None:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return None
            return self._model_path

    def recent_stderr(self, n: int = 30) -> list[str]:
        """Return the last n lines llama-server has emitted to stderr.

        Useful when an inference call fails with a generic 5xx — the actual
        cause (OOM, Metal kernel error, KV-cache corruption, etc.) is in
        these lines.
        """
        with self._lock:
            buf = list(self._stderr_buffer)
        return buf[-n:]

    def start(
        self,
        model_path: str,
        n_ctx: int = 32768,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        log("SERVER", f"start called, model={model_path}, n_ctx={n_ctx}")

        from core.llama_setup import ensure_server

        ensure_server(on_progress=on_progress)

        capabilities = current_device_capabilities()
        gpu_args = llama_gpu_layer_args(capabilities)
        log(
            "SERVER",
            "Device capabilities: "
            f"system={capabilities.system}, machine={capabilities.machine}, "
            f"metal={capabilities.metal_available}, gpu_args={' '.join(gpu_args)}",
        )

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
            *gpu_args,
            "--no-webui",
            # Single-user app — one slot is enough. Two slots reserve a
            # second copy of the KV cache (significant on large models)
            # for no benefit.
            "--parallel",
            "1",
            # Smaller batch / ubatch keeps Metal activation memory under
            # control during prefill on Apple Silicon. The default
            # batch=2048 OOM'd on 27B models with KV+ctx already
            # competing for the unified-memory budget.
            "--batch-size",
            "512",
            "--ubatch-size",
            "128",
        ]
        if model := approved_model_for_file(model_path):
            mmproj_local = model.get("mmproj_local")
            if mmproj_local:
                mmproj_path = approved_model_file_path(model, mmproj_local)
                if mmproj_path.exists():
                    cmd.extend(["--mmproj", str(mmproj_path)])
        log("SERVER", f"Launching llama-server: {' '.join(cmd)}")

        with self._lock:
            self._start_generation += 1
            generation = self._start_generation
            self._stop_locked(invalidate_start=False)
            _kill_port()

            self._stderr_buffer.clear()
            self._progress_cb = on_progress
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            proc = self._process
            log("SERVER", f"Process started, pid={self._process.pid}")

            self._stderr_thread = threading.Thread(
                target=self._pump_stderr, daemon=True, name="llama-stderr-pump"
            )
            self._stderr_thread.start()

        try:
            self._wait_for_ready(generation, proc)
        except Exception:
            with self._lock:
                if generation == self._start_generation:
                    self._dump_stderr_buffer()
                    self._stop_locked(invalidate_start=False)
            raise
        with self._lock:
            if (
                generation != self._start_generation
                or self._process is not proc
                or self._process.poll() is not None
            ):
                raise RuntimeError("llama-server start was cancelled.")
            self._model_path = model_path
        log("SERVER", "Server is ready")

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self, *, invalidate_start: bool = True) -> None:
        if invalidate_start:
            self._start_generation += 1
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

    def _wait_for_ready(
        self,
        generation: int,
        proc: subprocess.Popen,
        timeout: int = LLM_SERVER_READY_TIMEOUT_SECONDS,
    ) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if generation != self._start_generation or self._process is not proc:
                    raise RuntimeError("llama-server start was cancelled.")
                if proc.poll() is not None:
                    raise RuntimeError("llama-server process exited unexpectedly")
            if proc.poll() is not None:
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


def recent_server_stderr(n: int = 30) -> list[str]:
    """Last n stderr lines from the running llama-server process."""
    return _server.recent_stderr(n=n)
