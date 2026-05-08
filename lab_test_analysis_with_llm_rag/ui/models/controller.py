import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from config import approved_model_for_file, load_model_meta, load_model_path, save_model_path
from core.llm_engine import start_server
from core.llm_server import stop_server
from core.logger import log
from core.model_hub import ensure_default_model


class ServerStartWorker(QThread):
    progress = Signal(str)
    meta_ready = Signal(object)
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, model_path: str, token: int):
        super().__init__()
        self.model_path = model_path
        self.token = token

    def stop(self):
        # llama-server's start path uses an internal generation counter
        # to detect cancellation: stop_server() bumps it and terminates
        # any subprocess in flight, so the worker's _wait_for_ready loop
        # raises "start was cancelled" on its next tick and run() exits.
        stop_server()

    def run(self):
        log("WORKER", f"ServerStartWorker: starting with {self.model_path}")
        try:
            from config import load_ctx_size, save_model_meta
            from core.model_meta import read_model_meta

            meta = read_model_meta(self.model_path)
            display_name = (
                model["display_name"]
                if (model := approved_model_for_file(self.model_path))
                else meta.name
            )
            save_model_meta(
                self.model_path,
                {
                    "name": display_name,
                    "context_length": meta.context_length,
                },
            )
            self.meta_ready.emit(meta)

            # Per-model ctx setting: each model remembers its own value.
            # First-time load of a model defaults to that model's own max
            # context length (so the user gets full capability without
            # guessing). They can lower it per-model in Settings if memory
            # is tight.
            n_ctx = load_ctx_size(self.model_path) or (meta.context_length or 8192)
            start_server(self.model_path, n_ctx=n_ctx, on_progress=self.progress.emit)
            log("WORKER", "ServerStartWorker: finished successfully")
            self.finished.emit(display_name or Path(self.model_path).stem)
        except Exception as e:
            log("WORKER", f"ServerStartWorker: ERROR {e}")
            self.error_occurred.emit(str(e))


class ModelDownloadWorker(QThread):
    progress = Signal(str)
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, token: int):
        super().__init__()
        self.token = token
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        log("WORKER", "ModelDownloadWorker: ensuring default model")
        try:
            model_path = ensure_default_model(
                on_progress=self.progress.emit,
                stop_event=self._stop_event,
            )
            log("WORKER", f"ModelDownloadWorker: done, path={model_path}")
            self.finished.emit(model_path)
        except Exception as e:
            log("WORKER", f"ModelDownloadWorker: ERROR {e}")
            self.error_occurred.emit(str(e))


class ModelController:
    def __init__(self, window):
        self.window = window

    def try_load_saved_model(self):
        saved_path = load_model_path()
        if saved_path and Path(saved_path).exists():
            self.load_model(saved_path)
        else:
            self.ensure_default_model()

    def new_load_token(self) -> int:
        self.window._load_token += 1
        return self.window._load_token

    def ensure_default_model(self):
        self.window._status_label.setText("Preparing default model...")
        self.window._model_btn.setEnabled(False)
        self.window._send_btn.setEnabled(False)

        token = self.new_load_token()
        self.window._download_worker = ModelDownloadWorker(token)
        self.window._download_worker.progress.connect(self.window._status_label.setText)
        self.window._download_worker.finished.connect(self.on_default_model_ready)
        self.window._download_worker.error_occurred.connect(self.on_default_model_error)
        self.window._download_worker.start()

    def on_default_model_ready(self, model_path: str):
        worker = self.window._download_worker
        if worker is None or worker.token != self.window._load_token:
            log("UI", "Stale default-model ready event ignored")
            return
        save_model_path(model_path)
        self.load_model(model_path, token=worker.token)

    def on_default_model_error(self, error: str):
        worker = self.window._download_worker
        if worker is None or worker.token != self.window._load_token:
            log("UI", "Stale default-model error event ignored")
            return
        log("UI", f"Default model download error: {error}")
        self.window._status_label.setText(
            "Failed to download default model — click 'Load Model' to select manually"
        )
        self.window._model_btn.setEnabled(not self.window._parsing_active)

    def _server_start_in_progress(self) -> bool:
        worker = self.window._server_worker
        if worker is None:
            return False
        try:
            return bool(worker.isRunning())
        except RuntimeError:
            return False

    def load_model(self, model_path: str, token: int | None = None):
        log("UI", f"_load_model: {model_path}")
        if self._server_start_in_progress():
            active_path = getattr(self.window._server_worker, "model_path", "")
            if active_path == model_path:
                log("UI", f"Duplicate model load ignored while already loading: {model_path}")
            else:
                log("UI", f"Model load ignored while another model is loading: {model_path}")
            return

        if token is None:
            token = self.new_load_token()
        cached = load_model_meta(model_path) or {}
        if model := approved_model_for_file(model_path):
            display = model["display_name"]
        else:
            display = cached.get("name") or Path(model_path).stem
        self.window._status_label.setText(f"Loading model: {display}...")
        self.window._model_btn.setEnabled(False)
        self.window._send_btn.setEnabled(False)

        self.window._server_worker = ServerStartWorker(model_path, token)
        self.window._server_worker.progress.connect(self.window._status_label.setText)
        self.window._server_worker.meta_ready.connect(self.on_model_meta)
        self.window._server_worker.finished.connect(self.on_server_started)
        self.window._server_worker.error_occurred.connect(self.on_server_error)
        self.window._server_worker.start()

    def on_model_meta(self, meta):
        worker = self.window._server_worker
        if worker is None or worker.token != self.window._load_token:
            return
        log("UI", f"Model meta: name={meta.name!r}, ctx={meta.context_length}")
        display = (
            model["display_name"]
            if (model := approved_model_for_file(worker.model_path))
            else meta.name
        )
        if display:
            self.window._status_label.setText(f"Loading model: {display}...")

    def on_server_started(self, model_name: str):
        worker = self.window._server_worker
        if worker is None or worker.token != self.window._load_token:
            log("UI", "Stale server-started event ignored")
            return
        log("UI", f"Server started, model: {model_name}")
        self.window._model_name = model_name
        self.window._status_label.setText(f"Model: {model_name}")
        self.window._model_btn.setEnabled(not self.window._parsing_active)
        self.window._send_btn.setEnabled(True)
        self.window._update_ctx_chip()
        # Refresh the My Models table so the row for the just-loaded model
        # immediately shows the disabled "Loaded" badge instead of the
        # active "Load" button — without waiting for a tile re-activation.
        home = getattr(self.window, "_home_screen", None)
        if home is not None and hasattr(home, "model_hub"):
            home.model_hub.refresh_local()

    def on_server_error(self, error: str):
        worker = self.window._server_worker
        if worker is None or worker.token != self.window._load_token:
            log("UI", "Stale server-error event ignored")
            return
        log("UI", f"Server error: {error}")
        self.window._status_label.setText("Model load error: not supported")
        self.window._model_btn.setEnabled(not self.window._parsing_active)
