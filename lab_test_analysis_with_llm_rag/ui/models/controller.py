from pathlib import Path

from PySide6.QtCore import QThread, Signal

from config import load_model_meta, load_model_path, save_model_path
from core.llm_engine import start_server
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

    def run(self):
        log("WORKER", f"ServerStartWorker: starting with {self.model_path}")
        try:
            from config import load_ctx_size, save_model_meta
            from core.model_meta import read_model_meta

            meta = read_model_meta(self.model_path)
            save_model_meta(
                self.model_path,
                {
                    "name": meta.name,
                    "context_length": meta.context_length,
                },
            )
            self.meta_ready.emit(meta)

            n_ctx = load_ctx_size() or meta.context_length
            start_server(self.model_path, n_ctx=n_ctx, on_progress=self.progress.emit)
            log("WORKER", "ServerStartWorker: finished successfully")
            self.finished.emit(meta.name or Path(self.model_path).stem)
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

    def run(self):
        log("WORKER", "ModelDownloadWorker: ensuring default model")
        try:
            model_path = ensure_default_model(on_progress=self.progress.emit)
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
        if saved_path:
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

    def select_model(self):
        from ui.model_select_dialog import ModelSelectDialog

        dlg = ModelSelectDialog(parent=self.window)
        dlg.loaded_model_deleted.connect(self.ensure_default_model)
        if dlg.exec() and dlg.selected_path:
            save_model_path(dlg.selected_path)
            self.load_model(dlg.selected_path)

    def load_model(self, model_path: str, token: int | None = None):
        log("UI", f"_load_model: {model_path}")
        if token is None:
            token = self.new_load_token()
        cached = load_model_meta(model_path) or {}
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
        if meta.name:
            self.window._status_label.setText(f"Loading model: {meta.name}...")

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

    def on_server_error(self, error: str):
        worker = self.window._server_worker
        if worker is None or worker.token != self.window._load_token:
            log("UI", "Stale server-error event ignored")
            return
        log("UI", f"Server error: {error}")
        self.window._status_label.setText("Model load error: not supported")
        self.window._model_btn.setEnabled(not self.window._parsing_active)
