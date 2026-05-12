from pathlib import Path

from config import approved_model_for_file, load_model_meta, load_model_path, save_model_path
from core.logger import log
from ui.models.workers import ModelDownloadWorker, ServerStartWorker


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

    def _is_current_worker(self, worker) -> bool:
        """True if `worker` is the in-flight load (its token matches).

        Used to drop stale callbacks from workers that completed after
        the user (or our own code) advanced the load token — e.g. a
        cancelled model swap whose `finished`/`error_occurred` arrives
        on the main thread after a new load has started.
        """
        return worker is not None and worker.token == self.window._load_token

    def ensure_default_model(self):
        self.window._set_general_status("Preparing default model...")
        self.window._model_btn.setEnabled(False)
        self.window._send_btn.setEnabled(False)

        token = self.new_load_token()
        self.window._download_worker = ModelDownloadWorker(token)
        self.window._download_worker.progress.connect(self.window._set_general_status)
        self.window._download_worker.finished.connect(self.on_default_model_ready)
        self.window._download_worker.error_occurred.connect(self.on_default_model_error)
        self.window._download_worker.start()

    def on_default_model_ready(self, model_path: str):
        worker = self.window._download_worker
        if not self._is_current_worker(worker):
            log("UI", "Stale default-model ready event ignored")
            return
        save_model_path(model_path)
        self.load_model(model_path, token=worker.token)

    def on_default_model_error(self, error: str):
        worker = self.window._download_worker
        if not self._is_current_worker(worker):
            log("UI", "Stale default-model error event ignored")
            return
        log("UI", f"Default model download error: {error}")
        self.window._set_general_status(
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
        # Defense at the source: any caller asking to load the model that's
        # already running should be a no-op. Re-launching llama-server while
        # a parse/chat HTTP request is in flight against it kills the
        # in-flight call and races the new process onto the same port —
        # which has crashed the app (e.g. Back→Continue on the onboarding
        # download screen during a parse cancellation).
        from core.llm_engine import get_current_model_path

        if get_current_model_path() == model_path:
            log("UI", f"Model already loaded — skipping reload: {model_path}")
            return

        # Refuse to swap models while a parse / chat / report / extraction
        # is in flight: stopping llama-server out from under an active
        # request lands its retry on the new (potentially non-vision)
        # model, silently corrupting the result. Force the user to
        # cancel or wait first.
        if self.window.is_llm_busy():
            log("UI", f"Refusing model swap to {model_path}: LLM-using worker is active")
            self.window._set_general_status(
                "Wait for the current operation to finish or cancel it, "
                "then try loading the model again."
            )
            return

        if token is None:
            token = self.new_load_token()
        cached = load_model_meta(model_path) or {}
        if model := approved_model_for_file(model_path):
            display = model["display_name"]
        else:
            display = cached.get("name") or Path(model_path).stem
        self.window._set_general_status(f"Loading model: {display}...")
        self.window._model_btn.setEnabled(False)
        self.window._send_btn.setEnabled(False)

        self.window._server_worker = ServerStartWorker(model_path, token)
        self.window._server_worker.progress.connect(self.window._set_general_status)
        self.window._server_worker.meta_ready.connect(self.on_model_meta)
        self.window._server_worker.finished.connect(self.on_server_started)
        self.window._server_worker.error_occurred.connect(self.on_server_error)
        self.window._server_worker.start()

    def on_model_meta(self, meta):
        worker = self.window._server_worker
        if not self._is_current_worker(worker):
            return
        log("UI", f"Model meta: name={meta.name!r}, ctx={meta.context_length}")
        display = (
            model["display_name"]
            if (model := approved_model_for_file(worker.model_path))
            else meta.name
        )
        if display:
            self.window._set_general_status(f"Loading model: {display}...")

    def on_server_started(self, model_name: str):
        worker = self.window._server_worker
        if not self._is_current_worker(worker):
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
        if not self._is_current_worker(worker):
            log("UI", "Stale server-error event ignored")
            return
        log("UI", f"Server error: {error}")
        # `error` is already the classified user-facing message from
        # llm_server._classify_stderr (e.g. "Not enough memory..."),
        # or the bare RuntimeError text from _wait_for_ready (timeout,
        # unexpected exit) when no stderr pattern matched. Show it
        # as-is — concise and specific to the actual failure.
        self.window._set_general_status(error)
        self.window._model_btn.setEnabled(not self.window._parsing_active)
