"""Background workers for the Models section.

Three workers live here for the three asynchronous things this section
can do: start (or swap) the llama-server with a given model file,
fetch the default model when nothing's installed yet, and download a
new model on demand from the Models tab.
"""

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from config import approved_model_for_file, format_size
from core.llm_engine import start_server
from core.llm_server import stop_server
from core.logger import log, log_exception
from core.model_hub import download_model, ensure_default_model
from core.qthread_utils import StoppableQThread


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
            log_exception("WORKER", "ServerStartWorker failed")
            self.error_occurred.emit(str(e))


class ModelDownloadWorker(StoppableQThread):
    progress = Signal(str)
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, token: int):
        super().__init__()
        self.token = token

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
            log_exception("WORKER", "ModelDownloadWorker failed")
            self.error_occurred.emit(str(e))


class DownloadWorker(StoppableQThread):
    progress = Signal(int, str)
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, model_id: str, hf_name: str, local_name: str):
        super().__init__()
        self.model_id = model_id
        self.hf_name = hf_name
        self.local_path = Path(local_name)
        self.local_name = self.local_path.name

    def run(self):
        log("WORKER", f"DownloadWorker: downloading {self.hf_name} from {self.model_id}")
        try:
            def on_progress(downloaded: int, total: int):
                pct = int(downloaded * 100 / total) if total > 0 else 0
                label = (
                    f"{self.local_name}: {format_size(downloaded)} / {format_size(total)}"
                    if total
                    else f"{self.local_name}: {format_size(downloaded)}"
                )
                self.progress.emit(pct, label)

            path = download_model(
                self.model_id,
                self.hf_name,
                on_progress=on_progress,
                stop_event=self._stop_event,
                download_dir=self.local_path.parent,
            )
            dest = self.local_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if path != dest and path.exists():
                path.rename(dest)
                path = dest
            self.progress.emit(100, f"Done: {self.local_name}")
            log("WORKER", f"DownloadWorker: done, saved to {path}")
            self.finished.emit(str(path))
        except InterruptedError:
            log("WORKER", "DownloadWorker: cancelled")
            self.error_occurred.emit("Download cancelled")
        except Exception as e:
            log_exception("WORKER", "DownloadWorker failed")
            self.error_occurred.emit(str(e))


__all__ = ["DownloadWorker", "ModelDownloadWorker", "ServerStartWorker"]
