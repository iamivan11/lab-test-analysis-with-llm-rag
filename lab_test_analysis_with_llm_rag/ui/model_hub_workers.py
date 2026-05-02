"""Background workers for Model Hub operations."""

import threading

from PySide6.QtCore import QThread, Signal

from config import format_size
from core.logger import log
from core.model_hub import download_model, list_gguf_files, search_models


class SearchWorker(QThread):
    finished = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, query: str):
        super().__init__()
        self.query = query

    def run(self):
        log("WORKER", f"SearchWorker: query='{self.query}'")
        try:
            results = search_models(self.query)
            log("WORKER", f"SearchWorker: found {len(results)} models")
            self.finished.emit(results)
        except Exception as e:
            log("WORKER", f"SearchWorker: ERROR {e}")
            self.error_occurred.emit(str(e))


class FileListWorker(QThread):
    finished = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, model_id: str):
        super().__init__()
        self.model_id = model_id

    def run(self):
        log("WORKER", f"FileListWorker: loading files for {self.model_id}")
        try:
            files = list_gguf_files(self.model_id)
            log("WORKER", f"FileListWorker: found {len(files)} GGUF files")
            self.finished.emit(files)
        except Exception as e:
            log("WORKER", f"FileListWorker: ERROR {e}")
            self.error_occurred.emit(str(e))


class DownloadWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, model_id: str, filename: str):
        super().__init__()
        self.model_id = model_id
        self.filename = filename
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        log("WORKER", f"DownloadWorker: downloading {self.filename} from {self.model_id}")
        try:

            def on_progress(downloaded: int, total: int):
                pct = int(downloaded * 100 / total) if total > 0 else 0
                label = (
                    f"{format_size(downloaded)} / {format_size(total)}"
                    if total
                    else format_size(downloaded)
                )
                self.progress.emit(pct, label)

            path = download_model(
                self.model_id,
                self.filename,
                on_progress=on_progress,
                stop_event=self._stop_event,
            )
            log("WORKER", f"DownloadWorker: done, saved to {path}")
            self.finished.emit(str(path))
        except InterruptedError:
            log("WORKER", "DownloadWorker: cancelled")
            self.error_occurred.emit("Download cancelled")
        except Exception as e:
            log("WORKER", f"DownloadWorker: ERROR {e}")
            self.error_occurred.emit(str(e))
