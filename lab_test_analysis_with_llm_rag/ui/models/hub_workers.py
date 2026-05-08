"""Background workers for Model Hub operations."""

import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from config import format_size
from core.logger import log
from core.model_hub import download_model


class DownloadWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, model_id: str, hf_name: str, local_name: str):
        super().__init__()
        self.model_id = model_id
        self.hf_name = hf_name
        self.local_path = Path(local_name)
        self.local_name = self.local_path.name
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

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
            log("WORKER", f"DownloadWorker: ERROR {e}")
            self.error_occurred.emit(str(e))
