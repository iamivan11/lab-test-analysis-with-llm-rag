"""Background workers used by onboarding screens."""

import threading

from PySide6.QtCore import QThread, Signal

from config import (
    approved_model_file_path,
    get_default_model,
)
from core.logger import log
from core.model_hub import download_model


class DefaultModelDownloadWorker(QThread):
    progress = Signal(int, int, str)
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            model = get_default_model()
            files = [
                (model["model_file"], model["model_file"]),
                (model["mmproj_file"], model["mmproj_local"]),
            ]
            model_path = approved_model_file_path(model)

            done_bytes_prior = 0
            current_total_estimate = 0

            for hf_name, local_name in files:
                dest = approved_model_file_path(model, local_name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    size = dest.stat().st_size
                    done_bytes_prior += size
                    current_total_estimate += size
                    self.progress.emit(done_bytes_prior, current_total_estimate, local_name)
                    continue

                file_total_holder = {"total": 0}

                def on_progress(
                    downloaded: int,
                    total: int,
                    _name=local_name,
                    _done_bytes_prior=done_bytes_prior,
                    _file_total_holder=file_total_holder,
                ):
                    if total and _file_total_holder["total"] != total:
                        _file_total_holder["total"] = total
                    estimated_total = _done_bytes_prior + max(
                        _file_total_holder["total"], downloaded
                    )
                    self.progress.emit(
                        _done_bytes_prior + downloaded, estimated_total, _name
                    )

                log("HUB", f"Onboarding download: {hf_name} -> {local_name}")
                download_model(
                    model["repo_id"],
                    hf_name,
                    on_progress=on_progress,
                    stop_event=self._stop_event,
                    download_dir=dest.parent,
                )
                if self._stop_event.is_set():
                    self.error_occurred.emit("Download cancelled")
                    return

                if hf_name != local_name:
                    src = dest.parent / hf_name
                    if src.exists():
                        src.rename(dest)

                size = dest.stat().st_size if dest.exists() else file_total_holder["total"]
                done_bytes_prior += size

            self.finished.emit(str(model_path))
        except Exception as e:
            log("HUB", f"DefaultModelDownloadWorker: ERROR {e}")
            self.error_occurred.emit(str(e))
