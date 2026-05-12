"""Background workers used by onboarding screens."""

import threading
from pathlib import Path

from PySide6.QtCore import Signal

from config import KB_EMBEDDING_MODEL
from core.logger import log, log_exception
from core.model_hub import download_model
from core.qthread_utils import StoppableQThread


def _make_aggregating_tqdm(on_downloaded):
    """Return a tqdm subclass that accumulates byte-progress across instances.

    `huggingface_hub.snapshot_download` creates one tqdm per file *and*
    a top-level tqdm tracking file count (unit="it"/"files"). We only
    aggregate the byte-unit instances and report the running total of
    downloaded bytes. The caller pairs this with a precomputed total
    from HF metadata so the displayed denominator is correct from byte
    one, instead of growing as files come online.
    """
    from tqdm.auto import tqdm as _Base

    state = {"downloaded": 0}
    lock = threading.Lock()

    def _is_byte_unit(t: _Base) -> bool:
        unit = (getattr(t, "unit", "") or "").strip().lower()
        return unit in ("b", "byte", "bytes")

    class _AggregatingTqdm(_Base):
        def update(self, n=1):
            result = super().update(n)
            if _is_byte_unit(self):
                with lock:
                    state["downloaded"] += int(n or 0)
                    on_downloaded(state["downloaded"])
            return result

    return _AggregatingTqdm


def _hf_repo_total_size(repo_id: str) -> int:
    """Sum of all sibling file sizes for a HuggingFace repo (bytes).

    Used as a fixed denominator for the embedder progress bar so the
    "downloaded / total" readout is correct from the first byte —
    matching what the model/mmproj rows show via _hf_file_size().
    Returns 0 on any failure; the caller falls back to dynamic
    aggregation in that case.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True)
        total = 0
        for sibling in getattr(info, "siblings", []) or []:
            lfs = getattr(sibling, "lfs", None)
            if lfs is not None and getattr(lfs, "size", None):
                total += int(lfs.size)
                continue
            size = getattr(sibling, "size", None)
            if size:
                total += int(size)
        return total
    except Exception as e:
        log("HUB", f"HF repo size lookup failed for {repo_id}: {e}")
        return 0


class EmbedderPrefetchWorker(StoppableQThread):
    """Pre-downloads the sentence-transformer embedder during onboarding.

    Without this, the model is fetched lazily on the first document
    indexing — which means the user can complete onboarding and start
    uploading docs only to hit a network/permission failure deep into
    parsing. Doing it up-front while we're already on the download
    screen surfaces problems early and removes a runtime stall later.

    Uses `huggingface_hub.snapshot_download` rather than constructing a
    full SentenceTransformer: snapshot_download only fetches files (no
    PyTorch load, no large RAM allocation), is idempotent if the cache
    already has the snapshot, and respects HF_HOME/HF_CACHE_DIR set in
    config.py.

    Per-byte progress is captured via a custom `tqdm_class` injected
    into `snapshot_download` — without it, we'd have to render an
    indeterminate (busy-animation) bar that confuses users about
    whether the download is making any progress."""

    progress = Signal("qint64", "qint64")  # downloaded, total — aggregated across files
    finished = Signal()
    error_occurred = Signal(str)

    def run(self):
        # snapshot_download has no in-flight cancellation hook, so the
        # most we can do is bail out before spawning network work if the
        # user has already cancelled (e.g. closed the app between Qt
        # scheduling this thread and run() actually firing).
        #
        # Contract: `.stop()` on this worker MUST only be called during
        # app shutdown. If a non-shutdown caller invokes stop() and the
        # check below trips, no `finished`/`error_occurred` signal fires
        # and the onboarding screen hangs on the indeterminate progress
        # bar with no recovery path.
        if self._stop_event.is_set():
            log("HUB", "EmbedderPrefetchWorker: stopped before start")
            return
        try:
            from huggingface_hub import snapshot_download

            log("HUB", f"Embedder prefetch starting: {KB_EMBEDDING_MODEL}")
            # Fixed denominator from HF metadata so the "downloaded / total"
            # readout is correct from the first byte. If the API call fails
            # (e.g. offline mode), fall back to passing each emitted
            # downloaded count back as both numerator and denominator —
            # ugly but better than mixing-units garbage.
            total = _hf_repo_total_size(KB_EMBEDDING_MODEL)
            log("HUB", f"Embedder repo total size: {total} bytes")
            if total > 0:
                emit = lambda d: self.progress.emit(d, total)
            else:
                emit = lambda d: self.progress.emit(d, d)
            tqdm_class = _make_aggregating_tqdm(emit)
            # No explicit cache_dir: HF_HOME (set in config.py) controls
            # the path. Forcing it here would risk drifting from where
            # SentenceTransformer later loads the model from.
            snapshot_download(repo_id=KB_EMBEDDING_MODEL, tqdm_class=tqdm_class)
            # Snapshot may have hit cached files (no tqdm updates fired)
            # so the last emitted progress may be < total. Fire a final
            # (total, total) so the bar lands on 100% / "Ready".
            if total > 0:
                self.progress.emit(total, total)
            log("HUB", "Embedder prefetch: done")
            self.finished.emit()
        except Exception as e:
            log_exception("HUB", "EmbedderPrefetchWorker failed")
            self.error_occurred.emit(str(e))


class SingleFileDownloadWorker(StoppableQThread):
    """Downloads ONE file from HuggingFace.

    Used by onboarding to fetch the main GGUF, the mmproj, and (via
    EmbedderPrefetchWorker for the embedder) in parallel — each worker
    reports its own bytes so the screen can render a separate progress
    bar per file. qint64 byte counts because GGUFs exceed 2 GB."""

    progress = Signal("qint64", "qint64")  # downloaded, total
    finished = Signal(str)  # final on-disk path
    error_occurred = Signal(str)

    def __init__(
        self,
        repo_id: str,
        hf_name: str,
        local_name: str,
        download_dir: Path,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.hf_name = hf_name
        self.local_name = local_name
        self.download_dir = download_dir

    def run(self):
        try:
            self.download_dir.mkdir(parents=True, exist_ok=True)
            dest = self.download_dir / self.local_name
            if dest.exists():
                size = dest.stat().st_size
                self.progress.emit(size, size)
                self.finished.emit(str(dest))
                return

            log("HUB", f"Onboarding download: {self.hf_name} -> {self.local_name}")
            download_model(
                self.repo_id,
                self.hf_name,
                on_progress=lambda d, t: self.progress.emit(d, t),
                stop_event=self._stop_event,
                download_dir=self.download_dir,
            )
            if self._stop_event.is_set():
                self.error_occurred.emit("Download cancelled")
                return

            # download_model writes to download_dir/hf_name; rename if
            # the local destination uses a different filename
            # (e.g. shared "mmproj-BF16.gguf" → model-specific local name).
            if self.hf_name != self.local_name:
                src = self.download_dir / self.hf_name
                if src.exists() and src != dest:
                    src.rename(dest)
            self.finished.emit(str(dest))
        except Exception as e:
            log_exception("HUB", f"SingleFileDownloadWorker {self.hf_name} failed")
            self.error_occurred.emit(str(e))
