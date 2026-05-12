import json
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from PySide6.QtCore import Signal

from config import (
    FILTERING_OUTPUT_DIR,
    PARSING_OUTPUT_DIR,
    approved_model_file_path,
    approved_model_for_file,
    get_default_model,
    load_ctx_size,
    load_model_meta,
    save_model_meta,
)
from core.document_parser import extract_document_metadata, parse_document, set_save_dirs
from core.knowledge_base import index_document, remove_document
from core.llm_engine import SERVER_URL, get_current_model_path, is_server_running, start_server
from core.logger import log, log_exception
from core.model_hub import ensure_default_model
from core.model_meta import read_model_meta
from core.qthread_utils import StoppableQThread
from core.security import (
    read_protected_bytes,
    read_protected_json,
    read_protected_text,
    write_protected_json,
)


def format_eta(seconds: float) -> str:
    total_seconds = int(seconds)
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, rem_seconds = divmod(total_seconds, 60)
    return f"{minutes}m {rem_seconds:02d}s"


def _metadata_cache_path(path: Path) -> Path:
    return FILTERING_OUTPUT_DIR / f"{path.stem}.meta.json"


_METADATA_KEYS = ("report_date", "report_type")


def _normalize_metadata(raw: object) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    return {k: str(raw.get(k, "")).strip() for k in _METADATA_KEYS}


def _load_cached_metadata(path: Path) -> dict[str, str] | None:
    cache_path = _metadata_cache_path(path)
    try:
        return _normalize_metadata(read_protected_json(cache_path))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        log("WORKER", f"IndexWorker: metadata cache unreadable for {path.name}: {e}")
        return None


def _save_cached_metadata(path: Path, metadata: dict[str, str]) -> None:
    try:
        write_protected_json(_metadata_cache_path(path), metadata)
    except OSError as e:
        log("WORKER", f"IndexWorker: failed to cache metadata for {path.name}: {e}")


def _extract_or_load_metadata(
    path: Path,
    markdown: str,
    *,
    reuse_filtered: bool,
    stop_event=None,
) -> dict[str, str]:
    if reuse_filtered:
        cached = _load_cached_metadata(path)
        if cached is not None:
            log("WORKER", f"IndexWorker: loaded cached metadata for {path.name}")
            return cached

    try:
        metadata = extract_document_metadata(
            markdown, server_url=SERVER_URL, stop_event=stop_event
        )
    except InterruptedError:
        raise
    except Exception as e:
        log("WORKER", f"IndexWorker: metadata extraction failed for {path.name}: {e}")
        return _normalize_metadata({})

    _save_cached_metadata(path, metadata)
    return metadata


@contextmanager
def _readable_source_document(path: Path):
    with tempfile.TemporaryDirectory(prefix="lab-analyzer-doc-") as tmp_dir:
        readable_path = Path(tmp_dir) / path.name
        readable_path.write_bytes(read_protected_bytes(path))
        yield readable_path


class EnsureVisionModelWorker(StoppableQThread):
    progress = Signal(str)
    finished = Signal(str, str)
    error_occurred = Signal(str)

    def _display_name(self, model_path: str) -> str:
        if model := approved_model_for_file(model_path):
            return model["display_name"]
        try:
            meta = read_model_meta(model_path)
            save_model_meta(
                model_path,
                {"name": meta.name, "context_length": meta.context_length},
            )
            return meta.name or Path(model_path).stem
        except Exception as e:
            log("DOCS", f"read_model_meta failed for {model_path}: {e}")
            return Path(model_path).stem

    def run(self):
        try:
            default_model_path = approved_model_file_path(get_default_model())
            current = get_current_model_path()
            already_loaded = (
                current is not None
                and Path(current) == default_model_path
                and is_server_running()
            )
            if already_loaded:
                log("DOCS", "Default vision model already loaded, skipping swap")
                name = self._display_name(str(default_model_path))
                self.finished.emit(str(default_model_path), name)
                return

            self.progress.emit("Loading vision model for document parsing...")
            model_path = ensure_default_model(
                on_progress=self.progress.emit,
                stop_event=self._stop_event,
            )
            if self._stop_event.is_set():
                log("DOCS", "EnsureVisionModelWorker: cancelled before server start")
                self.finished.emit("", "")
                return
            name = self._display_name(model_path)
            # Per-model ctx setting (same scheme as chat). First load of a
            # given vision model defaults to its own max context.
            cached_meta = load_model_meta(model_path) or {}
            model_max = cached_meta.get("context_length")
            n_ctx = load_ctx_size(model_path) or model_max or 8192
            start_server(model_path, n_ctx=n_ctx, on_progress=self.progress.emit)
            log("DOCS", f"Vision model ready: {model_path} ({name})")
            self.finished.emit(model_path, name)
        except Exception as e:
            if self._stop_event.is_set():
                log("DOCS", f"EnsureVisionModelWorker: cancelled via exception ({e})")
                # The host (IndexWorker pipeline) treats an empty
                # model_path as "user cancelled" — same path as the
                # explicit cancel check above.
                self.finished.emit("", "")
                return
            log_exception("DOCS", "EnsureVisionModelWorker failed")
            self.error_occurred.emit(str(e))


class IndexWorker(StoppableQThread):
    progress = Signal(str)
    file_progress = Signal(int, int)
    finished = Signal(int, bool)
    failed_files = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, file_paths: list[Path], *, reuse_filtered: bool = False):
        super().__init__()
        self.file_paths = file_paths
        self.reuse_filtered = reuse_filtered

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self):
        total_files = len(self.file_paths)
        mode = "reindex" if self.reuse_filtered else "index"
        log("WORKER", f"IndexWorker: starting {mode}, {total_files} files")

        if not self.reuse_filtered:
            set_save_dirs(PARSING_OUTPUT_DIR, FILTERING_OUTPUT_DIR)

        try:
            failed: list[Path] = []
            indexed: list[Path] = []
            total_chunks = 0
            start_time = time.monotonic()

            for i, path in enumerate(self.file_paths):
                if self.is_stopped():
                    log("WORKER", "IndexWorker: cancel detected")
                    break

                elapsed = time.monotonic() - start_time
                eta = ""
                if i > 0:
                    avg_per_file = elapsed / i
                    remaining = avg_per_file * (total_files - i)
                    eta = f" — ~{format_eta(remaining)} remaining"

                action = "Reindexing" if self.reuse_filtered else "Parsing"
                self.progress.emit(f"{action} {path.name} ({i + 1}/{total_files}){eta}")
                self.file_progress.emit(i, total_files)

                try:
                    if self.reuse_filtered:
                        filtered_path = FILTERING_OUTPUT_DIR / f"{path.stem}.md"
                        markdown = read_protected_text(filtered_path)
                    else:
                        with _readable_source_document(path) as readable_path:
                            markdown = parse_document(
                                str(readable_path),
                                server_url=SERVER_URL,
                                stop_event=self._stop_event,
                            )
                    metadata = _extract_or_load_metadata(
                        path,
                        markdown,
                        reuse_filtered=self.reuse_filtered,
                        stop_event=self._stop_event,
                    )
                    log("WORKER", f"IndexWorker: parsed {path.name}, {len(markdown)} chars")
                except InterruptedError:
                    log("WORKER", f"IndexWorker: cancelled mid-parse on {path.name}")
                    break
                except Exception as e:
                    # Cancellation can also surface as a generic exception
                    # (e.g. an httpx error if the cancel watcher won the
                    # race). Always recheck stop_event before treating a
                    # failure as "this file's fault" — otherwise the UI
                    # gets stuck on "Cancelling..." until every remaining
                    # file is also attempted and fails.
                    if self.is_stopped():
                        log("WORKER", f"IndexWorker: cancelled during {path.name} ({e})")
                        break
                    log("WORKER", f"IndexWorker: FAILED to parse {path.name}: {e}")
                    failed.append(path)
                    continue

                if self.is_stopped():
                    log("WORKER", f"IndexWorker: cancel detected before indexing {path.name}")
                    break

                self.progress.emit(f"Indexing {path.name} ({i + 1}/{total_files})...")
                try:
                    chunks = index_document(
                        filename=path.name,
                        markdown_text=markdown,
                        report_date=metadata["report_date"],
                        report_type=metadata["report_type"],
                        on_progress=self.progress.emit,
                    )
                    total_chunks += chunks
                    indexed.append(path)
                    log("WORKER", f"IndexWorker: indexed {path.name}, {chunks} chunks")
                except Exception as e:
                    if self.is_stopped():
                        log("WORKER", f"IndexWorker: cancelled during index of {path.name} ({e})")
                        break
                    log("WORKER", f"IndexWorker: FAILED to index {path.name}: {e}")
                    failed.append(path)

            if self.is_stopped() and indexed:
                for path in indexed:
                    try:
                        remove_document(path.name)
                        log("WORKER", f"IndexWorker: removed cancelled chunks for {path.name}")
                    except Exception as e:
                        log(
                            "WORKER",
                            f"IndexWorker: failed to remove cancelled chunks for "
                            f"{path.name}: {e}",
                        )
                total_chunks = 0

            self.file_progress.emit(total_files, total_files)
            total_elapsed = format_eta(time.monotonic() - start_time)
            log(
                "WORKER",
                f"IndexWorker: done, {total_chunks} chunks, "
                f"{len(failed)} failed in {total_elapsed}, "
                f"cancelled={self.is_stopped()}",
            )

            if failed and not self.is_stopped():
                self.failed_files.emit(failed)
            self.finished.emit(total_chunks, self.is_stopped())
        except Exception as e:
            log_exception("WORKER", "IndexWorker failed")
            self.error_occurred.emit(str(e))
        finally:
            # Suppress cleanup-time failures so they can't mask the real
            # exception we'd otherwise propagate through error_occurred —
            # an uncaught raise in `finally` shadows the original error
            # and leaves the UI waiting on a signal that never fires.
            if not self.reuse_filtered:
                try:
                    set_save_dirs(None, None)
                except Exception as e:
                    log("WORKER", f"IndexWorker: cleanup failed (suppressed): {e}")
