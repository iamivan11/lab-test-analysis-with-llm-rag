"""Dialog for managing uploaded lab test documents."""

import shutil
import threading
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.config import (
    DEFAULT_MODEL_FILE,
    DOCS_DIR,
    MODELS_DIR,
    format_size,
    load_ctx_size,
    save_model_meta,
    save_model_path,
)
from app.core.document_parser import SUPPORTED_EXTENSIONS, parse_document, set_save_dir
from app.core.knowledge_base import (
    index_document,
    remove_document,
)
from app.core.llm_engine import get_current_model_path, is_server_running, start_server
from app.core.logger import log
from app.core.model_hub import ensure_default_model
from app.core.model_meta import read_model_meta
from app.ui.styles import STYLESHEET

_VISION_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tmp" / "vision_output"
_MASTER_DIR = _VISION_OUTPUT_DIR / "master"


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    return f"{m}m {s:02d}s"


# ── Workers ─────────────────────────────────────────────────────────────


class EnsureVisionModelWorker(QThread):
    """Ensure the default Qwen3.5-9B (vision-capable) model is loaded.

    The default model includes an mmproj file, enabling the vision path used by
    `document_parser`. If a non-default model is currently loaded we swap to it;
    otherwise this is a no-op.
    """

    progress = Signal(str)
    finished = Signal(str, str)  # (model_path, display_name)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _display_name(self, model_path: str) -> str:
        """Prefer the GGUF metadata name, fall back to the filename stem."""
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
            default_model_path = MODELS_DIR / DEFAULT_MODEL_FILE
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
            n_ctx = load_ctx_size() or 32768
            start_server(model_path, n_ctx=n_ctx, on_progress=self.progress.emit)
            log("DOCS", f"Vision model ready: {model_path} ({name})")
            self.finished.emit(model_path, name)
        except Exception as e:
            log("DOCS", f"EnsureVisionModelWorker: ERROR {e}")
            self.error_occurred.emit(str(e))


class IndexWorker(QThread):
    """Background worker: phase 1 parses all files, phase 2 indexes them.

    Supports cooperative cancellation via `stop()`. The in-flight vision
    request cannot be interrupted mid-call, but no further files will be
    parsed or indexed once the stop flag is set.
    """

    progress = Signal(str)
    file_progress = Signal(int, int)
    finished = Signal(int, bool)  # (total_chunks, cancelled)
    failed_files = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, file_paths: list[Path]):
        super().__init__()
        self.file_paths = file_paths
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self):
        total_files = len(self.file_paths)
        log("WORKER", f"IndexWorker: starting, {total_files} files")

        set_save_dir(_VISION_OUTPUT_DIR)

        try:
            # Phase 1: Parse all files via vision LLM
            parsed: dict[Path, str] = {}
            failed: list[Path] = []
            start_time = time.monotonic()

            for i, path in enumerate(self.file_paths):
                if self.is_stopped():
                    log("WORKER", "IndexWorker: cancel detected, stopping parse phase")
                    break

                elapsed = time.monotonic() - start_time
                eta = ""
                if i > 0:
                    avg_per_file = elapsed / i
                    remaining = avg_per_file * (total_files - i)
                    eta = f" — ~{_fmt_time(remaining)} remaining"

                self.progress.emit(f"Parsing {path.name} ({i + 1}/{total_files}){eta}")
                self.file_progress.emit(i, total_files)

                try:
                    markdown = parse_document(str(path))
                    parsed[path] = markdown
                    log("WORKER", f"IndexWorker: parsed {path.name}, {len(markdown)} chars")
                except Exception as e:
                    log("WORKER", f"IndexWorker: FAILED to parse {path.name}: {e}")
                    failed.append(path)

            parse_elapsed = _fmt_time(time.monotonic() - start_time)
            log(
                "WORKER",
                f"IndexWorker: parsing done in {parse_elapsed}, "
                f"{len(parsed)} OK, {len(failed)} failed, "
                f"cancelled={self.is_stopped()}",
            )

            # Phase 2: Index all successfully parsed files (skip entirely on cancel)
            total_chunks = 0
            if not self.is_stopped():
                for i, (path, markdown) in enumerate(parsed.items()):
                    if self.is_stopped():
                        log("WORKER", "IndexWorker: cancel detected, stopping index phase")
                        break
                    self.progress.emit(f"Indexing {path.name} ({i + 1}/{len(parsed)})...")
                    try:
                        chunks = index_document(
                            filename=path.name,
                            markdown_text=markdown,
                            on_progress=self.progress.emit,
                        )
                        total_chunks += chunks
                        log("WORKER", f"IndexWorker: indexed {path.name}, {chunks} chunks")
                    except Exception as e:
                        log("WORKER", f"IndexWorker: FAILED to index {path.name}: {e}")
                        failed.append(path)

            self.file_progress.emit(total_files, total_files)
            total_elapsed = _fmt_time(time.monotonic() - start_time)
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
            log("WORKER", f"IndexWorker: ERROR {e}")
            self.error_occurred.emit(str(e))
        finally:
            set_save_dir(None)


class OrganizeWorker(QThread):
    """Background worker that classifies and synthesizes documents into master files."""

    progress = Signal(str)
    finished = Signal(int, bool)  # (count, cancelled)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self):
        try:
            from app.core.organizer import organize_documents

            count = organize_documents(
                _VISION_OUTPUT_DIR,
                _MASTER_DIR,
                on_progress=self.progress.emit,
                stop_event=self._stop_event,
            )
            self.finished.emit(count, self.is_stopped())
        except Exception as e:
            log("ORGANIZER", f"OrganizeWorker error: {e}")
            self.error_occurred.emit(str(e))


# ── Dialog ──────────────────────────────────────────────────────────────


class DocumentsHubDialog(QDialog):
    model_swapped = Signal(str, str)  # (model_path, display_name)
    parsing_active_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Documents")
        self.setMinimumSize(600, 400)
        self.setStyleSheet(STYLESHEET)
        self._index_worker = None
        self._organize_worker = None
        self._ensure_worker = None
        self._pending_files: list[Path] = []
        self._current_batch: list[Path] = []
        self._cancelled = False
        self._organize_cancelled = False
        self._showing_master = False
        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header row
        header_row = QHBoxLayout()

        self._compile_btn = QPushButton("Compile")
        self._compile_btn.setFixedSize(100, 38)
        self._compile_btn.clicked.connect(self._start_organize)
        header_row.addWidget(self._compile_btn)

        self._view_toggle = QPushButton("Master Files")
        self._view_toggle.setObjectName("attachButton")
        self._view_toggle.setFixedSize(120, 38)
        self._view_toggle.clicked.connect(self._toggle_view)
        self._view_toggle.setVisible(False)
        header_row.addWidget(self._view_toggle)

        header_row.addStretch()

        self._upload_btn = QPushButton("Upload")
        self._upload_btn.setFixedSize(100, 38)
        self._upload_btn.clicked.connect(self._upload_files)
        header_row.addWidget(self._upload_btn)

        self._delete_all_btn = QPushButton("Delete All")
        self._delete_all_btn.setObjectName("attachButton")
        self._delete_all_btn.setFixedSize(100, 38)
        self._delete_all_btn.clicked.connect(self._delete_all)
        header_row.addWidget(self._delete_all_btn)

        layout.addLayout(header_row)

        # Status label
        self._status = QLabel("")
        self._status.setObjectName("statusLabel")
        layout.addWidget(self._status)

        # Progress row: progress bar + cancel button (hidden until work starts)
        self._progress_widget = QWidget()
        progress_row = QHBoxLayout(self._progress_widget)
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(8)

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%v / %m files")
        progress_row.addWidget(self._progress_bar, stretch=1)

        self._cancel_btn = QPushButton("\u2715")
        self._cancel_btn.setObjectName("iconSecondary")
        self._cancel_btn.setFixedSize(28, 28)
        self._cancel_btn.setToolTip("Cancel")
        self._cancel_btn.clicked.connect(self._cancel_current)
        progress_row.addWidget(self._cancel_btn)

        self._progress_widget.setVisible(False)
        layout.addWidget(self._progress_widget)

        # Documents table: Document | Size | Delete
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Document", "Size", ""])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._table, stretch=1)

        # Bottom row
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setObjectName("attachButton")
        close_btn.setFixedSize(100, 38)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # ── View toggle ─────────────────────────────────────────────────────

    def _toggle_view(self):
        self._showing_master = not self._showing_master
        self._view_toggle.setText("Sources" if self._showing_master else "Master Files")
        # Upload/Delete All only relevant in sources view
        self._upload_btn.setVisible(not self._showing_master)
        self._delete_all_btn.setVisible(not self._showing_master)
        self._compile_btn.setVisible(not self._showing_master)
        self._refresh_list()

    def _refresh_list(self):
        if self._showing_master:
            self._refresh_master_list()
        else:
            self._refresh_source_list()

        # Show toggle whenever master files exist
        has_masters = _MASTER_DIR.is_dir() and any(_MASTER_DIR.glob("master_*.md"))
        self._view_toggle.setVisible(has_masters)

    def _refresh_source_list(self):
        files = sorted(
            [f for f in DOCS_DIR.iterdir() if f.is_file() and not f.name.startswith(".")],
            key=lambda f: f.name.lower(),
        )
        self._populate_table(files, deletable=True)
        self._status.setText(f"{len(files)} document(s)")
        self._delete_all_btn.setEnabled(len(files) > 0)

    def _refresh_master_list(self):
        if not _MASTER_DIR.is_dir():
            files = []
        else:
            files = sorted(_MASTER_DIR.glob("master_*.md"), key=lambda f: f.name.lower())
        self._populate_table(list(files), deletable=False)
        self._status.setText(f"{len(files)} master file(s)")

    def _populate_table(self, files: list[Path], deletable: bool):
        self._table.setRowCount(len(files))
        for row, f in enumerate(files):
            name_item = QTableWidgetItem(f.name)
            self._table.setItem(row, 0, name_item)

            size_item = QTableWidgetItem(format_size(f.stat().st_size))
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, 1, size_item)

            if deletable:
                del_btn = QPushButton("\u2212")
                del_btn.setObjectName("iconSecondary")
                del_btn.setFixedSize(28, 28)
                del_btn.setToolTip("Delete")
                del_btn.clicked.connect(lambda checked, path=f: self._delete_file(path))
                self._table.setCellWidget(row, 2, del_btn)
            else:
                self._table.setCellWidget(row, 2, None)

    # ── Upload & index ───────────────────────────────────────────────────

    def _upload_files(self):
        log("DOCS", "Opening upload file dialog")
        ext_filter = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Lab Test Documents",
            "",
            f"Supported Files ({ext_filter});;All Files (*)",
        )
        if not file_paths:
            return

        new_files = []
        for src_path in file_paths:
            src = Path(src_path)
            dest = DOCS_DIR / src.name
            if dest.exists():
                continue
            shutil.copy2(src, dest)
            new_files.append(dest)

        if new_files:
            log("DOCS", f"Uploaded {len(new_files)} new files: {[f.name for f in new_files]}")
            self._start_indexing(new_files)

    def _start_indexing(self, file_paths: list[Path]):
        if self._index_worker and self._index_worker.isRunning():
            self._status.setText("Indexing already in progress...")
            return
        if self._ensure_worker and self._ensure_worker.isRunning():
            self._status.setText("Vision model is still loading...")
            return

        self._cancelled = False
        self._current_batch = list(file_paths)

        self._upload_btn.setEnabled(False)
        self._delete_all_btn.setEnabled(False)
        self._compile_btn.setEnabled(False)

        self._progress_bar.setValue(0)
        self._progress_bar.setMaximum(len(file_paths))
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setVisible(True)
        self._progress_widget.setVisible(True)

        self._pending_files = file_paths
        self._status.setText("Preparing vision model...")

        self.parsing_active_changed.emit(True)

        self._ensure_worker = EnsureVisionModelWorker()
        self._ensure_worker.progress.connect(self._on_index_progress)
        self._ensure_worker.finished.connect(self._on_vision_model_ready)
        self._ensure_worker.error_occurred.connect(self._on_index_error)
        self._ensure_worker.start()

    def _cancel_current(self):
        """Dispatch cancel to whichever worker is currently running."""
        if self._organize_worker and self._organize_worker.isRunning():
            if self._organize_cancelled:
                return
            log("DOCS", "User cancelled compile")
            self._organize_cancelled = True
            self._cancel_btn.setEnabled(False)
            self._status.setText("Cancelling compilation...")
            self._organize_worker.stop()
            return

        if self._cancelled:
            return
        log("DOCS", "User cancelled parsing")
        self._cancelled = True
        self._cancel_btn.setEnabled(False)
        self._status.setText("Cancelling — waiting for current page to finish...")
        if self._ensure_worker and self._ensure_worker.isRunning():
            self._ensure_worker.stop()
        if self._index_worker and self._index_worker.isRunning():
            self._index_worker.stop()

    def _cleanup_batch(self):
        """Drop every file in the cancelled batch: DOCS_DIR copy, RAG entry, parsed .md."""
        for path in self._current_batch:
            remove_document(path.name)
            path.unlink(missing_ok=True)
            md_path = _VISION_OUTPUT_DIR / f"{path.stem}.md"
            md_path.unlink(missing_ok=True)
            log("DOCS", f"Cleaned up cancelled file: {path.name}")
        self._current_batch = []

    def _finalize_parsing(self, status_text: str):
        self._progress_widget.setVisible(False)
        self._upload_btn.setEnabled(True)
        self._delete_all_btn.setEnabled(True)
        self._compile_btn.setEnabled(True)
        self._status.setText(status_text)
        self._refresh_list()
        self.parsing_active_changed.emit(False)

    def _on_vision_model_ready(self, model_path: str, display_name: str):
        if model_path and Path(model_path).exists():
            save_model_path(model_path)
            self.model_swapped.emit(model_path, display_name)

        file_paths = self._pending_files
        self._pending_files = []

        if self._cancelled:
            self._cleanup_batch()
            self._finalize_parsing("Parsing cancelled")
            return

        if not file_paths:
            self._finalize_parsing("")
            return

        self._index_worker = IndexWorker(file_paths)
        self._index_worker.progress.connect(self._on_index_progress)
        self._index_worker.file_progress.connect(self._on_file_progress)
        self._index_worker.finished.connect(self._on_index_done)
        self._index_worker.failed_files.connect(self._on_files_failed)
        self._index_worker.error_occurred.connect(self._on_index_error)
        self._index_worker.start()

    def _on_index_progress(self, msg: str):
        self._status.setText(msg)

    def _on_file_progress(self, done: int, total: int):
        self._progress_bar.setValue(done)

    def _on_files_failed(self, failed: list[Path]):
        for path in failed:
            remove_document(path.name)
            path.unlink(missing_ok=True)
            log("DOCS", f"Removed failed file: {path.name}")
        self._refresh_list()

    def _on_index_done(self, total_chunks: int, cancelled: bool):
        log(
            "DOCS",
            f"Indexing complete: {total_chunks} chunks stored, cancelled={cancelled}",
        )
        if cancelled:
            self._cleanup_batch()
            self._finalize_parsing("Parsing cancelled — all files dropped")
        else:
            self._current_batch = []
            self._finalize_parsing(f"Indexing complete — {total_chunks} chunks stored")

    def _on_index_error(self, error: str):
        log("DOCS", f"Indexing error: {error}")
        if self._cancelled:
            self._cleanup_batch()
            self._finalize_parsing("Parsing cancelled")
        else:
            self._current_batch = []
            self._finalize_parsing(f"Error: {error}")

    # ── Compile (organize) ───────────────────────────────────────────────

    def _start_organize(self):
        if not _VISION_OUTPUT_DIR.is_dir() or not any(_VISION_OUTPUT_DIR.glob("*.md")):
            self._status.setText("No parsed documents found. Upload and process documents first.")
            return

        if self._organize_worker and self._organize_worker.isRunning():
            self._status.setText("Compilation already in progress...")
            return

        self._organize_cancelled = False
        self._compile_btn.setEnabled(False)
        self._upload_btn.setEnabled(False)
        self._delete_all_btn.setEnabled(False)
        self._progress_bar.setMaximum(0)  # indeterminate
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setVisible(True)
        self._progress_widget.setVisible(True)
        self._status.setText("Starting compilation...")

        self._organize_worker = OrganizeWorker()
        self._organize_worker.progress.connect(self._on_compile_progress)
        self._organize_worker.finished.connect(self._on_compile_done)
        self._organize_worker.error_occurred.connect(self._on_compile_error)
        self._organize_worker.start()

    def _on_compile_progress(self, msg: str):
        self._status.setText(msg)

    def _on_compile_done(self, count: int, cancelled: bool):
        log("DOCS", f"Compilation finished: {count} master files, cancelled={cancelled}")
        self._progress_bar.setMaximum(100)
        self._progress_widget.setVisible(False)
        self._compile_btn.setEnabled(True)
        self._upload_btn.setEnabled(True)
        self._delete_all_btn.setEnabled(True)
        if cancelled:
            self._status.setText(f"Compilation cancelled — {count} master files created")
        else:
            self._status.setText(f"Compilation complete — {count} master files created")
        # Switch to master files view automatically if any files were created
        if count > 0 and not self._showing_master:
            self._showing_master = True
            self._view_toggle.setText("Sources")
            self._upload_btn.setVisible(False)
            self._delete_all_btn.setVisible(False)
            self._compile_btn.setVisible(False)
        self._refresh_list()

    def _on_compile_error(self, error: str):
        log("DOCS", f"Compilation error: {error}")
        self._progress_bar.setMaximum(100)
        self._progress_widget.setVisible(False)
        self._compile_btn.setEnabled(True)
        self._upload_btn.setEnabled(True)
        self._delete_all_btn.setEnabled(True)
        self._status.setText(f"Error: {error}")

    # ── Delete ───────────────────────────────────────────────────────────

    def _delete_file(self, path: Path):
        log("DOCS", f"Deleting document: {path.name}")
        remove_document(path.name)
        path.unlink(missing_ok=True)
        self._status.setText(f"Deleted: {path.name}")
        self._refresh_list()

    def _delete_all(self):
        log("DOCS", "Deleting all documents")
        files = [f for f in DOCS_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]
        for f in files:
            remove_document(f.name)
            f.unlink(missing_ok=True)
        self._status.setText(f"Deleted {len(files)} document(s)")
        self._refresh_list()
