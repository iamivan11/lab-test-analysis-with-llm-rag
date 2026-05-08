"""Documents content widget — manage uploaded lab test documents."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
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

from config import DOCS_DIR, format_size, save_model_path
from core.document_parser import SUPPORTED_EXTENSIONS
from core.knowledge_base import remove_document
from core.logger import log
from core.security import write_protected_bytes
from ui.documents.workers import (
    FILTERING_OUTPUT_DIR,
    PARSING_OUTPUT_DIR,
    EnsureVisionModelWorker,
    IndexWorker,
)


class DocumentsHubWidget(QWidget):
    """Documents management UI without a host dialog/window chrome."""

    model_swapped = Signal(str, str)
    parsing_active_changed = Signal(bool)
    docs_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._index_worker = None
        self._ensure_worker = None
        self._pending_files: list[Path] = []
        self._current_batch: list[Path] = []
        self._cancelled = False
        self._reindex_active = False
        self._operation_token = 0
        self._table_signature: tuple[object, ...] | None = None
        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        # Public so a host screen can inject extra leading widgets (e.g. a
        # Back button on the Documents-section page).
        self.header_row = QHBoxLayout()
        self.header_row.addStretch()

        self._upload_btn = QPushButton("Upload")
        self._upload_btn.setFixedSize(100, 38)
        self._upload_btn.clicked.connect(self._upload_files)
        self.header_row.addWidget(self._upload_btn)

        self._delete_all_btn = QPushButton("Delete All")
        self._delete_all_btn.setObjectName("stopButton")
        self._delete_all_btn.setFixedSize(100, 38)
        self._delete_all_btn.setEnabled(False)
        self._delete_all_btn.clicked.connect(self._delete_all)
        self.header_row.addWidget(self._delete_all_btn)

        layout.addLayout(self.header_row)

        self._status = QLabel("")
        self._status.setObjectName("statusLabel")
        layout.addWidget(self._status)

        self._progress_widget = QWidget()
        progress_row = QHBoxLayout(self._progress_widget)
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(8)

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%v / %m files")
        progress_row.addWidget(self._progress_bar, stretch=1)

        self._cancel_btn = QPushButton("✕")
        self._cancel_btn.setObjectName("iconSecondary")
        self._cancel_btn.setFixedSize(28, 28)
        self._cancel_btn.setToolTip("Cancel")
        self._cancel_btn.clicked.connect(self._cancel_current)
        progress_row.addWidget(self._cancel_btn)

        self._progress_widget.setVisible(False)
        layout.addWidget(self._progress_widget)

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

    def is_busy(self) -> bool:
        ensure_running = bool(self._ensure_worker and self._ensure_worker.isRunning())
        index_running = bool(self._index_worker and self._index_worker.isRunning())
        return ensure_running or index_running

    def _refresh_list(self):
        self._refresh_source_list()

    def _refresh_source_list(self):
        files = sorted(
            [f for f in DOCS_DIR.iterdir() if f.is_file() and not f.name.startswith(".")],
            key=lambda f: f.name.lower(),
        )
        self._populate_table(files, deletable=True)
        self._status.setText(f"{len(files)} document(s)")
        self._delete_all_btn.setEnabled(len(files) > 0)
        self.docs_changed.emit()

    def _populate_table(self, files: list[Path], deletable: bool):
        signature = (
            deletable,
            tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files),
        )
        if signature == self._table_signature:
            return

        self._table_signature = signature
        self._table.setRowCount(len(files))
        for row, file_path in enumerate(files):
            name_item = QTableWidgetItem(file_path.name)
            self._table.setItem(row, 0, name_item)

            size_item = QTableWidgetItem(format_size(file_path.stat().st_size))
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, 1, size_item)

            if deletable:
                del_btn = QPushButton("−")  # noqa: RUF001 - UI glyph, not arithmetic.
                del_btn.setObjectName("iconSecondary")
                del_btn.setFixedSize(28, 28)
                del_btn.setToolTip("Delete")
                del_btn.clicked.connect(lambda checked, path=file_path: self._delete_file(path))
                self._table.setCellWidget(row, 2, del_btn)
            else:
                self._table.setCellWidget(row, 2, None)

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
            write_protected_bytes(dest, src.read_bytes())
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
        self._operation_token += 1
        token = self._operation_token
        self._reindex_active = False
        self._current_batch = list(file_paths)
        self._upload_btn.setEnabled(False)
        self._delete_all_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setMaximum(len(file_paths))
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setVisible(True)
        self._progress_widget.setVisible(True)
        self._pending_files = file_paths
        self._status.setText("Preparing vision model...")
        self.parsing_active_changed.emit(True)

        self._ensure_worker = EnsureVisionModelWorker()
        self._ensure_worker.progress.connect(
            lambda msg, token=token: self._on_index_progress(token, msg)
        )
        self._ensure_worker.finished.connect(
            lambda model_path, display_name, token=token: self._on_vision_model_ready(
                token, model_path, display_name
            )
        )
        self._ensure_worker.error_occurred.connect(
            lambda error, token=token: self._on_index_error(token, error)
        )
        self._ensure_worker.start()

    def _start_reindexing(self, file_paths: list[Path]):
        if self._index_worker and self._index_worker.isRunning():
            self._status.setText("Indexing already in progress...")
            return
        if self._ensure_worker and self._ensure_worker.isRunning():
            self._status.setText("Vision model is still loading...")
            return

        self._cancelled = False
        self._operation_token += 1
        token = self._operation_token
        self._reindex_active = True
        self._current_batch = []
        self._upload_btn.setEnabled(False)
        self._delete_all_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setMaximum(len(file_paths))
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setVisible(True)
        self._progress_widget.setVisible(True)
        self._status.setText("Reindexing from saved filtered results...")
        self.parsing_active_changed.emit(True)

        self._index_worker = IndexWorker(file_paths, reuse_filtered=True)
        self._wire_index_worker(token)
        self._index_worker.start()

    def reindex_files(self):
        files = sorted(
            [f for f in DOCS_DIR.iterdir() if f.is_file() and not f.name.startswith(".")],
            key=lambda f: f.name.lower(),
        )
        if not files:
            self._status.setText("No documents to reindex")
            return

        missing = [f.name for f in files if not (FILTERING_OUTPUT_DIR / f"{f.stem}.md").exists()]
        if missing:
            self._status.setText(f"Missing filtered results for {len(missing)} document(s)")
            return

        self._start_reindexing(files)

    def _cancel_current(self):
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
        for path in self._current_batch:
            remove_document(path.name)
            path.unlink(missing_ok=True)
            raw_md_path = PARSING_OUTPUT_DIR / f"{path.stem}.md"
            filtered_md_path = FILTERING_OUTPUT_DIR / f"{path.stem}.md"
            raw_md_path.unlink(missing_ok=True)
            filtered_md_path.unlink(missing_ok=True)
            log("DOCS", f"Cleaned up cancelled file: {path.name}")
        self._current_batch = []

    def _finalize_parsing(self, status_text: str):
        self._progress_widget.setVisible(False)
        self._upload_btn.setEnabled(True)
        self._delete_all_btn.setEnabled(True)
        self._reindex_active = False
        self._status.setText(status_text)
        self._refresh_list()
        self.parsing_active_changed.emit(False)

    def _is_current_operation(self, token: int) -> bool:
        if token == self._operation_token:
            return True
        log("DOCS", f"Ignored stale worker signal: token={token}")
        return False

    def _wire_index_worker(self, token: int) -> None:
        self._index_worker.progress.connect(
            lambda msg, token=token: self._on_index_progress(token, msg)
        )
        self._index_worker.file_progress.connect(
            lambda done, total, token=token: self._on_file_progress(token, done, total)
        )
        self._index_worker.finished.connect(
            lambda chunks, cancelled, token=token: self._on_index_done(
                token, chunks, cancelled
            )
        )
        self._index_worker.failed_files.connect(
            lambda failed, token=token: self._on_files_failed(token, failed)
        )
        self._index_worker.error_occurred.connect(
            lambda error, token=token: self._on_index_error(token, error)
        )

    def _on_vision_model_ready(self, token: int, model_path: str, display_name: str):
        if not self._is_current_operation(token):
            return
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
        self._wire_index_worker(token)
        self._index_worker.start()

    def _on_index_progress(self, token: int, msg: str):
        if not self._is_current_operation(token):
            return
        self._status.setText(msg)

    def _on_file_progress(self, token: int, done: int, total: int):
        if not self._is_current_operation(token):
            return
        self._progress_bar.setValue(done)

    def _on_files_failed(self, token: int, failed: list[Path]):
        if not self._is_current_operation(token):
            return
        if self._reindex_active:
            self._refresh_list()
            self._status.setText(f"Reindex failed for {len(failed)} document(s)")
            return
        for path in failed:
            remove_document(path.name)
            path.unlink(missing_ok=True)
            log("DOCS", f"Removed failed file: {path.name}")
        self._refresh_list()

    def _on_index_done(self, token: int, total_chunks: int, cancelled: bool):
        if not self._is_current_operation(token):
            return
        log("DOCS", f"Indexing complete: {total_chunks} chunks stored, cancelled={cancelled}")
        if cancelled:
            self._cleanup_batch()
            self._finalize_parsing("Parsing cancelled — all files dropped")
        else:
            self._current_batch = []
            self._finalize_parsing(f"Indexing complete — {total_chunks} chunks stored")

    def _on_index_error(self, token: int, error: str):
        if not self._is_current_operation(token):
            return
        log("DOCS", f"Indexing error: {error}")
        if self._cancelled:
            self._cleanup_batch()
            self._finalize_parsing("Parsing cancelled")
        else:
            self._current_batch = []
            self._finalize_parsing(f"Error: {error}")

    def _delete_file(self, path: Path):
        log("DOCS", f"Deleting document: {path.name}")
        remove_document(path.name)
        path.unlink(missing_ok=True)
        self._table_signature = None
        self._status.setText(f"Deleted: {path.name}")
        self._refresh_list()

    def _delete_all(self):
        log("DOCS", "Deleting all documents")
        files = [f for f in DOCS_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]
        for file_path in files:
            remove_document(file_path.name)
            file_path.unlink(missing_ok=True)
        self._table_signature = None
        self._status.setText(f"Deleted {len(files)} document(s)")
        self._refresh_list()


__all__ = ["DocumentsHubWidget"]
