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

from config import DOCS_DIR, format_size, list_uploaded_doc_paths, save_model_path
from core.document_parser import SUPPORTED_EXTENSIONS
from core.knowledge_base import list_indexed_documents
from core.logger import log
from core.user_data import purge_document_artifacts
from core.messages import (
    DOCS_COULD_NOT_DELETE_SINGLE,
    DOCS_COULD_NOT_REMOVE_BATCH,
    INDEXING_FAILED,
    REINDEX_FAILED_BATCH,
    REINDEX_FAILED_SINGLE,
)
from core.security import write_protected_bytes


def _safe_unlink(path: Path) -> tuple[bool, str | None]:
    """Try to delete `path`. Returns (success, error_message_or_None).

    Surfacing the OS error is critical: an earlier silent
    `path.unlink(missing_ok=True)` would let permission/ENOENT failures
    pass while the UI cheerfully claimed "Deleted: X" — leaving the
    document permanently in the list.
    """
    try:
        path.unlink()
    except FileNotFoundError:
        return True, None
    except OSError as e:
        log("DOCS", f"Failed to delete {path}: {e}")
        return False, str(e)
    return True, None
from ui.components import icon_button
from ui.documents.workers import (
    FILTERING_OUTPUT_DIR,
    EnsureVisionModelWorker,
    IndexWorker,
)


class DocumentsHubWidget(QWidget):
    """Documents management UI without a host dialog/window chrome."""

    model_swapped = Signal(str, str)
    parsing_active_changed = Signal(bool)
    docs_changed = Signal()
    # (status_text, is_active) — fired at every reindex state transition
    # so peer UIs (Settings → Reindex button) can mirror progress without
    # the user having to navigate to the Documents tab to know what's
    # happening.
    reindex_status_changed = Signal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Injected by main_window so document mutations (delete, delete
        # all) can refuse while another LLM-using worker is reading the
        # filtered text / chunks of these docs — Trends extraction and
        # Health Report iterate over uploaded docs, and chat retrieval
        # queries chunks from chromadb.
        from collections.abc import Callable

        self.busy_check: Callable[[], bool] | None = None
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

        self._cancel_btn = icon_button(
            "✕", tooltip="Cancel", on_click=self._cancel_current
        )
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
        # Show only documents that are actually in the vector DB. Files
        # that are on disk but mid-parse haven't been indexed yet —
        # surfacing them as if they were ready misleads the user (the
        # status would say "30 document(s)" while none are searchable).
        indexed = set(list_indexed_documents())
        files = [p for p in list_uploaded_doc_paths() if p.name in indexed]
        self._populate_table(files, deletable=True)
        # Don't clobber the parsing-progress status when an indexing or
        # vision-model-load operation is in flight. Tab navigation away
        # and back would otherwise replace "Parsing P-001..." with
        # "30 document(s)" — confusing while a bar is still moving.
        if not self.is_busy():
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
                del_btn = icon_button(
                    "−",  # noqa: RUF001 - UI glyph, not arithmetic.
                    tooltip="Delete",
                    on_click=lambda path=file_path: self._delete_file(path),
                )
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
        self.reindex_status_changed.emit(
            f"Reindexing {len(file_paths)} document(s)...", True
        )

        self._index_worker = IndexWorker(file_paths, reuse_filtered=True)
        self._wire_index_worker(token)
        self._index_worker.start()

    def reindex_files(self):
        # Edge case: another reindex/parse already running. Surface that
        # to peer UIs (Settings) so the click feels responsive even if
        # we don't actually start a second worker. We keep `active=False`
        # in the "already in progress" branches so the Settings button
        # doesn't get stuck disabled — the in-flight worker is owned by
        # a different code path and won't fire a reindex completion to
        # re-enable it.
        if self._index_worker and self._index_worker.isRunning():
            self.reindex_status_changed.emit(
                "Indexing already in progress; try again when it finishes.", False
            )
            return
        if self._ensure_worker and self._ensure_worker.isRunning():
            self.reindex_status_changed.emit(
                "Vision model is still loading; try again in a moment.", False
            )
            return

        files = list_uploaded_doc_paths()
        if not files:
            self._status.setText("No documents to reindex")
            self.reindex_status_changed.emit("No documents to reindex.", False)
            return

        missing = [f.name for f in files if not (FILTERING_OUTPUT_DIR / f"{f.stem}.md").exists()]
        if missing:
            self._status.setText(f"Missing filtered results for {len(missing)} document(s)")
            self.reindex_status_changed.emit(
                f"Missing filtered results for {len(missing)} document(s).", False
            )
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
        failures: list[str] = []
        for path in self._current_batch:
            purge_document_artifacts(path.name)
            ok, err = _safe_unlink(path)
            if not ok:
                failures.append(f"{path.name}: {err}")
            log("DOCS", f"Cleaned up cancelled file: {path.name}")
        self._current_batch = []
        if failures:
            self._status.setText(
                DOCS_COULD_NOT_REMOVE_BATCH.format(
                    count=len(failures), first=failures[0]
                )
            )

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
        if self._reindex_active:
            self.reindex_status_changed.emit(
                f"Reindexing document {done}/{total}...", True
            )

    def _on_files_failed(self, token: int, failed: list[Path]):
        if not self._is_current_operation(token):
            return
        if self._reindex_active:
            self._refresh_list()
            msg = REINDEX_FAILED_BATCH.format(count=len(failed))
            self._status.setText(msg)
            self.reindex_status_changed.emit(msg, False)
            return
        unlink_failures: list[str] = []
        for path in failed:
            purge_document_artifacts(path.name)
            ok, err = _safe_unlink(path)
            if ok:
                log("DOCS", f"Removed failed file: {path.name}")
            else:
                log("DOCS", f"Could not remove failed file {path.name}: {err}")
                unlink_failures.append(f"{path.name}: {err}")
        self._refresh_list()
        # _refresh_list resets status to "N document(s)"; surface the
        # unlink failures after so the user sees why orphaned files
        # might still be in DOCS_DIR (e.g. permission issue).
        if unlink_failures:
            self._status.setText(
                DOCS_COULD_NOT_REMOVE_BATCH.format(
                    count=len(unlink_failures), first=unlink_failures[0]
                )
            )

    def _on_index_done(self, token: int, total_chunks: int, cancelled: bool):
        if not self._is_current_operation(token):
            return
        log("DOCS", f"Indexing complete: {total_chunks} chunks stored, cancelled={cancelled}")
        was_reindex = self._reindex_active
        if cancelled:
            self._cleanup_batch()
            self._finalize_parsing("Parsing cancelled — all files dropped")
            if was_reindex:
                self.reindex_status_changed.emit("Reindex cancelled.", False)
        else:
            self._current_batch = []
            self._finalize_parsing(f"Indexing complete — {total_chunks} chunks stored")
            if was_reindex:
                self.reindex_status_changed.emit(
                    f"Reindex complete — {total_chunks} chunks stored.", False
                )

    def _on_index_error(self, token: int, error: str):
        if not self._is_current_operation(token):
            return
        log("DOCS", f"Indexing error: {error}")
        was_reindex = self._reindex_active
        if self._cancelled:
            self._cleanup_batch()
            self._finalize_parsing("Parsing cancelled")
            if was_reindex:
                self.reindex_status_changed.emit("Reindex cancelled.", False)
        else:
            self._current_batch = []
            self._finalize_parsing(INDEXING_FAILED.format(error=error))
            if was_reindex:
                self.reindex_status_changed.emit(
                    REINDEX_FAILED_SINGLE.format(error=error), False
                )

    def _delete_file(self, path: Path):
        # Refuse delete while ANY LLM-using worker is reading these docs:
        #  - parse/reindex (own IndexWorker) writes chunks for this file
        #  - Trends extraction iterates uploaded filtered markdown
        #  - Health Report reads uploaded filtered markdown
        #  - Chat RAG retrieval queries chunks from chromadb
        # Pulling a doc out mid-flight leaves phantom chunks or silently
        # produces an incomplete extraction.
        if self.is_busy() or (self.busy_check is not None and self.busy_check()):
            self._status.setText(
                "Wait for the current operation to finish or cancel it, "
                "then try again."
            )
            return
        log("DOCS", f"Deleting document: {path.name}")
        purge_document_artifacts(path.name)
        ok, err = _safe_unlink(path)
        self._table_signature = None
        if ok:
            self._status.setText(f"Deleted: {path.name}")
        else:
            self._status.setText(
                DOCS_COULD_NOT_DELETE_SINGLE.format(name=path.name, err=err)
            )
        self._refresh_list()

    def _delete_all(self):
        # The Delete-All button is already disabled during the local
        # parse/reindex, but Trends/Health Report iterate the same docs
        # without disabling it. Same guard as _delete_file.
        if self.is_busy() or (self.busy_check is not None and self.busy_check()):
            self._status.setText(
                "Wait for the current operation to finish or cancel it, "
                "then try again."
            )
            return
        log("DOCS", "Deleting all documents")
        files = list_uploaded_doc_paths()
        deleted = 0
        failures: list[str] = []
        for file_path in files:
            purge_document_artifacts(file_path.name)
            ok, err = _safe_unlink(file_path)
            if ok:
                deleted += 1
            else:
                failures.append(f"{file_path.name}: {err}")
        self._table_signature = None
        if failures:
            self._status.setText(
                f"Deleted {deleted}/{len(files)}; "
                f"failed: {failures[0]}"
            )
        else:
            self._status.setText(f"Deleted {deleted} document(s)")
        self._refresh_list()


__all__ = ["DocumentsHubWidget"]
