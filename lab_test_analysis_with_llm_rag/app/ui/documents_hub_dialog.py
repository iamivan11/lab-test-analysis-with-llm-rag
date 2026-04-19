"""Dialog for managing uploaded lab test documents."""

import shutil
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
)

from app.config import DOCS_DIR, format_size
from app.core.document_parser import SUPPORTED_EXTENSIONS, parse_document, set_save_dir
from app.core.knowledge_base import (
    index_document,
    remove_document,
)
from app.core.logger import log
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


class IndexWorker(QThread):
    """Background worker: phase 1 parses all files, phase 2 indexes them."""

    progress = Signal(str)
    file_progress = Signal(int, int)
    finished = Signal(int)
    failed_files = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, file_paths: list[Path]):
        super().__init__()
        self.file_paths = file_paths

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
                f"{len(parsed)} OK, {len(failed)} failed",
            )

            # Phase 2: Index all successfully parsed files
            total_chunks = 0
            for i, (path, markdown) in enumerate(parsed.items()):
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
                f"{len(failed)} failed in {total_elapsed}",
            )

            if failed:
                self.failed_files.emit(failed)
            self.finished.emit(total_chunks)
        except Exception as e:
            log("WORKER", f"IndexWorker: ERROR {e}")
            self.error_occurred.emit(str(e))
        finally:
            set_save_dir(None)


class OrganizeWorker(QThread):
    """Background worker that classifies and synthesizes documents into master files."""

    progress = Signal(str)
    finished = Signal(int)  # number of master files created
    error_occurred = Signal(str)

    def run(self):
        try:
            from app.core.organizer import organize_documents

            count = organize_documents(
                _VISION_OUTPUT_DIR,
                _MASTER_DIR,
                on_progress=self.progress.emit,
            )
            self.finished.emit(count)
        except Exception as e:
            log("ORGANIZER", f"OrganizeWorker error: {e}")
            self.error_occurred.emit(str(e))


# ── Dialog ──────────────────────────────────────────────────────────────


class DocumentsHubDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Documents Hub")
        self.setMinimumSize(600, 400)
        self.setStyleSheet(STYLESHEET)
        self._index_worker = None
        self._organize_worker = None
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

        # Progress bar (hidden until work starts)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%v / %m files")
        layout.addWidget(self._progress_bar)

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
                del_btn = QPushButton("-")
                del_btn.setObjectName("attachButton")
                del_btn.setFixedSize(56, 28)
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

        self._upload_btn.setEnabled(False)
        self._delete_all_btn.setEnabled(False)
        self._compile_btn.setEnabled(False)
        self._status.setText("Starting...")

        self._progress_bar.setValue(0)
        self._progress_bar.setMaximum(len(file_paths))
        self._progress_bar.setVisible(True)

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

    def _on_index_done(self, total_chunks: int):
        log("DOCS", f"Indexing complete: {total_chunks} chunks stored")
        self._progress_bar.setVisible(False)
        self._upload_btn.setEnabled(True)
        self._delete_all_btn.setEnabled(True)
        self._compile_btn.setEnabled(True)
        self._status.setText(f"Indexing complete — {total_chunks} chunks stored")
        self._refresh_list()

    def _on_index_error(self, error: str):
        log("DOCS", f"Indexing error: {error}")
        self._progress_bar.setVisible(False)
        self._upload_btn.setEnabled(True)
        self._delete_all_btn.setEnabled(True)
        self._compile_btn.setEnabled(True)
        self._status.setText(f"Error: {error}")
        self._refresh_list()

    # ── Compile (organize) ───────────────────────────────────────────────

    def _start_organize(self):
        if not _VISION_OUTPUT_DIR.is_dir() or not any(_VISION_OUTPUT_DIR.glob("*.md")):
            self._status.setText("No parsed documents found. Upload and process documents first.")
            return

        if self._organize_worker and self._organize_worker.isRunning():
            self._status.setText("Compilation already in progress...")
            return

        self._compile_btn.setEnabled(False)
        self._upload_btn.setEnabled(False)
        self._delete_all_btn.setEnabled(False)
        self._progress_bar.setMaximum(0)  # indeterminate
        self._progress_bar.setVisible(True)
        self._status.setText("Starting compilation...")

        self._organize_worker = OrganizeWorker()
        self._organize_worker.progress.connect(self._on_compile_progress)
        self._organize_worker.finished.connect(self._on_compile_done)
        self._organize_worker.error_occurred.connect(self._on_compile_error)
        self._organize_worker.start()

    def _on_compile_progress(self, msg: str):
        self._status.setText(msg)

    def _on_compile_done(self, count: int):
        log("DOCS", f"Compilation complete: {count} master files")
        self._progress_bar.setVisible(False)
        self._compile_btn.setEnabled(True)
        self._upload_btn.setEnabled(True)
        self._delete_all_btn.setEnabled(True)
        self._status.setText(f"Compilation complete — {count} master files created")
        # Switch to master files view automatically
        if not self._showing_master:
            self._showing_master = True
            self._view_toggle.setText("Sources")
            self._upload_btn.setVisible(False)
            self._delete_all_btn.setVisible(False)
            self._compile_btn.setVisible(False)
        self._refresh_list()

    def _on_compile_error(self, error: str):
        log("DOCS", f"Compilation error: {error}")
        self._progress_bar.setVisible(False)
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
