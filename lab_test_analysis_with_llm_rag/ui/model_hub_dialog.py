"""Dialog for browsing and downloading GGUF models from HuggingFace."""

import threading

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config import format_size
from core.logger import log
from core.model_hub import (
    download_model,
    list_gguf_files,
    search_models,
)
from ui.styles import STYLESHEET


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
    progress = Signal(int, str)  # (percentage, "downloaded / total" label)
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


class ModelHubDialog(QDialog):
    model_downloaded = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Hub")
        self.setMinimumSize(700, 500)
        self.setStyleSheet(STYLESHEET)

        self._search_worker = None
        self._file_worker = None
        self._download_worker = None
        self._selected_model_id = ""
        self._search_results = []

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Search bar
        search_row = QHBoxLayout()
        search_row.setSpacing(8)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search GGUF models on HuggingFace...")
        self._search_input.returnPressed.connect(self._do_search)
        search_row.addWidget(self._search_input, stretch=1)

        self._search_btn = QPushButton("Search")
        self._search_btn.setFixedSize(100, 38)
        self._search_btn.clicked.connect(self._do_search)
        search_row.addWidget(self._search_btn)

        layout.addLayout(search_row)

        # Status label
        self._status = QLabel("")
        self._status.setObjectName("statusLabel")
        layout.addWidget(self._status)

        # Models table
        self._models_table = QTableWidget(0, 2)
        self._models_table.setHorizontalHeaderLabels(["Model", "Downloads"])
        self._models_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._models_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._models_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._models_table.verticalHeader().setVisible(False)
        header = self._models_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._models_table.cellClicked.connect(self._on_model_clicked)
        layout.addWidget(self._models_table, stretch=1)

        # Files table (shown after selecting a model)
        self._files_label = QLabel("")
        self._files_label.setVisible(False)
        layout.addWidget(self._files_label)

        self._files_table = QTableWidget(0, 3)
        self._files_table.setHorizontalHeaderLabels(["File", "Size", ""])
        self._files_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._files_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._files_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._files_table.verticalHeader().setVisible(False)
        files_header = self._files_table.horizontalHeader()
        files_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        files_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        files_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._files_table.setVisible(False)
        self._files_table.setMaximumHeight(180)
        layout.addWidget(self._files_table)

        # Download progress
        self._progress_widget = QWidget()
        self._progress_widget.setVisible(False)
        progress_layout = QHBoxLayout(self._progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(8)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        progress_layout.addWidget(self._progress_bar, stretch=1)

        self._progress_label = QLabel("")
        self._progress_label.setObjectName("statusLabel")
        self._progress_label.setMinimumWidth(100)
        progress_layout.addWidget(self._progress_label)

        self._cancel_btn = QPushButton("\u2715")
        self._cancel_btn.setObjectName("iconSecondary")
        self._cancel_btn.setFixedSize(28, 28)
        self._cancel_btn.setToolTip("Cancel")
        self._cancel_btn.clicked.connect(self._cancel_download)
        progress_layout.addWidget(self._cancel_btn)

        layout.addWidget(self._progress_widget)

        # Bottom buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setObjectName("attachButton")
        close_btn.setFixedSize(100, 38)
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # ── Search ────────────────────────────────────────────────────────────

    def _do_search(self):
        query = self._search_input.text().strip()
        if not query:
            return

        self._status.setText("Searching...")
        self._search_btn.setEnabled(False)
        self._files_table.setVisible(False)
        self._files_label.setVisible(False)
        self._models_table.setRowCount(0)

        self._search_worker = SearchWorker(query)
        self._search_worker.finished.connect(self._on_search_done)
        self._search_worker.error_occurred.connect(self._on_search_error)
        self._search_worker.start()

    def _on_search_done(self, results: list):
        self._search_btn.setEnabled(True)
        self._models_table.setRowCount(len(results))
        self._search_results = results

        for row, model in enumerate(results):
            name_item = QTableWidgetItem(model["id"])
            self._models_table.setItem(row, 0, name_item)

            dl_item = QTableWidgetItem(f"{model['downloads']:,}")
            dl_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._models_table.setItem(row, 1, dl_item)

        self._status.setText(f"Found {len(results)} models")

    def _on_search_error(self, error: str):
        self._search_btn.setEnabled(True)
        self._status.setText(f"Search error: {error}")

    # ── File listing ──────────────────────────────────────────────────────

    def _on_model_clicked(self, row: int, _col: int):
        if row >= len(self._search_results):
            return
        model = self._search_results[row]
        self._selected_model_id = model["id"]

        self._files_label.setText(f"Loading files from {model['id']}...")
        self._files_label.setVisible(True)
        self._files_table.setRowCount(0)
        self._files_table.setVisible(True)

        self._file_worker = FileListWorker(model["id"])
        self._file_worker.finished.connect(self._on_files_loaded)
        self._file_worker.error_occurred.connect(self._on_files_error)
        self._file_worker.start()

    def _on_files_loaded(self, files: list):
        self._files_label.setText(f"{self._selected_model_id} — {len(files)} GGUF files")
        self._files_table.setRowCount(len(files))

        for row, f in enumerate(files):
            name_item = QTableWidgetItem(f["name"])
            self._files_table.setItem(row, 0, name_item)

            size_item = QTableWidgetItem(format_size(f["size"]))
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._files_table.setItem(row, 1, size_item)

            dl_btn = QPushButton("\u2193")
            dl_btn.setObjectName("iconPrimary")
            dl_btn.setFixedSize(28, 28)
            dl_btn.setToolTip("Download")
            dl_btn.clicked.connect(lambda checked, name=f["name"]: self._start_download(name))
            self._files_table.setCellWidget(row, 2, dl_btn)

    def _on_files_error(self, error: str):
        self._files_label.setText(f"Error loading files: {error}")

    # ── Download ──────────────────────────────────────────────────────────

    def _start_download(self, filename: str):
        log("HUB", f"Starting download: {filename} from {self._selected_model_id}")
        if self._download_worker and self._download_worker.isRunning():
            self._status.setText("A download is already in progress")
            return

        self._status.setText(f"Downloading {filename}...")
        self._progress_widget.setVisible(True)
        self._progress_bar.setValue(0)
        self._progress_label.setText("0%")
        self._cancel_btn.setEnabled(True)

        self._download_worker = DownloadWorker(self._selected_model_id, filename)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(self._on_download_done)
        self._download_worker.error_occurred.connect(self._on_download_error)
        self._download_worker.start()

    def _on_download_progress(self, pct: int, label: str):
        self._progress_bar.setValue(pct)
        self._progress_label.setText(label)

    def _on_download_done(self, path: str):
        log("HUB", f"Download complete: {path}")
        self._progress_widget.setVisible(False)
        self._status.setText(f"Downloaded to {path}")
        self.model_downloaded.emit(path)

    def _on_download_error(self, error: str):
        self._progress_widget.setVisible(False)
        self._status.setText(f"Download error: {error}")

    def _cancel_download(self):
        if self._download_worker and self._download_worker.isRunning():
            log("HUB", "User cancelled download")
            self._cancel_btn.setEnabled(False)
            self._status.setText("Cancelling download...")
            self._download_worker.stop()
