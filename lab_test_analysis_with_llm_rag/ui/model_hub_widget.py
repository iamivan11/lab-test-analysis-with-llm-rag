"""Reusable Model Hub widget."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config import DEFAULT_MODEL_FILE, MODELS_DIR, format_size
from core.llm_engine import get_current_model_path
from core.logger import log
from ui.model_file_helpers import is_main_model, model_display_name
from ui.model_hub_workers import DownloadWorker, FileListWorker, SearchWorker


class ModelHubWidget(QWidget):
    """Tabs for local models and HuggingFace GGUF downloads."""

    load_requested = Signal(str)
    loaded_model_deleted = Signal()
    model_downloaded = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._search_worker = None
        self._file_worker = None
        self._download_worker = None
        self._selected_model_id = ""
        self._search_results = []
        self._local_paths: list[str] = []

        self._build_ui()
        self.refresh_local()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_local_tab(), "My Models")
        self._tabs.addTab(self._build_browse_tab(), "Download")
        layout.addWidget(self._tabs, stretch=1)

    def _build_local_tab(self) -> QWidget:
        wrap = QWidget()
        wrap_layout = QVBoxLayout(wrap)
        wrap_layout.setSpacing(12)
        wrap_layout.setContentsMargins(0, 12, 0, 0)

        self._local_table = QTableWidget(0, 4)
        self._local_table.setHorizontalHeaderLabels(["Model", "Size", "", ""])
        self._local_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._local_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._local_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._local_table.verticalHeader().setVisible(False)
        local_header = self._local_table.horizontalHeader()
        local_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        local_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        local_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        local_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        wrap_layout.addWidget(self._local_table, stretch=1)

        self._local_empty = QLabel("No local models yet — switch to 'Browse & Download'.")
        self._local_empty.setObjectName("statusLabel")
        self._local_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._local_empty.setVisible(False)
        wrap_layout.addWidget(self._local_empty)

        return wrap

    def refresh_local(self):
        models = sorted(
            [p for p in MODELS_DIR.glob("*.gguf") if is_main_model(p)],
            key=lambda p: p.name.lower(),
        )
        self._local_paths = [str(p) for p in models]
        self._local_table.setRowCount(len(models))
        loaded_path = get_current_model_path()

        if not models:
            self._local_table.setVisible(False)
            self._local_empty.setVisible(True)
            return

        self._local_table.setVisible(True)
        self._local_empty.setVisible(False)

        for row, path in enumerate(models):
            is_loaded = str(path) == loaded_path

            name_item = QTableWidgetItem(model_display_name(path, fallback=path.stem))
            size_item = QTableWidgetItem(format_size(path.stat().st_size))
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._local_table.setItem(row, 0, name_item)
            self._local_table.setItem(row, 1, size_item)

            load_btn = QPushButton("Load")
            load_btn.setObjectName("attachButton")
            load_btn.setFixedSize(72, 28)
            load_btn.setStyleSheet("padding: 0 6px; font-size: 11px;")
            if is_loaded:
                load_btn.setEnabled(False)
                load_btn.setText("Loaded")
            else:
                load_btn.clicked.connect(lambda _, p=str(path): self.load_requested.emit(p))
            self._local_table.setCellWidget(row, 2, load_btn)

            del_btn = QPushButton("−")  # noqa: RUF001 - UI glyph, not arithmetic.
            del_btn.setObjectName("iconSecondary")
            del_btn.setFixedSize(28, 28)
            if path.name == DEFAULT_MODEL_FILE:
                del_btn.setEnabled(False)
                del_btn.setToolTip("Default model — cannot be deleted")
            else:
                del_btn.setToolTip("Delete")
                del_btn.clicked.connect(lambda _, p=path: self._delete_local(p))
            self._local_table.setCellWidget(row, 3, del_btn)

    def _delete_local(self, path: Path):
        log("MODELS", f"Deleting model: {path.name}")
        was_loaded = get_current_model_path() == str(path)
        path.unlink(missing_ok=True)
        self.refresh_local()
        if was_loaded:
            self.loaded_model_deleted.emit()

    def _build_browse_tab(self) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 12, 0, 0)

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

        self._status = QLabel("")
        self._status.setObjectName("statusLabel")
        layout.addWidget(self._status)

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

        self._cancel_btn = QPushButton("✕")
        self._cancel_btn.setObjectName("iconSecondary")
        self._cancel_btn.setFixedSize(28, 28)
        self._cancel_btn.setToolTip("Cancel")
        self._cancel_btn.clicked.connect(self._cancel_download)
        progress_layout.addWidget(self._cancel_btn)

        layout.addWidget(self._progress_widget)
        return wrap

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

            dl_btn = QPushButton("↓")
            dl_btn.setObjectName("iconPrimary")
            dl_btn.setFixedSize(28, 28)
            dl_btn.setToolTip("Download")
            dl_btn.clicked.connect(lambda _, name=f["name"]: self._start_download(name))
            self._files_table.setCellWidget(row, 2, dl_btn)

    def _on_files_error(self, error: str):
        self._files_label.setText(f"Error loading files: {error}")

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
        self.refresh_local()

    def _on_download_error(self, error: str):
        self._progress_widget.setVisible(False)
        self._status.setText(f"Download error: {error}")

    def _cancel_download(self):
        if self._download_worker and self._download_worker.isRunning():
            log("HUB", "User cancelled download")
            self._cancel_btn.setEnabled(False)
            self._status.setText("Cancelling download...")
            self._download_worker.stop()
