"""Reusable Model Hub widget."""

import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config import (
    APPROVED_MODELS,
    approved_main_model_paths,
    approved_model_file_path,
    approved_model_for_file,
    format_size,
    get_default_model,
    load_default_model_id,
)
from core.llm_engine import get_current_model_path
from core.logger import log
from ui.components import icon_button
from ui.models.file_helpers import is_main_model, model_display_name
from ui.models.workers import DownloadWorker


class ModelHubWidget(QWidget):
    """Tabs for local models and HuggingFace GGUF downloads."""

    load_requested = Signal(str)
    loaded_model_deleted = Signal()
    model_downloaded = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Injected by main_window so deleting the currently-loaded model
        # can refuse while LLM work is in flight — otherwise the file
        # vanishes from disk while llama-server holds it via mmap, the
        # auto-recovery load_model() gets refused by the same gate, and
        # the app sits in a half-state until the next manual action.
        from collections.abc import Callable

        self.busy_check: Callable[[], bool] | None = None
        self._download_workers: dict[int, DownloadWorker] = {}
        self._download_rows: dict[int, QWidget] = {}
        self._download_targets: dict[int, str] = {}
        self._active_download_targets: set[str] = set()
        self._local_paths: list[str] = []
        self._download_token = 0
        self._local_signature: tuple[object, ...] | None = None

        self._build_ui()

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

        self._local_empty = QLabel("No local models yet — switch to 'Download'.")
        self._local_empty.setObjectName("statusLabel")
        self._local_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._local_empty.setVisible(False)
        wrap_layout.addWidget(self._local_empty)

        return wrap

    def refresh_local(self):
        # Rebuild the Download tab too so its per-row "↓" buttons reflect
        # the current on-disk state (a model installed via onboarding or
        # downloaded just now should NOT show as still downloadable).
        self._populate_approved_downloads()

        models = sorted(
            approved_main_model_paths(),
            key=lambda p: p.name.lower(),
        )
        loaded_path = get_current_model_path()
        # Include the default model id in the signature so changing the
        # default in Settings re-renders the table — otherwise the
        # per-row delete buttons keep their previous enabled/disabled
        # state (the old default stays uneditable, the new default
        # stays deletable).
        default_id = load_default_model_id()
        signature = (
            loaded_path,
            default_id,
            tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in models),
        )
        if signature == self._local_signature:
            return

        self._local_signature = signature
        self._local_paths = [str(p) for p in models]
        self._local_table.setRowCount(len(models))

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

            if path == approved_model_file_path(get_default_model()):
                del_btn = icon_button(
                    "−",  # noqa: RUF001 - UI glyph, not arithmetic.
                    tooltip="Default model — cannot be deleted",
                )
                del_btn.setEnabled(False)
            else:
                del_btn = icon_button(
                    "−",  # noqa: RUF001 - UI glyph, not arithmetic.
                    tooltip="Delete",
                    on_click=lambda p=path: self._delete_local(p),
                )
            self._local_table.setCellWidget(row, 3, del_btn)

    def _delete_local(self, path: Path):
        log("MODELS", f"Deleting model: {path.name}")
        was_loaded = get_current_model_path() == str(path)
        # Deleting the loaded model triggers an auto-swap to the default
        # via loaded_model_deleted → ensure_default_model → load_model,
        # which is gated by is_llm_busy. So if any LLM worker is active,
        # the swap would be refused mid-flow and the app would sit with
        # the file unlinked but llama-server still mmap'd onto it.
        # Refuse the delete up front in that case.
        if was_loaded and self.busy_check is not None and self.busy_check():
            self._status.setText(
                "Wait for the current operation to finish or cancel it, "
                "then try again."
            )
            return
        model = approved_model_for_file(path)
        if model and path.parent.name == model["display_name"]:
            shutil.rmtree(path.parent, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        self.refresh_local()
        self._populate_approved_downloads()
        if was_loaded:
            self.loaded_model_deleted.emit()

    def _build_browse_tab(self) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 12, 0, 0)

        self._status = QLabel("")
        self._status.setObjectName("statusLabel")
        layout.addWidget(self._status)

        self._download_table = QTableWidget(0, 4)
        self._download_table.setHorizontalHeaderLabels(["Model", "Size", "Files", ""])
        self._download_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._download_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._download_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._download_table.verticalHeader().setVisible(False)
        header = self._download_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._download_table, stretch=1)

        self._downloads_layout = QVBoxLayout()
        self._downloads_layout.setSpacing(8)
        layout.addLayout(self._downloads_layout)
        self._populate_approved_downloads()
        return wrap

    def _populate_approved_downloads(self) -> None:
        models = list(APPROVED_MODELS.values())
        self._download_table.setRowCount(len(models))
        for row, model in enumerate(models):
            name_item = QTableWidgetItem(model["display_name"])
            self._download_table.setItem(row, 0, name_item)
            size_item = QTableWidgetItem(format_size(model["download_size_bytes"]))
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._download_table.setItem(row, 1, size_item)

            file_count = len(self._download_files(model))
            self._download_table.setItem(row, 2, QTableWidgetItem(str(file_count)))

            dl_btn = icon_button(
                "↓",
                name="iconPrimary",
                tooltip="Download",
                on_click=lambda m=model: self._start_download(m),
            )
            dl_btn.setEnabled(bool(self._missing_download_files(model)))
            self._download_table.setCellWidget(row, 3, dl_btn)

    def _download_files(self, model: dict) -> list[tuple[str, str]]:
        files = [(model["model_file"], model["model_file"])]
        if mmproj_file := model.get("mmproj_file"):
            files.append((mmproj_file, model.get("mmproj_local", mmproj_file)))
        return files

    def _missing_download_files(self, model: dict) -> list[tuple[str, str]]:
        missing = []
        for hf_name, local_name in self._download_files(model):
            target = approved_model_file_path(model, local_name)
            if target.exists() or str(target) in self._active_download_targets:
                continue
            missing.append((hf_name, local_name))
        return missing

    def _start_download(self, model: dict):
        files = self._missing_download_files(model)
        log("HUB", f"Starting download: {model['display_name']} ({files})")
        if not files:
            self._status.setText("Model is already downloaded")
            return
        free_slots = 3 - len(self._download_workers)
        if len(files) > free_slots:
            self._status.setText("Up to 3 files can download at the same time")
            return

        for hf_name, local_name in files:
            self._start_file_download(
                model,
                hf_name,
                str(approved_model_file_path(model, local_name)),
            )
        self._status.setText(f"Started {len(files)} download(s)")
        # `_active_download_targets` was just populated; rebuild the
        # Download tab so the per-row "↓" button reflects the active
        # state (otherwise it stays clickable and a second click
        # would spawn duplicate workers).
        self._populate_approved_downloads()

    def _start_file_download(self, model: dict, hf_name: str, local_name: str) -> None:
        self._download_token += 1
        token = self._download_token
        self._download_rows[token] = self._create_download_row(token, local_name)
        self._download_targets[token] = local_name
        self._active_download_targets.add(local_name)

        worker = DownloadWorker(model["repo_id"], hf_name, local_name)
        self._download_workers[token] = worker
        worker.progress.connect(
            lambda pct, label, token=token: self._on_download_progress(token, pct, label)
        )
        worker.finished.connect(
            lambda path, token=token: self._on_download_done(token, path)
        )
        worker.error_occurred.connect(
            lambda error, token=token: self._on_download_error(token, error)
        )
        worker.start()

    def _create_download_row(self, token: int, filename: str) -> QWidget:
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        title = QLabel(f"Downloading {filename}...")
        title.setObjectName("statusLabel")
        row_layout.addWidget(title)

        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(8)

        bar = QProgressBar()
        bar.setMinimum(0)
        bar.setMaximum(100)
        progress_row.addWidget(bar, stretch=1)

        size_label = QLabel("0%")
        size_label.setObjectName("statusLabel")
        size_label.setMinimumWidth(100)
        progress_row.addWidget(size_label)

        cancel_btn = icon_button(
            "✕", tooltip="Cancel", on_click=lambda: self._cancel_download(token)
        )
        progress_row.addWidget(cancel_btn)

        row._progress_bar = bar  # type: ignore[attr-defined]
        row._progress_label = size_label  # type: ignore[attr-defined]
        row._cancel_btn = cancel_btn  # type: ignore[attr-defined]

        row_layout.addLayout(progress_row)
        self._downloads_layout.addWidget(row)
        return row

    def _on_download_progress(self, token: int, pct: int, label: str):
        row = self._download_rows.get(token)
        if row is None:
            log("HUB", f"Ignored stale download progress: token={token}")
            return
        row._progress_bar.setValue(pct)  # type: ignore[attr-defined]
        row._progress_label.setText(label)  # type: ignore[attr-defined]

    def _on_download_done(self, token: int, path: str):
        if token not in self._download_workers:
            log("HUB", f"Ignored stale download result: token={token}")
            return
        log("HUB", f"Download complete: {path}")
        self._remove_download_row(token)
        self._status.setText(f"Downloaded to {path}")
        if is_main_model(Path(path)):
            self.model_downloaded.emit(path)
        self.refresh_local()
        self._populate_approved_downloads()

    def _on_download_error(self, token: int, error: str):
        if token not in self._download_workers:
            log("HUB", f"Ignored stale download error: token={token}")
            return
        self._remove_download_row(token)
        self._status.setText(f"Download error: {error}")
        # The Download tab's per-row "↓" buttons consult
        # `_active_download_targets`; rebuild the table now that the
        # target was released, otherwise the button stays disabled
        # until the user navigates away and back.
        self._populate_approved_downloads()

    def _cancel_download(self, token: int):
        worker = self._download_workers.get(token)
        row = self._download_rows.get(token)
        if worker and worker.isRunning():
            log("HUB", "User cancelled download")
            if row:
                row._cancel_btn.setEnabled(False)  # type: ignore[attr-defined]
            self._status.setText("Cancelling download...")
            worker.stop()

    def _remove_download_row(self, token: int):
        self._download_workers.pop(token, None)
        local_name = self._download_targets.pop(token, None)
        if local_name:
            self._active_download_targets.discard(local_name)
        row = self._download_rows.pop(token, None)
        if row is not None:
            self._downloads_layout.removeWidget(row)
            row.deleteLater()
