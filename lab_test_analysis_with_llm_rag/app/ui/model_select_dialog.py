"""Dialog for selecting a locally available GGUF model."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from app.config import DEFAULT_MODEL_FILE, MODELS_DIR, format_size, load_model_meta
from app.core.llm_engine import get_current_model_path
from app.core.logger import log
from app.core.model_meta import _clean_name, read_model_name
from app.ui.styles import STYLESHEET


def _is_main_model(path: Path) -> bool:
    return "mmproj" not in path.name.lower()


def _display_name(path: Path) -> str:
    meta = load_model_meta(str(path))
    if meta and meta.get("name"):
        return _clean_name(meta["name"])
    return read_model_name(str(path)) or ""


class ModelSelectDialog(QDialog):
    loaded_model_deleted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Model")
        self.setMinimumSize(500, 340)
        self.setStyleSheet(STYLESHEET)
        self.selected_path: str | None = None
        self._paths: list[str] = []
        self._build_ui()
        self._populate()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Model", "Size", ""])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.doubleClicked.connect(self._select)
        self._table.selectionModel().selectionChanged.connect(
            lambda: self._load_btn.setEnabled(bool(self._table.selectedItems()))
        )
        layout.addWidget(self._table)

        self._empty_label = QLabel("No models found. Use Model Hub to download one.")
        self._empty_label.setObjectName("statusLabel")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setVisible(False)
        layout.addWidget(self._empty_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("attachButton")
        cancel_btn.setFixedSize(100, 38)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._load_btn = QPushButton("Load")
        self._load_btn.setFixedSize(100, 38)
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._select)
        btn_row.addWidget(self._load_btn)

        layout.addLayout(btn_row)

    def _populate(self):
        models = sorted(
            [p for p in MODELS_DIR.glob("*.gguf") if _is_main_model(p)],
            key=lambda p: p.name.lower(),
        )
        if not models:
            self._table.setVisible(False)
            self._empty_label.setVisible(True)
            return

        self._paths = [str(p) for p in models]
        self._table.setRowCount(len(models))
        loaded_path = get_current_model_path()
        for row, path in enumerate(models):
            is_loaded = str(path) == loaded_path

            name_item = QTableWidgetItem(_display_name(path))
            size_item = QTableWidgetItem(format_size(path.stat().st_size))
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if is_loaded:
                disabled = (
                    name_item.flags()
                    & ~Qt.ItemFlag.ItemIsSelectable
                    & ~Qt.ItemFlag.ItemIsEnabled
                )
                name_item.setFlags(disabled)
                size_item.setFlags(disabled)
                name_item.setToolTip("Currently loaded")
                size_item.setToolTip("Currently loaded")
            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, size_item)

            del_btn = QPushButton("\u2212")
            del_btn.setObjectName("iconSecondary")
            del_btn.setFixedSize(28, 28)
            if path.name == DEFAULT_MODEL_FILE:
                del_btn.setEnabled(False)
                del_btn.setToolTip("Default model — cannot be deleted")
            else:
                del_btn.setToolTip("Delete")
                del_btn.clicked.connect(lambda _, p=path: self._delete_model(p))
            self._table.setCellWidget(row, 2, del_btn)

    def _delete_model(self, path: Path):
        log("MODELS", f"Deleting model: {path.name}")
        was_loaded = get_current_model_path() == str(path)
        path.unlink(missing_ok=True)
        self._populate()
        if was_loaded:
            self.loaded_model_deleted.emit()

    def _select(self):
        row = self._table.currentRow()
        if not (0 <= row < len(self._paths)):
            return
        path = self._paths[row]
        if path == get_current_model_path():
            return
        self.selected_path = path
        self.accept()
