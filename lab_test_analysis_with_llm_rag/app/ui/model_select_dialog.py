"""Dialog for selecting a locally available GGUF model."""

from pathlib import Path

from PySide6.QtCore import Qt
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

from app.config import MODELS_DIR, format_size, load_model_meta
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

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Model", "Size"])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
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
        for row, path in enumerate(models):
            self._table.setItem(row, 0, QTableWidgetItem(_display_name(path)))
            size_item = QTableWidgetItem(format_size(path.stat().st_size))
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, 1, size_item)

    def _select(self):
        row = self._table.currentRow()
        if 0 <= row < len(self._paths):
            self.selected_path = self._paths[row]
            self.accept()
