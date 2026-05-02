"""Settings — global app preferences."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from config import (
    load_ctx_size,
    load_max_tokens,
    load_model_meta,
    load_model_path,
    save_ctx_size,
    save_max_tokens,
)

_CTX_OPTIONS = [2048, 4096, 8192, 16384]
_MAX_TOKENS_OPTIONS = [1024, 2048, 4096, 8192]
_MAX_TOKENS_DEFAULT = 4096


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
    return lbl


class SettingsForm(QWidget):
    """Settings inputs without any host chrome. Used by both the dialog and
    the in-window Settings section."""

    submitted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(_section_label("Context Window"))

        model_path = load_model_path()
        meta = load_model_meta(model_path) if model_path else None
        model_max = meta.get("context_length") if meta else None

        standard = [v for v in _CTX_OPTIONS if v < model_max] if model_max else list(_CTX_OPTIONS)

        current = load_ctx_size() or (model_max or standard[-1])

        self._ctx_combo = QComboBox()
        for v in standard:
            self._ctx_combo.addItem(f"{v:,}", v)
        if model_max:
            self._ctx_combo.addItem(f"{model_max:,} (model max)", model_max)

        all_options = standard + ([model_max] if model_max else [])
        idx = next((i for i, v in enumerate(all_options) if v == current), len(all_options) - 1)
        self._ctx_combo.setCurrentIndex(idx)
        layout.addWidget(self._ctx_combo)

        ctx_note = QLabel("Takes effect on next model reload.")
        ctx_note.setObjectName("statusLabel")
        layout.addWidget(ctx_note)

        layout.addWidget(_section_label("Max Tokens"))

        current_max = load_max_tokens() or _MAX_TOKENS_DEFAULT
        self._max_tokens_combo = QComboBox()
        for v in _MAX_TOKENS_OPTIONS:
            self._max_tokens_combo.addItem(f"{v:,}", v)
        idx = next(
            (i for i, v in enumerate(_MAX_TOKENS_OPTIONS) if v == current_max),
            _MAX_TOKENS_OPTIONS.index(_MAX_TOKENS_DEFAULT),
        )
        self._max_tokens_combo.setCurrentIndex(idx)
        layout.addWidget(self._max_tokens_combo)

        max_note = QLabel("Max output tokens per response (thinking + response).")
        max_note.setObjectName("statusLabel")
        layout.addWidget(max_note)

    def save(self):
        save_ctx_size(self._ctx_combo.currentData())
        save_max_tokens(self._max_tokens_combo.currentData())


__all__ = ["SettingsForm"]
