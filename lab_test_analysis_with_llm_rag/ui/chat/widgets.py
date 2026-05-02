from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QListWidget, QMenu, QTextBrowser, QTextEdit


class PlainTextPasteEdit(QTextEdit):
    """QTextEdit that always pastes plain text from the clipboard."""

    def insertFromMimeData(self, source) -> None:
        self.insertPlainText(source.text())


class AutoHideScrollListWidget(QListWidget):
    """QListWidget whose vertical scrollbar shows briefly while scrolling."""

    HIDE_DELAY_MS = 900

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scroll_hide_timer = QTimer(self)
        self._scroll_hide_timer.setSingleShot(True)
        self._scroll_hide_timer.timeout.connect(self._hide_scrollbar)
        self.verticalScrollBar().valueChanged.connect(self._on_scroll_activity)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def _on_scroll_activity(self):
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll_hide_timer.start(self.HIDE_DELAY_MS)

    def _hide_scrollbar(self):
        if self.verticalScrollBar().isSliderDown():
            self._scroll_hide_timer.start(self.HIDE_DELAY_MS)
            return
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def wheelEvent(self, event):
        self._on_scroll_activity()
        super().wheelEvent(event)


class ChatDisplayBrowser(QTextBrowser):
    """QTextBrowser with a minimal text-only context menu."""

    def build_context_menu(self) -> QMenu:
        menu = QMenu(self)

        if self.textCursor().hasSelection():
            copy_action = menu.addAction("Copy")
            copy_action.triggered.connect(self.copy)

        select_all_action = menu.addAction("Select All")
        select_all_action.triggered.connect(self.selectAll)
        return menu

    def contextMenuEvent(self, event) -> None:
        self.build_context_menu().exec(event.globalPos())
