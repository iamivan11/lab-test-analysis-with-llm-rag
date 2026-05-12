import re

import markdown as md_lib
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)


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

from ui.styles import STYLESHEET

_LATEX_INLINE = re.compile(r"\$\$?(.*?)\$\$?", re.DOTALL)
_LATEX_COMMANDS = [
    (re.compile(r"\\text\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\textbf\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\times"), "x"),
    (re.compile(r"\\div"), "/"),
    (re.compile(r"\\pm"), "+/-"),
    (re.compile(r"\\leq?"), "<="),
    (re.compile(r"\\geq?"), ">="),
    (re.compile(r"\\approx"), "~"),
    (re.compile(r"\\cdot"), "*"),
    (re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}"), r"\1/\2"),
    (re.compile(r"\\[a-zA-Z]+"), ""),
    (re.compile(r"[{}]"), ""),
]
_MD_BULLET = re.compile(r"^[ \t]*[-*+] ", re.MULTILINE)
_MD_NUMBERED = re.compile(r"^[ \t]*\d+\. ", re.MULTILINE)
_TABLE_TAG = re.compile(r"<table>")
_TH_STYLE = re.compile(r'<th\b(?: style="[^"]*")?')
_TD_STYLE = re.compile(r'<td\b(?: style="[^"]*")?')


def _strip_latex(text: str) -> str:
    def _replace_math(match: re.Match) -> str:
        content = match.group(1)
        for pattern, replacement in _LATEX_COMMANDS:
            content = pattern.sub(replacement, content)
        return content.strip()

    return _LATEX_INLINE.sub(_replace_math, text)


def _strip_lists(text: str) -> str:
    text = _MD_BULLET.sub("", text)
    return _MD_NUMBERED.sub("", text)


def _style_tables(html: str) -> str:
    html = _TABLE_TAG.sub(
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse: collapse; border-color: #45475a; margin: 8px 0;'>",
        html,
    )
    html = _TH_STYLE.sub(
        "<th style='background-color: #313244; padding: 6px 10px; text-align: left;'",
        html,
    )
    return _TD_STYLE.sub(
        "<td style='padding: 6px 10px; text-align: left;'",
        html,
    )


def render_message_html(text: str) -> str:
    rendered = md_lib.markdown(_strip_lists(_strip_latex(text)), extensions=["tables"])
    return _style_tables(rendered)


class ChatItemWidget(QWidget):
    rename_requested = Signal(str, str)
    delete_requested = Signal(str)

    def __init__(self, chat_id: str, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("chatItemWidget")
        self._chat_id = chat_id
        self._title = title

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 4, 0)
        layout.setSpacing(4)

        self._label = QLabel(title)
        self._label.setObjectName("chatItemLabel")
        layout.addWidget(self._label, stretch=1)

        self._menu_btn = QPushButton("\u22ef")
        self._menu_btn.setObjectName("chatMenuButton")
        self._menu_btn.setFixedSize(24, 24)
        self._menu_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._menu_btn.setVisible(False)
        self._menu_btn.clicked.connect(self._show_menu)
        layout.addWidget(self._menu_btn)

    @property
    def title(self) -> str:
        return self._title

    def _show_menu(self):
        menu = QMenu(self)
        rename_action = menu.addAction("Rename")

        delete_label = QLabel("Delete")
        delete_label.setStyleSheet(
            """
            QLabel {
                color: #f38ba8; padding: 6px 16px; border-radius: 4px;
            }
            QLabel:hover { background-color: #45475a; }
            """
        )
        delete_action = QWidgetAction(menu)
        delete_action.setDefaultWidget(delete_label)
        menu.addAction(delete_action)

        action = menu.exec(self._menu_btn.mapToGlobal(self._menu_btn.rect().bottomLeft()))
        if action == rename_action:
            self.rename_requested.emit(self._chat_id, self._title)
        elif action == delete_action:
            self.delete_requested.emit(self._chat_id)

    def enterEvent(self, event):
        self._menu_btn.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._menu_btn.setVisible(False)
        super().leaveEvent(event)


class RenameChatDialog(QDialog):
    def __init__(self, current_title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rename Chat")
        self.setMinimumWidth(360)
        self.setStyleSheet(STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        layout.addWidget(QLabel("Chat title:"))

        self._input = QLineEdit(current_title)
        self._input.returnPressed.connect(self.accept)
        layout.addWidget(self._input)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("attachButton")
        cancel_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        save_btn.clicked.connect(self.accept)
        btn_row.addWidget(save_btn)

        layout.addLayout(btn_row)

    def title(self) -> str:
        return self._input.text().strip()
