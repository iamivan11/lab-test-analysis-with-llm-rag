import re
import threading
from pathlib import Path

import markdown as md_lib
from PySide6.QtCore import QEvent, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QTextBlockFormat, QTextCharFormat
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from app.config import APP_NAME, load_model_path, load_profile, save_model_path
from app.core.chat_store import (
    delete_chat,
    list_chats,
    load_chat,
    new_chat,
    rename_chat,
    save_chat,
    title_from_first_message,
)
from app.core.llm_engine import (
    generate_stream,
    is_server_running,
    start_server,
    stop_server,
    summarize_history,
)
from app.core.logger import log
from app.core.model_hub import ensure_default_model
from app.ui.documents_hub_dialog import DocumentsHubDialog
from app.ui.model_hub_dialog import ModelHubDialog
from app.ui.profile_dialog import ProfileDialog
from app.ui.styles import STYLESHEET

# LaTeX patterns: $...$, $$...$$, and common commands inside them
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
    (re.compile(r"\\[a-zA-Z]+"), ""),  # remove any remaining commands
    (re.compile(r"[{}]"), ""),  # remove leftover braces
]


def _strip_latex(text: str) -> str:
    """Convert LaTeX math notation to plain text."""

    def _replace_math(m: re.Match) -> str:
        content = m.group(1)
        for pattern, repl in _LATEX_COMMANDS:
            content = pattern.sub(repl, content)
        return content.strip()

    return _LATEX_INLINE.sub(_replace_math, text)


# Convert markdown lists to plain paragraphs so QTextEdit doesn't corrupt formatting.
_MD_BULLET = re.compile(r"^[ \t]*[-*+] ", re.MULTILINE)
_MD_NUMBERED = re.compile(r"^[ \t]*\d+\. ", re.MULTILINE)


def _strip_lists(text: str) -> str:
    """Convert markdown list items to plain lines (prevents QTextEdit list contamination)."""
    text = _MD_BULLET.sub("", text)
    return _MD_NUMBERED.sub("", text)


_TABLE_TAG = re.compile(r"<table>")
_TH_STYLE = re.compile(r'<th\b(?: style="[^"]*")?')
_TD_STYLE = re.compile(r'<td\b(?: style="[^"]*")?')


def _style_tables(html: str) -> str:
    """Add styling to HTML tables so they render visibly in QTextEdit."""
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
        delete_label.setStyleSheet("""
            QLabel {
                color: #f38ba8; padding: 6px 16px; border-radius: 4px;
            }
            QLabel:hover { background-color: #45475a; }
        """)
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


class ServerStartWorker(QThread):
    progress = Signal(str)
    meta_ready = Signal(object)  # ModelMeta — emitted before server starts
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

    def run(self):
        log("WORKER", f"ServerStartWorker: starting with {self.model_path}")
        try:
            from app.config import load_ctx_size, save_model_meta
            from app.core.model_meta import read_model_meta

            meta = read_model_meta(self.model_path)

            save_model_meta(
                self.model_path,
                {
                    "name": meta.name,
                    "context_length": meta.context_length,
                },
            )
            self.meta_ready.emit(meta)

            n_ctx = load_ctx_size() or meta.context_length
            start_server(self.model_path, n_ctx=n_ctx, on_progress=self.progress.emit)
            log("WORKER", "ServerStartWorker: finished successfully")
            self.finished.emit(meta.name or Path(self.model_path).name)
        except Exception as e:
            log("WORKER", f"ServerStartWorker: ERROR {e}")
            self.error_occurred.emit(str(e))


class ModelDownloadWorker(QThread):
    """Downloads the default model files if missing, then signals the path."""

    progress = Signal(str)
    finished = Signal(str)  # model_path
    error_occurred = Signal(str)

    def run(self):
        log("WORKER", "ModelDownloadWorker: ensuring default model")
        try:
            model_path = ensure_default_model(on_progress=self.progress.emit)
            log("WORKER", f"ModelDownloadWorker: done, path={model_path}")
            self.finished.emit(model_path)
        except Exception as e:
            log("WORKER", f"ModelDownloadWorker: ERROR {e}")
            self.error_occurred.emit(str(e))


class LLMWorker(QThread):
    thinking_token = Signal(str)
    response_token = Signal(str)
    finished_generation = Signal()
    error_occurred = Signal(str)

    def __init__(self, history: list[dict], context: str = "", max_tokens: int | None = None):
        super().__init__()
        self.history = history
        self.context = context
        self.max_tokens = max_tokens
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        log(
            "WORKER",
            f"LLMWorker: starting, history={len(self.history)} msgs, "
            f"context={len(self.context)} chars",
        )
        try:
            token_count = 0
            for kind, token in generate_stream(
                self.history,
                context=self.context,
                stop_event=self._stop_event,
                max_tokens=self.max_tokens,
            ):
                if self._stop_event.is_set():
                    log("WORKER", "LLMWorker: stopped by user")
                    break
                token_count += 1
                if kind == "thinking":
                    self.thinking_token.emit(token)
                else:
                    self.response_token.emit(token)
            log("WORKER", f"LLMWorker: finished, {token_count} tokens emitted")
            self.finished_generation.emit()
        except Exception as e:
            log("WORKER", f"LLMWorker: ERROR {e}")
            self.error_occurred.emit(str(e))


class CompressionWorker(QThread):
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, history: list[dict]):
        super().__init__()
        self.history = history

    def run(self):
        log("WORKER", f"CompressionWorker: compressing {len(self.history)} messages")
        try:
            summary = summarize_history(self.history)
            log("WORKER", f"CompressionWorker: done, summary={len(summary)} chars")
            self.finished.emit(summary)
        except Exception as e:
            log("WORKER", f"CompressionWorker: ERROR {e}")
            self.error_occurred.emit(str(e))


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(STYLESHEET)

        self._history: list[dict] = []
        self._current_chat: dict = new_chat()
        self._thinking = False
        self._thinking_text = ""
        self._thinking_blocks: dict[int, dict] = {}
        self._thinking_id_counter = 0
        self._current_thinking_id = -1
        self._model_name = ""
        self._current_response = ""
        self._response_anchor = 0
        self._worker = None
        self._server_worker = None
        self._compression_worker = None
        self._pending_prompt = ""
        self._compression_attempted = False
        self._generation_stopped = False

        self._build_ui()
        self._init_chat()
        self._try_load_saved_model()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Sidebar ---
        self._sidebar = QWidget()
        self._sidebar.setObjectName("sidebar")
        self._sidebar.setFixedWidth(260)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(12, 16, 12, 16)
        sidebar_layout.setSpacing(8)

        new_chat_btn = QPushButton("New Chat")
        new_chat_btn.setObjectName("attachButton")
        new_chat_btn.setFixedHeight(38)
        new_chat_btn.clicked.connect(self._new_chat)
        sidebar_layout.addWidget(new_chat_btn)

        self._chat_list = QListWidget()
        self._chat_list.setObjectName("chatList")
        self._chat_list.itemClicked.connect(self._on_chat_selected)
        self._chat_list.itemDoubleClicked.connect(self._on_chat_rename)
        self._chat_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._chat_list.customContextMenuRequested.connect(self._on_chat_context_menu)
        sidebar_layout.addWidget(self._chat_list, stretch=1)

        settings_btn = QPushButton("Settings")
        settings_btn.setObjectName("attachButton")
        settings_btn.setFixedHeight(38)
        settings_btn.clicked.connect(self._open_settings)
        sidebar_layout.addWidget(settings_btn)

        chip_style = (
            "font-size: 11px; color: #cdd6f4; background: #313244;"
            "border: 1px solid #45475a; border-radius: 8px;"
            "padding: 3px 8px;"
        )
        self._mem_chip = QLabel("Memory ...")
        self._mem_chip.setStyleSheet(chip_style)
        self._mem_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cpu_chip = QLabel("CPU ...")
        self._cpu_chip.setStyleSheet(chip_style)
        self._cpu_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)

        chips_row = QHBoxLayout()
        chips_row.setSpacing(6)
        chips_row.addWidget(self._mem_chip)
        chips_row.addWidget(self._cpu_chip)
        sidebar_layout.addLayout(chips_row)

        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start(2000)
        self._update_stats()

        outer.addWidget(self._sidebar)

        # --- Main content ---
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Top bar
        top_row = QHBoxLayout()

        self._sidebar_btn = QPushButton("☰")
        self._sidebar_btn.setObjectName("attachButton")
        self._sidebar_btn.setFixedSize(38, 38)
        self._sidebar_btn.clicked.connect(self._toggle_sidebar)
        top_row.addWidget(self._sidebar_btn)

        self._hub_btn = QPushButton("Model Hub")
        self._hub_btn.setObjectName("attachButton")
        self._hub_btn.setStyleSheet("font-size: 13px;")
        self._hub_btn.clicked.connect(self._open_model_hub)
        top_row.addWidget(self._hub_btn)

        self._model_btn = QPushButton("Load Model")
        self._model_btn.setObjectName("attachButton")
        self._model_btn.setStyleSheet("font-size: 13px;")
        self._model_btn.clicked.connect(self._select_model)
        top_row.addWidget(self._model_btn)

        self._status_label = QLabel("No model loaded")
        self._status_label.setObjectName("statusLabel")
        top_row.addWidget(self._status_label, stretch=1)

        self._docs_btn = QPushButton("Documents")
        self._docs_btn.setObjectName("attachButton")
        self._docs_btn.setStyleSheet("font-size: 13px;")
        self._docs_btn.clicked.connect(self._open_documents_hub)
        top_row.addWidget(self._docs_btn)

        self._profile_btn = QPushButton("Profile")
        self._profile_btn.setObjectName("attachButton")
        self._profile_btn.clicked.connect(self._open_profile)
        top_row.addWidget(self._profile_btn)

        # Make all four top-bar buttons the same size
        btn_w = max(
            self._model_btn.sizeHint().width(),
            self._hub_btn.sizeHint().width(),
            self._docs_btn.sizeHint().width(),
            self._profile_btn.sizeHint().width(),
        )
        for btn in (self._model_btn, self._hub_btn, self._docs_btn, self._profile_btn):
            btn.setFixedSize(btn_w, 38)

        main_layout.addLayout(top_row)

        # Chat display
        self._chat_display = QTextBrowser()
        self._chat_display.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._chat_display.setReadOnly(True)
        self._chat_display.setOpenLinks(False)
        self._chat_display.setOpenExternalLinks(False)
        self._chat_display.anchorClicked.connect(self._on_link_clicked)
        self._hovered_link_range = None
        self._chat_display.setPlaceholderText("Chat will appear here...")
        main_layout.addWidget(self._chat_display, stretch=1)

        # Input row
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        BTN_H = 38
        BTN_SPACING = 4

        btn_widget = QWidget()
        btn_column = QVBoxLayout(btn_widget)
        btn_column.setContentsMargins(0, 0, 0, 0)
        btn_column.setSpacing(BTN_SPACING)

        # Row 1: [■]
        self._stop_btn = QPushButton("\u25a0")
        self._stop_btn.setObjectName("stopButton")
        self._stop_btn.setFixedSize(BTN_H * 2 + 4, BTN_H)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_generation)
        btn_column.addWidget(self._stop_btn)

        # Row 2: [Send]
        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedSize(BTN_H * 2 + 4, BTN_H)
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._send_message)
        btn_column.addWidget(self._send_btn)

        input_row.addWidget(btn_widget)

        input_height = BTN_H * 2 + BTN_SPACING
        self._input_field = QTextEdit()
        self._input_field.setPlaceholderText("Type your message...")
        self._input_field.setFixedHeight(input_height)
        self._input_field.installEventFilter(self)
        self._chat_display.viewport().setMouseTracking(True)
        self._chat_display.viewport().installEventFilter(self)
        input_row.addWidget(self._input_field, stretch=1)

        main_layout.addLayout(input_row)
        outer.addWidget(main_widget, stretch=1)

    # ── Chat management ────────────────────────────────────────────────────

    def _init_chat(self):
        """Load the most recent chat on startup, or start with a blank one."""
        chats = list_chats()
        log("UI", f"_init_chat: found {len(chats)} existing chats")
        if chats:
            chat = load_chat(chats[0]["id"])
            if chat:
                log("UI", f"Loaded chat '{chat['title']}' with {len(chat['history'])} messages")
                self._current_chat = chat
                self._history = chat["history"]
                self._refresh_chat_list()
                self._load_chat_into_display(chat)
                return
        self._current_chat = new_chat()
        self._history = []
        self._refresh_chat_list()
        log("UI", "Started with new empty chat")

    def _toggle_sidebar(self):
        self._sidebar.setVisible(not self._sidebar.isVisible())

    def _new_chat(self):
        log("UI", "Creating new chat")
        self._save_current_chat()
        self._current_chat = new_chat()
        self._history = []
        self._thinking_blocks.clear()
        self._chat_display.clear()
        self._refresh_chat_list()

    def _save_current_chat(self):
        if self._history:
            self._current_chat["history"] = self._history
            save_chat(self._current_chat)

    def _refresh_chat_list(self):
        self._chat_list.blockSignals(True)
        self._chat_list.clear()
        for chat in list_chats():
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, chat["id"])
            item.setData(Qt.ItemDataRole.DisplayRole, "")
            item.setSizeHint(QSize(0, 42))
            self._chat_list.addItem(item)
            widget = ChatItemWidget(chat["id"], chat["title"])
            widget.rename_requested.connect(self._rename_chat_by_id)
            widget.delete_requested.connect(self._delete_chat_by_id)
            self._chat_list.setItemWidget(item, widget)
            if chat["id"] == self._current_chat["id"]:
                self._chat_list.setCurrentItem(item)
        self._chat_list.blockSignals(False)

    def _rename_chat_by_id(self, chat_id: str, current_title: str):
        dlg = RenameChatDialog(current_title, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            title = dlg.title()
            if title:
                rename_chat(chat_id, title)
                if chat_id == self._current_chat["id"]:
                    self._current_chat["title"] = title
                self._refresh_chat_list()

    def _delete_chat_by_id(self, chat_id: str):
        delete_chat(chat_id)
        if chat_id == self._current_chat["id"]:
            self._current_chat = new_chat()
            self._history = []
            self._thinking_blocks.clear()
            self._chat_display.clear()
        self._refresh_chat_list()

    def _on_chat_selected(self, item: QListWidgetItem):
        if item is None:
            return
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        if chat_id == self._current_chat["id"]:
            return
        log("UI", f"Switching to chat {chat_id}")
        self._save_current_chat()
        chat = load_chat(chat_id)
        if chat:
            self._current_chat = chat
            self._history = chat["history"]
            self._load_chat_into_display(chat)

    def _on_chat_rename(self, item: QListWidgetItem):
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        widget = self._chat_list.itemWidget(item)
        current_title = widget.title if widget else ""
        self._rename_chat_by_id(chat_id, current_title)

    def _on_chat_context_menu(self, pos):
        item = self._chat_list.itemAt(pos)
        if not item:
            return
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")
        action = menu.exec(self._chat_list.mapToGlobal(pos))
        if action == rename_action:
            self._on_chat_rename(item)
        elif action == delete_action:
            self._delete_chat_by_id(chat_id)

    def _reset_format(self):
        """Reset cursor formatting so appended text doesn't inherit HTML styles."""
        cursor = self._chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.setBlockFormat(QTextBlockFormat())
        cursor.setCharFormat(QTextCharFormat())
        self._chat_display.setTextCursor(cursor)

    def _load_chat_into_display(self, chat: dict):
        self._thinking_blocks.clear()
        self._chat_display.clear()
        first_msg = True
        for msg in chat["history"]:
            if msg["role"] == "user":
                self._reset_format()
                if not first_msg:
                    self._chat_display.append("<p>&nbsp;</p>")
                self._chat_display.append("<b style='color: #89b4fa;'>You</b>")
                self._chat_display.append(msg["content"])
                first_msg = False
            elif msg["role"] == "assistant":
                self._reset_format()
                if not first_msg:
                    self._chat_display.append("<p>&nbsp;</p>")
                model = msg.get("model", "")
                label = f"Assistant ({model})" if model else "Assistant"
                self._chat_display.append(f"<b style='color: #a6e3a1;'>{label}</b>")
                first_msg = False
                if thinking := msg.get("thinking"):
                    tid = self._thinking_id_counter
                    self._thinking_id_counter += 1
                    self._chat_display.append(
                        f"<a href='#thinking-{tid}' "
                        f"style='color: #6c7086; text-decoration: none;'>"
                        f"\u25b6 Thinking</a>"
                    )
                    cursor = self._chat_display.textCursor()
                    cursor.movePosition(cursor.MoveOperation.End)
                    self._thinking_blocks[tid] = {
                        "collapsed": True,
                        "header_block": cursor.blockNumber(),
                        "text": thinking,
                        "content_start": -1,
                        "content_end": -1,
                    }
                if msg.get("error"):
                    self._chat_display.append(f"<i style='color: #f38ba8;'>{msg['content']}</i>")
                else:
                    rendered = md_lib.markdown(
                        _strip_lists(_strip_latex(msg["content"])),
                        extensions=["tables"],
                    )
                    html = _style_tables(rendered)
                    cursor = self._chat_display.textCursor()
                    cursor.movePosition(cursor.MoveOperation.End)
                    cursor.insertBlock()
                    cursor.setBlockFormat(QTextBlockFormat())
                    cursor.setCharFormat(QTextCharFormat())
                    cursor.insertHtml(html)
                    cursor.movePosition(cursor.MoveOperation.End)
                    self._chat_display.setTextCursor(cursor)

    # ── Qt overrides ───────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        if (
            obj == self._input_field
            and event.type() == event.Type.KeyPress
            and (
                event.key() == Qt.Key.Key_Return
                and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            )
        ):
            self._send_message()
            return True

        if obj == self._chat_display.viewport() and event.type() == QEvent.Type.MouseMove:
            anchor = self._chat_display.anchorAt(event.pos())
            is_thinking = "thinking-" in anchor if anchor else False
            self._update_link_hover(event.pos() if is_thinking else None)

        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        log("UI", "Application closing")
        self._save_current_chat()
        stop_server()
        super().closeEvent(event)

    # ── System stats ────────────────────────────────────────────────────────

    def _update_stats(self):
        import psutil

        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        self._mem_chip.setText(f"Memory {used_gb:.1f}/{total_gb:.1f} GB")
        self._cpu_chip.setText(f"CPU {psutil.cpu_percent(interval=None):.0f}%")

    # ── Profile ────────────────────────────────────────────────────────────

    def _open_settings(self):
        from app.ui.settings_dialog import SettingsDialog

        dlg = SettingsDialog(parent=self)
        dlg.exec()

    def _open_profile(self):
        dlg = ProfileDialog(self)
        dlg.exec()

    def _build_profile_context(self) -> str:
        profile = load_profile()
        labels = {
            "name": "Name",
            "age": "Age",
            "gender": "Gender",
            "weight": "Weight",
            "height": "Height",
            "other": "Other",
        }
        lines = [f"{labels[k]}: {v}" for k, v in profile.items() if v]
        if not lines:
            return ""
        return "## Patient Profile\n\n" + "\n".join(lines)

    def _try_load_saved_model(self):
        saved_path = load_model_path()
        if saved_path:
            self._load_model(saved_path)
        else:
            self._ensure_default_model()

    def _ensure_default_model(self):
        """Auto-download the default model on first launch, then load it."""
        self._status_label.setText("Preparing default model...")
        self._model_btn.setEnabled(False)
        self._send_btn.setEnabled(False)

        self._download_worker = ModelDownloadWorker()
        self._download_worker.progress.connect(self._status_label.setText)
        self._download_worker.finished.connect(self._on_default_model_ready)
        self._download_worker.error_occurred.connect(self._on_default_model_error)
        self._download_worker.start()

    def _on_default_model_ready(self, model_path: str):
        save_model_path(model_path)
        self._load_model(model_path)

    def _on_default_model_error(self, error: str):
        log("UI", f"Default model download error: {error}")
        self._status_label.setText(
            "Failed to download default model — click 'Load Model' to select manually"
        )
        self._model_btn.setEnabled(True)

    def _select_model(self):
        from app.ui.model_select_dialog import ModelSelectDialog

        dlg = ModelSelectDialog(parent=self)
        if dlg.exec() and dlg.selected_path:
            save_model_path(dlg.selected_path)
            self._load_model(dlg.selected_path)

    def _open_model_hub(self):
        dlg = ModelHubDialog(self)
        dlg.model_downloaded.connect(self._on_hub_model_downloaded)
        dlg.exec()

    def _open_documents_hub(self):
        dlg = DocumentsHubDialog(self)
        dlg.exec()

    def _on_hub_model_downloaded(self, path: str):
        save_model_path(path)
        self._load_model(path)

    def _load_model(self, model_path: str):
        log("UI", f"_load_model: {model_path}")
        self._status_label.setText(f"Loading model: {Path(model_path).name}...")
        self._model_btn.setEnabled(False)
        self._send_btn.setEnabled(False)

        self._server_worker = ServerStartWorker(model_path)
        self._server_worker.progress.connect(self._status_label.setText)
        self._server_worker.meta_ready.connect(self._on_model_meta)
        self._server_worker.finished.connect(self._on_server_started)
        self._server_worker.error_occurred.connect(self._on_server_error)
        self._server_worker.start()

    def _on_model_meta(self, meta) -> None:
        log("UI", f"Model meta: ctx={meta.context_length}")

    def _on_server_started(self, model_name: str):
        log("UI", f"Server started, model: {model_name}")
        self._model_name = model_name
        self._status_label.setText(f"Model: {model_name}")
        self._model_btn.setEnabled(True)
        self._send_btn.setEnabled(True)

    def _on_server_error(self, error: str):
        log("UI", f"Server error: {error}")
        self._status_label.setText(f"Model load error: {error}")
        self._model_btn.setEnabled(True)

    # ── Messaging ──────────────────────────────────────────────────────────

    def _send_message(self):
        prompt = self._input_field.toPlainText().strip()
        if not prompt or not is_server_running():
            return
        log("UI", f"_send_message: '{prompt[:80]}...'")

        self._input_field.clear()
        self._send_btn.setEnabled(False)

        self._reset_format()
        if self._history:
            self._chat_display.append("<p>&nbsp;</p>")
        self._chat_display.append("<b style='color: #89b4fa;'>You</b>")
        self._chat_display.append(f"{prompt}")
        self._history.append({"role": "user", "content": prompt})

        # Set chat title from the first message
        if len(self._history) == 1:
            self._current_chat["title"] = title_from_first_message(prompt)
            self._refresh_chat_list()

        self._compression_attempted = False

        context = self._build_profile_context()
        self._launch_llm_worker(context)

    def _launch_llm_worker(self, context: str = ""):
        from app.config import load_max_tokens

        log(
            "UI",
            f"_launch_llm_worker: context={len(context)} chars, history={len(self._history)} msgs",
        )
        self._thinking = True
        self._thinking_text = ""
        self._current_response = ""
        self._response_anchor = 0
        self._generation_stopped = False
        max_tokens = load_max_tokens()
        self._worker = LLMWorker(list(self._history), context=context, max_tokens=max_tokens)
        self._worker.thinking_token.connect(self._on_thinking_token)
        self._worker.response_token.connect(self._on_response_token)
        self._worker.finished_generation.connect(self._on_generation_done)
        self._worker.error_occurred.connect(self._on_generation_error)
        self._stop_btn.setEnabled(True)
        self._worker.start()

    def _stop_generation(self):
        log("UI", "Generation stopped by user")
        self._generation_stopped = True
        self._stop_btn.setEnabled(False)
        if self._worker:
            self._worker.stop()

    def _update_link_hover(self, pos):
        """Highlight or unhighlight the thinking link under the cursor."""
        # Clear previous highlight
        if self._hovered_link_range is not None:
            start, end = self._hovered_link_range
            cursor = self._chat_display.textCursor()
            cursor.setPosition(start)
            cursor.setPosition(end, cursor.MoveMode.KeepAnchor)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#6c7086"))
            cursor.mergeCharFormat(fmt)
            self._hovered_link_range = None

        if pos is None:
            return

        # Find the anchor text range and highlight it
        cursor = self._chat_display.cursorForPosition(pos)
        block = cursor.block()
        it = block.begin()
        while not it.atEnd():
            fragment = it.fragment()
            if fragment.isValid() and fragment.charFormat().anchorHref():
                start = fragment.position()
                end = start + fragment.length()
                c = self._chat_display.textCursor()
                c.setPosition(start)
                c.setPosition(end, c.MoveMode.KeepAnchor)
                fmt = QTextCharFormat()
                fmt.setForeground(QColor("#a6e3a1"))
                c.mergeCharFormat(fmt)
                self._hovered_link_range = (start, end)
                return
            it += 1

    def _on_link_clicked(self, url):
        frag = url.fragment()
        if not frag.startswith("thinking-"):
            return
        try:
            tid = int(frag.split("-", 1)[1])
        except (ValueError, IndexError):
            return
        info = self._thinking_blocks.get(tid)
        if not info:
            return
        self._toggle_thinking(tid, info)

    def _toggle_thinking(self, tid: int, info: dict):
        doc = self._chat_display.document()
        header_block = doc.findBlockByNumber(info["header_block"])
        if not header_block.isValid():
            return

        old_char_count = doc.characterCount()
        cursor = self._chat_display.textCursor()

        if info["collapsed"]:
            # Expand: insert thinking text after header
            cursor.setPosition(header_block.position())
            cursor.movePosition(cursor.MoveOperation.EndOfBlock)
            cursor.insertBlock()
            start_block = cursor.blockNumber()
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#6c7086"))
            cursor.insertText(info["text"], fmt)
            end_block = cursor.blockNumber()
            info["content_start"] = start_block
            info["content_end"] = end_block
            info["collapsed"] = False
            inserted = end_block - start_block + 1
            for other_id, other in self._thinking_blocks.items():
                if other_id != tid and other["header_block"] > info["header_block"]:
                    other["header_block"] += inserted
                    if not other["collapsed"]:
                        other["content_start"] += inserted
                        other["content_end"] += inserted
        else:
            # Collapse: remove content blocks
            start_blk = doc.findBlockByNumber(info["content_start"])
            end_blk = doc.findBlockByNumber(info["content_end"])
            if start_blk.isValid() and end_blk.isValid():
                cursor.setPosition(start_blk.position() - 1)
                cursor.setPosition(
                    end_blk.position() + end_blk.length() - 1,
                    cursor.MoveMode.KeepAnchor,
                )
                cursor.removeSelectedText()
                removed = info["content_end"] - info["content_start"] + 1
                for other_id, other in self._thinking_blocks.items():
                    if other_id != tid and other["header_block"] > info["header_block"]:
                        other["header_block"] -= removed
                        if not other["collapsed"]:
                            other["content_start"] -= removed
                            other["content_end"] -= removed
            info["collapsed"] = True

        # Adjust response anchor if response is being streamed
        delta = doc.characterCount() - old_char_count
        if self._response_anchor > 0:
            self._response_anchor += delta

        # Update header arrow
        header_block = doc.findBlockByNumber(info["header_block"])
        if header_block.isValid():
            cursor.setPosition(header_block.position())
            cursor.movePosition(
                cursor.MoveOperation.EndOfBlock,
                cursor.MoveMode.KeepAnchor,
            )
            arrow = "\u25b6" if info["collapsed"] else "\u25bc"
            cursor.insertHtml(
                f"<a href='#thinking-{tid}' "
                f"style='color: #6c7086; text-decoration: none;'>"
                f"{arrow} Thinking...</a>"
            )

        self._chat_display.setTextCursor(cursor)
        self._chat_display.viewport().update()

    def _on_thinking_token(self, token: str):
        first = not self._thinking_text
        self._thinking_text += token

        if first:
            tid = self._thinking_id_counter
            self._thinking_id_counter += 1
            self._current_thinking_id = tid
            self._chat_display.append("<p>&nbsp;</p>")
            label = f"Assistant ({self._model_name})" if self._model_name else "Assistant"
            self._chat_display.append(f"<b style='color: #a6e3a1;'>{label}</b>")
            self._chat_display.append(
                f"<a href='#thinking-{tid}' "
                f"style='color: #6c7086; text-decoration: none;'>"
                f"\u25b6 Thinking</a>"
            )
            cursor = self._chat_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self._thinking_blocks[tid] = {
                "collapsed": True,
                "header_block": cursor.blockNumber(),
                "text": self._thinking_text,
                "content_start": -1,
                "content_end": -1,
            }
            return

        info = self._thinking_blocks[self._current_thinking_id]
        info["text"] = self._thinking_text

        if not info["collapsed"]:
            scrollbar = self._chat_display.verticalScrollBar()
            at_bottom = scrollbar.value() >= scrollbar.maximum() - 20
            cursor = self._chat_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#6c7086"))
            cursor.insertText(token, fmt)
            info["content_end"] = cursor.blockNumber()
            if at_bottom:
                scrollbar.setValue(scrollbar.maximum())

    def _on_response_token(self, token: str):
        if self._thinking:
            self._thinking = False

            if self._thinking_text:
                info = self._thinking_blocks[self._current_thinking_id]
                info["text"] = self._thinking_text
            else:
                # Non-thinking model
                self._chat_display.append("<p>&nbsp;</p>")
                label = f"Assistant ({self._model_name})" if self._model_name else "Assistant"
                self._chat_display.append(f"<b style='color: #a6e3a1;'>{label}</b>")

            cursor = self._chat_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self._response_anchor = cursor.position()

        self._current_response += token
        rendered = md_lib.markdown(
            _strip_lists(_strip_latex(self._current_response)),
            extensions=["tables"],
        )
        html = _style_tables(rendered)

        scrollbar = self._chat_display.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 20
        saved_scroll = scrollbar.value()

        cursor = self._chat_display.textCursor()
        if self._response_anchor <= 0:
            log("UI", "WARNING: _response_anchor is 0, skipping response render")
            return
        cursor.setPosition(self._response_anchor)
        cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.setBlockFormat(QTextBlockFormat())
        cursor.insertBlock()
        cursor.setBlockFormat(QTextBlockFormat())
        cursor.insertHtml(html)
        self._chat_display.setTextCursor(cursor)

        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(saved_scroll)

    def _on_generation_done(self):
        log(
            "UI",
            f"Generation done, response={len(self._current_response)} chars, "
            f"stopped={self._generation_stopped}",
        )
        if self._thinking:
            self._thinking = False
            self._status_label.setText(f"Model: {self._model_name}")
        self._stop_btn.setEnabled(False)
        if self._generation_stopped:
            if self._history and self._history[-1]["role"] == "user":
                self._history.pop()
        elif self._current_response:
            msg = {
                "role": "assistant",
                "content": self._current_response,
                "model": self._model_name,
            }
            if self._thinking_text:
                msg["thinking"] = self._thinking_text
            self._history.append(msg)
            self._save_current_chat()
            self._refresh_chat_list()
        elif self._thinking_text:
            # Model produced thinking but no response (ran out of tokens)
            error_text = (
                "Model used all available tokens on reasoning and produced "
                "no response. Try increasing Max Tokens in Settings or "
                "simplifying your question."
            )
            self._chat_display.append(f"<i style='color: #f38ba8;'>{error_text}</i>")
            self._history.append(
                {
                    "role": "assistant",
                    "content": error_text,
                    "thinking": self._thinking_text,
                    "model": self._model_name,
                    "error": True,
                }
            )
            self._save_current_chat()
            self._refresh_chat_list()

        self._send_btn.setEnabled(True)

    def _on_generation_error(self, error: str):
        log("UI", f"Generation error: {error}")
        self._thinking = False
        self._stop_btn.setEnabled(False)
        self._status_label.setText(f"Model: {self._model_name}")

        error_lower = error.lower()
        is_ctx_overflow = any(
            kw in error_lower
            for kw in (
                "context",
                "token",
                "length",
                "exceed",
                "too long",
                "413",
                "400",
                "bad request",
            )
        )

        if is_ctx_overflow and len(self._history) > 1 and not self._compression_attempted:
            self._pending_prompt = ""
            if self._history and self._history[-1]["role"] == "user":
                self._pending_prompt = self._history[-1]["content"]
                history_to_compress = self._history[:-1]
            else:
                history_to_compress = list(self._history)

            self._chat_display.append(
                "<b style='color: #a6e3a1;'>Assistant:</b> "
                "<i style='color: #6c7086;'>Compressing conversation history...</i>"
            )

            self._compression_attempted = True
            self._compression_worker = CompressionWorker(history_to_compress)
            self._compression_worker.finished.connect(self._on_compression_done)
            self._compression_worker.error_occurred.connect(self._on_compression_error)
            self._compression_worker.start()
        else:
            if self._history and self._history[-1]["role"] == "user":
                self._history.pop()
            self._chat_display.append(f"\n<i style='color: #f38ba8;'>Error: {error}</i>\n")
            self._send_btn.setEnabled(True)

    def _on_compression_done(self, summary: str):
        log("UI", f"Compression done, summary={len(summary)} chars, retrying with pending prompt")
        self._history = [
            {"role": "user", "content": f"[Summary of previous conversation]\n\n{summary}"},
            {
                "role": "assistant",
                "content": "Understood. I have the context from the previous conversation.",
            },
            {"role": "user", "content": self._pending_prompt},
        ]
        cursor = self._chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        self._chat_display.setTextCursor(cursor)
        self._launch_llm_worker()

    def _on_compression_error(self, error: str):
        log("UI", f"Compression error: {error}")
        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()
        cursor = self._chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        self._chat_display.setTextCursor(cursor)
        self._chat_display.append(
            "<b style='color: #a6e3a1;'>Assistant</b><br>"
            "<i style='color: #f38ba8;'>"
            "Could not compress history. Please start a new conversation."
            "</i>"
        )
        self._send_btn.setEnabled(True)
