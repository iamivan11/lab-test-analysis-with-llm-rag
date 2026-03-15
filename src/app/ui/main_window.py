import json
import threading
from pathlib import Path

import markdown as md_lib
from PySide6.QtCore import QSize, QThread, Signal, Qt
from PySide6.QtGui import QTextBlockFormat
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from app.config import APP_NAME, MODELS_DIR, load_model_path, load_profile, save_model_path
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
from app.core.pdf_parser import SUPPORTED_EXTENSIONS, parse_document_to_markdown
from app.ui.profile_dialog import ProfileDialog
from app.ui.styles import STYLESHEET

MAX_ATTACHMENTS = 10

FILE_FILTER = "Supported Files ({});;All Files (*)".format(
    " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
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

        self._menu_btn = QPushButton("\u22EF")
        self._menu_btn.setObjectName("chatMenuButton")
        self._menu_btn.setFixedSize(24, 24)
        self._menu_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._menu_btn.setVisible(False)
        self._menu_btn.clicked.connect(self._show_menu)
        layout.addWidget(self._menu_btn)

    def _show_menu(self):
        menu = QMenu(self)
        rename_action = menu.addAction("Rename")

        delete_label = QLabel("Delete")
        delete_label.setStyleSheet(
            "color: #f38ba8; padding: 6px 16px; border-radius: 4px;"
        )
        delete_widget_action = QWidgetAction(menu)
        delete_widget_action.setDefaultWidget(delete_label)
        menu.addAction(delete_widget_action)

        action = menu.exec(self._menu_btn.mapToGlobal(
            self._menu_btn.rect().bottomLeft()
        ))
        if action == rename_action:
            self.rename_requested.emit(self._chat_id, self._title)
        elif action == delete_widget_action:
            self.delete_requested.emit(self._chat_id)

    def enterEvent(self, event):
        self._menu_btn.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._menu_btn.setVisible(False)
        super().leaveEvent(event)


class ServerStartWorker(QThread):
    progress = Signal(str)
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            start_server(self.model_path, on_progress=self.progress.emit)
            self.finished.emit(Path(self.model_path).name)
        except Exception as e:
            self.error_occurred.emit(str(e))


class LLMWorker(QThread):
    thinking_token = Signal(str)
    response_token = Signal(str)
    finished_generation = Signal()
    error_occurred = Signal(str)

    def __init__(self, history: list[dict], context: str = "", thinking: bool = True):
        super().__init__()
        self.history = history
        self.context = context
        self.thinking = thinking
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            for kind, token in generate_stream(
                self.history, context=self.context, thinking=self.thinking,
                stop_event=self._stop_event,
            ):
                if self._stop_event.is_set():
                    break
                if kind == "thinking":
                    self.thinking_token.emit(token)
                else:
                    self.response_token.emit(token)
            self.finished_generation.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))


class FileParserWorker(QThread):
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, file_paths: list[str]):
        super().__init__()
        self.file_paths = file_paths

    def run(self):
        try:
            result: dict[str, str] = {}
            for path in self.file_paths:
                result[path] = parse_document_to_markdown(path)
            self.finished.emit(json.dumps(result))
        except Exception as e:
            self.error_occurred.emit(str(e))


class CompressionWorker(QThread):
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, history: list[dict]):
        super().__init__()
        self.history = history

    def run(self):
        try:
            summary = summarize_history(self.history)
            self.finished.emit(summary)
        except Exception as e:
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

        self._file_contents: dict[str, str] = {}
        self._history: list[dict] = []
        self._current_chat: dict = new_chat()
        self._thinking = False
        self._model_name = ""
        self._current_response = ""
        self._response_anchor = 0
        self._reasoning_shown = False
        self._worker = None
        self._file_worker = None
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
        self._sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(12, 16, 12, 16)
        sidebar_layout.setSpacing(8)

        new_chat_btn = QPushButton("New Chat")
        new_chat_btn.setObjectName("attachButton")
        new_chat_btn.clicked.connect(self._new_chat)
        sidebar_layout.addWidget(new_chat_btn)

        self._chat_list = QListWidget()
        self._chat_list.setObjectName("chatList")
        self._chat_list.itemClicked.connect(self._on_chat_selected)
        self._chat_list.itemDoubleClicked.connect(self._on_chat_rename)
        self._chat_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._chat_list.customContextMenuRequested.connect(self._on_chat_context_menu)
        sidebar_layout.addWidget(self._chat_list)

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

        self._model_btn = QPushButton("Load Model")
        self._model_btn.setObjectName("attachButton")
        self._model_btn.clicked.connect(self._select_model)
        top_row.addWidget(self._model_btn)

        self._status_label = QLabel("No model loaded")
        self._status_label.setObjectName("statusLabel")
        top_row.addWidget(self._status_label, stretch=1)

        self._reasoning_btn = QPushButton("Reasoning")
        self._reasoning_btn.setObjectName("reasoningButton")
        self._reasoning_btn.setCheckable(True)
        self._reasoning_btn.setChecked(False)
        top_row.addWidget(self._reasoning_btn)

        self._profile_btn = QPushButton("Profile")
        self._profile_btn.setObjectName("attachButton")
        self._profile_btn.clicked.connect(self._open_profile)
        self._profile_btn.setMinimumWidth(self._reasoning_btn.sizeHint().width())
        top_row.addWidget(self._profile_btn)

        main_layout.addLayout(top_row)

        # Chat display
        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setPlaceholderText("Chat will appear here...")
        main_layout.addWidget(self._chat_display, stretch=1)

        # File chips area
        self._chips_widget = QWidget()
        self._chips_widget.setVisible(False)
        self._chips_layout = QHBoxLayout(self._chips_widget)
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(4)
        self._chips_layout.addStretch()
        main_layout.addWidget(self._chips_widget)

        # Parsing status label
        self._parse_label = QLabel("")
        self._parse_label.setObjectName("fileLabel")
        self._parse_label.setVisible(False)
        main_layout.addWidget(self._parse_label)

        # Input row
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        btn_widget = QWidget()
        btn_widget.setFixedWidth(80)
        btn_column = QVBoxLayout(btn_widget)
        btn_column.setContentsMargins(0, 0, 0, 0)
        btn_column.setSpacing(12)

        top_btn_row = QHBoxLayout()
        top_btn_row.setSpacing(4)

        self._attach_btn = QPushButton("+")
        self._attach_btn.setObjectName("attachButton")
        self._attach_btn.setFixedSize(38, 38)
        self._attach_btn.clicked.connect(self._attach_files)
        top_btn_row.addWidget(self._attach_btn)

        self._stop_btn = QPushButton("■")
        self._stop_btn.setObjectName("stopButton")
        self._stop_btn.setFixedSize(38, 38)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_generation)
        top_btn_row.addWidget(self._stop_btn)

        btn_column.addLayout(top_btn_row)

        self._send_btn = QPushButton("Send")
        self._send_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._send_message)
        btn_column.addWidget(self._send_btn)

        input_row.addWidget(btn_widget)

        self._input_field = QTextEdit()
        self._input_field.setPlaceholderText("Type your message...")
        self._input_field.setFixedHeight(96)
        self._input_field.installEventFilter(self)
        input_row.addWidget(self._input_field, stretch=1)

        main_layout.addLayout(input_row)
        outer.addWidget(main_widget, stretch=1)

    # ── Chat management ────────────────────────────────────────────────────

    def _init_chat(self):
        """Load the most recent chat on startup, or start with a blank one."""
        chats = list_chats()
        if chats:
            chat = load_chat(chats[0]["id"])
            if chat:
                self._current_chat = chat
                self._history = chat["history"]
                self._refresh_chat_list()
                self._load_chat_into_display(chat)
                return
        self._current_chat = new_chat()
        self._history = []
        self._refresh_chat_list()

    def _toggle_sidebar(self):
        self._sidebar.setVisible(not self._sidebar.isVisible())

    def _new_chat(self):
        self._save_current_chat()
        self._current_chat = new_chat()
        self._history = []
        self._chat_display.clear()
        self._refresh_chat_list()

    def _save_current_chat(self):
        if self._history:
            self._current_chat["history"] = self._history
            save_chat(self._current_chat)

    def _refresh_chat_list(self):
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
            self._chat_display.clear()
        self._refresh_chat_list()

    def _on_chat_selected(self, item: QListWidgetItem):
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        if chat_id == self._current_chat["id"]:
            return
        self._save_current_chat()
        chat = load_chat(chat_id)
        if chat:
            self._current_chat = chat
            self._history = chat["history"]
            self._load_chat_into_display(chat)

    def _on_chat_rename(self, item: QListWidgetItem):
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        widget = self._chat_list.itemWidget(item)
        current_title = widget._title if widget else ""
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

    def _load_chat_into_display(self, chat: dict):
        self._chat_display.clear()
        for msg in chat["history"]:
            if msg["role"] == "user":
                self._chat_display.append("<b style='color: #89b4fa;'>You</b>")
                self._chat_display.append(msg["content"])
                self._chat_display.append("<p>&nbsp;</p>")
            elif msg["role"] == "assistant":
                self._chat_display.append("<b style='color: #a6e3a1;'>Assistant</b>")
                html = md_lib.markdown(msg["content"], extensions=["tables"])
                cursor = self._chat_display.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                cursor.insertBlock()
                cursor.setBlockFormat(QTextBlockFormat())
                cursor.insertHtml(html)
                cursor.movePosition(cursor.MoveOperation.End)
                cursor.insertBlock()
                active_list = cursor.currentList()
                if active_list:
                    active_list.remove(cursor.block())
                cursor.setBlockFormat(QTextBlockFormat())
                self._chat_display.setTextCursor(cursor)
                self._chat_display.append("<p>&nbsp;</p>")

    # ── Qt overrides ───────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        if obj == self._input_field and event.type() == event.Type.KeyPress:
            if (
                event.key() == Qt.Key.Key_Return
                and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            ):
                self._send_message()
                return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        self._save_current_chat()
        stop_server()
        super().closeEvent(event)

    # ── Profile ────────────────────────────────────────────────────────────

    def _open_profile(self):
        dlg = ProfileDialog(self)
        dlg.exec()

    def _build_profile_context(self) -> str:
        profile = load_profile()
        labels = {
            "name": "Name", "age": "Age", "gender": "Gender",
            "weight": "Weight", "height": "Height", "other": "Other",
        }
        lines = [f"{labels[k]}: {v}" for k, v in profile.items() if v]
        if not lines:
            return ""
        return "## Patient Profile\n\n" + "\n".join(lines)

    # ── File handling ──────────────────────────────────────────────────────

    def _build_file_context(self) -> str:
        parts = []
        for path, md_text in self._file_contents.items():
            filename = Path(path).name
            parts.append(f"## {filename}\n\n{md_text}")
        return "\n\n---\n\n".join(parts)

    def _rebuild_chips(self):
        while self._chips_layout.count() > 1:
            item = self._chips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for path in self._file_contents:
            name = Path(path).name
            btn = QPushButton(f"{name}  ×")
            btn.setObjectName("fileChip")
            btn.clicked.connect(lambda checked, p=path: self._remove_file(p))
            self._chips_layout.insertWidget(self._chips_layout.count() - 1, btn)

        self._chips_widget.setVisible(bool(self._file_contents))

    def _remove_file(self, path: str):
        self._file_contents.pop(path, None)
        self._rebuild_chips()

    def _try_load_saved_model(self):
        saved_path = load_model_path()
        if saved_path:
            self._load_model(saved_path)
        else:
            self._status_label.setText(
                "No model loaded — click 'Load Model' to select a GGUF file"
            )

    def _select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model File", str(MODELS_DIR),
            "GGUF Files (*.gguf);;All Files (*)",
        )
        if file_path:
            save_model_path(file_path)
            self._load_model(file_path)

    def _load_model(self, model_path: str):
        self._status_label.setText(f"Loading model: {Path(model_path).name}...")
        self._model_btn.setEnabled(False)
        self._send_btn.setEnabled(False)

        self._server_worker = ServerStartWorker(model_path)
        self._server_worker.progress.connect(self._status_label.setText)
        self._server_worker.finished.connect(self._on_server_started)
        self._server_worker.error_occurred.connect(self._on_server_error)
        self._server_worker.start()

    def _on_server_started(self, model_name: str):
        self._status_label.setText(f"Model: {model_name}")
        self._model_btn.setEnabled(True)
        self._send_btn.setEnabled(True)

    def _on_server_error(self, error: str):
        self._status_label.setText(f"Model load error: {error}")
        self._model_btn.setEnabled(True)

    def _attach_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Lab Report Files", "", FILE_FILTER
        )
        if not file_paths:
            return

        file_paths = file_paths[:MAX_ATTACHMENTS]
        names = [Path(p).name for p in file_paths]
        self._parse_label.setText(f"Parsing: {', '.join(names)}...")
        self._parse_label.setVisible(True)
        self._attach_btn.setEnabled(False)

        self._file_worker = FileParserWorker(file_paths)
        self._file_worker.finished.connect(self._on_files_parsed)
        self._file_worker.error_occurred.connect(self._on_files_error)
        self._file_worker.start()

    def _on_files_parsed(self, data: str):
        self._file_contents.update(json.loads(data))
        self._parse_label.setVisible(False)
        self._rebuild_chips()
        self._attach_btn.setEnabled(True)

    def _on_files_error(self, error: str):
        self._parse_label.setText(f"Parse error: {error}")
        self._file_contents = {}
        self._rebuild_chips()
        self._attach_btn.setEnabled(True)

    # ── Messaging ──────────────────────────────────────────────────────────

    def _send_message(self):
        prompt = self._input_field.toPlainText().strip()
        if not prompt or not is_server_running():
            return

        self._input_field.clear()
        self._send_btn.setEnabled(False)
        self._attach_btn.setEnabled(False)

        self._chat_display.append("<b style='color: #89b4fa;'>You</b>")
        if self._file_contents:
            names = [Path(p).name for p in self._file_contents]
            self._chat_display.append(
                f"{prompt}<br><i style='color: #6c7086; font-size: 12px;'>"
                f"with {', '.join(names)}</i>"
            )
        else:
            self._chat_display.append(f"{prompt}")
        self._chat_display.append("<p>&nbsp;</p>")
        self._history.append({"role": "user", "content": prompt})

        # Set chat title from the first message
        if len(self._history) == 1:
            self._current_chat["title"] = title_from_first_message(prompt)
            self._refresh_chat_list()

        self._model_name = Path(self._server_worker.model_path).name
        self._compression_attempted = False

        parts = [p for p in (self._build_profile_context(), self._build_file_context()) if p]
        context = "\n\n---\n\n".join(parts)
        self._file_contents.clear()
        self._rebuild_chips()
        self._launch_llm_worker(context)

    def _launch_llm_worker(self, context: str = ""):
        self._thinking = True
        self._reasoning_shown = False
        self._current_response = ""
        self._response_anchor = 0
        self._generation_stopped = False
        self._worker = LLMWorker(
            list(self._history), context=context, thinking=self._reasoning_btn.isChecked()
        )
        self._worker.thinking_token.connect(self._on_thinking_token)
        self._worker.response_token.connect(self._on_response_token)
        self._worker.finished_generation.connect(self._on_generation_done)
        self._worker.error_occurred.connect(self._on_generation_error)
        self._stop_btn.setEnabled(True)
        self._worker.start()

    def _stop_generation(self):
        self._generation_stopped = True
        self._stop_btn.setEnabled(False)
        if self._worker:
            self._worker.stop()

    def _on_thinking_token(self, _token: str):
        if not self._reasoning_btn.isChecked():
            return
        if self._thinking and not self._reasoning_shown:
            self._reasoning_shown = True
            self._chat_display.append("<b style='color: #a6e3a1;'>Assistant</b>")
            cursor = self._chat_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertBlock()
            cursor.insertHtml("<i style='color: #6c7086;'>Reasoning...</i>")
            self._chat_display.setTextCursor(cursor)

    def _on_response_token(self, token: str):
        if self._thinking:
            self._thinking = False
            self._status_label.setText(f"Model: {self._model_name}")
            if self._reasoning_shown:
                cursor = self._chat_display.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                cursor.select(cursor.SelectionType.BlockUnderCursor)
                cursor.removeSelectedText()
                cursor.deletePreviousChar()
                self._chat_display.setTextCursor(cursor)
            else:
                self._chat_display.append("<b style='color: #a6e3a1;'>Assistant</b>")
            cursor = self._chat_display.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self._response_anchor = cursor.position()

        self._current_response += token
        html = md_lib.markdown(self._current_response, extensions=["tables"])

        scrollbar = self._chat_display.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 20
        saved_scroll = scrollbar.value()

        cursor = self._chat_display.textCursor()
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
        if self._thinking:
            self._thinking = False
            self._status_label.setText(f"Model: {self._model_name}")
        self._stop_btn.setEnabled(False)
        if self._generation_stopped:
            if self._history and self._history[-1]["role"] == "user":
                self._history.pop()
        elif self._current_response:
            self._history.append({"role": "assistant", "content": self._current_response})
            self._save_current_chat()
            self._refresh_chat_list()

        # Add a blank separator line after the response.
        # For list responses, zero out the list item's bottom margin first so
        # the separator block doesn't stack with it and produce double spacing.
        cursor = self._chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        active_list = cursor.currentList()
        if active_list:
            fmt = cursor.blockFormat()
            fmt.setBottomMargin(0)
            cursor.setBlockFormat(fmt)
            active_list.remove(cursor.block())
        cursor.insertBlock()
        cursor.setBlockFormat(QTextBlockFormat())
        self._chat_display.setTextCursor(cursor)

        self._send_btn.setEnabled(True)
        self._attach_btn.setEnabled(True)

    def _on_generation_error(self, error: str):
        self._thinking = False
        self._stop_btn.setEnabled(False)
        self._status_label.setText(f"Model: {self._model_name}")

        error_lower = error.lower()
        is_ctx_overflow = any(
            kw in error_lower for kw in (
                "context", "token", "length", "exceed", "too long", "413", "400", "bad request"
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
                "<b style='color: #a6e3a1;'>Assistant:</b>"
                " <i style='color: #6c7086;'>Compressing conversation history...</i>"
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
            self._attach_btn.setEnabled(True)

    def _on_compression_done(self, summary: str):
        self._history = [
            {"role": "user", "content": f"[Summary of previous conversation]\n\n{summary}"},
            {"role": "assistant", "content": "Understood. I have the context from the previous conversation."},
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
        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()
        cursor = self._chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        self._chat_display.setTextCursor(cursor)
        self._chat_display.append(
            "<b style='color: #a6e3a1;'>Assistant</b>"
            "<br><i style='color: #f38ba8;'>Could not compress history. Please start a new conversation.</i>"
        )
        self._send_btn.setEnabled(True)
        self._attach_btn.setEnabled(True)
