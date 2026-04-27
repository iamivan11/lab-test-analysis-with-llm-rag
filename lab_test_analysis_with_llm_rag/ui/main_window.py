from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QPushButton,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import APP_NAME
from core.chat_store import list_chats, load_chat, new_chat
from core.llm_engine import stop_server
from core.logger import log
from ui.chat import ChatController
from ui.documents import DocumentsController
from ui.models import ModelController, ModelHubDialog
from ui.profile import ProfileController
from ui.styles import STYLESHEET


class PlainTextPasteEdit(QTextEdit):
    """QTextEdit that always pastes plain text from the clipboard."""

    def insertFromMimeData(self, source) -> None:
        self.insertPlainText(source.text())


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
        self._docs_dialog = None
        self._hub_dialog: ModelHubDialog | None = None
        self._parsing_active = False
        self._download_worker = None
        self._load_token = 0
        self._chat_controller = ChatController(self)
        self._documents_controller = DocumentsController(self)
        self._model_controller = ModelController(self)
        self._profile_controller = ProfileController(self)

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
        self._model_btn.setObjectName("loadModelButton")
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
        self._chat_display = ChatDisplayBrowser()
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
        self._input_field = PlainTextPasteEdit()
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
        self._chat_controller.new_chat()

    def _save_current_chat(self):
        self._chat_controller.save_current_chat()

    def _refresh_chat_list(self):
        self._chat_controller.refresh_chat_list()

    def _rename_chat_by_id(self, chat_id: str, current_title: str):
        self._chat_controller.rename_chat_by_id(chat_id, current_title)

    def _delete_chat_by_id(self, chat_id: str):
        self._chat_controller.delete_chat_by_id(chat_id)

    def _on_chat_selected(self, item):
        self._chat_controller.on_chat_selected(item)

    def _on_chat_rename(self, item):
        self._chat_controller.on_chat_rename(item)

    def _on_chat_context_menu(self, pos):
        self._chat_controller.on_chat_context_menu(pos)

    def _reset_format(self):
        self._chat_controller.reset_format()

    def _load_chat_into_display(self, chat: dict):
        self._chat_controller.load_chat_into_display(chat)

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
            self._chat_controller.send_message()
            return True

        if obj == self._chat_display.viewport() and event.type() == QEvent.Type.MouseMove:
            anchor = self._chat_display.anchorAt(event.pos())
            is_thinking = "thinking-" in anchor if anchor else False
            self._chat_controller.update_link_hover(event.pos() if is_thinking else None)

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
        from ui.settings_dialog import SettingsDialog

        dlg = SettingsDialog(parent=self)
        dlg.exec()

    def _open_profile(self):
        self._profile_controller.open_profile()

    def _build_profile_context(self) -> str:
        return self._profile_controller.build_profile_context()

    def _try_load_saved_model(self):
        self._model_controller.try_load_saved_model()

    def _new_load_token(self) -> int:
        return self._model_controller.new_load_token()

    def _ensure_default_model(self):
        self._model_controller.ensure_default_model()

    def _on_default_model_ready(self, model_path: str):
        self._model_controller.on_default_model_ready(model_path)

    def _on_default_model_error(self, error: str):
        self._model_controller.on_default_model_error(error)

    def _select_model(self):
        self._model_controller.select_model()

    def _open_model_hub(self):
        if self._hub_dialog is None:
            self._hub_dialog = ModelHubDialog(self)
        self._hub_dialog.show()
        self._hub_dialog.raise_()
        self._hub_dialog.activateWindow()

    def _open_documents_hub(self):
        self._documents_controller.open_documents_hub()

    def _on_docs_model_swapped(self, model_path: str, display_name: str):
        self._documents_controller.on_docs_model_swapped(model_path, display_name)

    def _on_parsing_active_changed(self, active: bool):
        self._documents_controller.on_parsing_active_changed(active)

    def _load_model(self, model_path: str, token: int | None = None):
        self._model_controller.load_model(model_path, token=token)

    def _on_model_meta(self, meta) -> None:
        self._model_controller.on_model_meta(meta)

    def _on_server_started(self, model_name: str):
        self._model_controller.on_server_started(model_name)

    def _on_server_error(self, error: str):
        self._model_controller.on_server_error(error)

    # ── Messaging ──────────────────────────────────────────────────────────

    def _send_message(self):
        self._chat_controller.send_message()

    def _reply_chat_disabled(self):
        self._chat_controller.reply_chat_disabled()

    def _launch_llm_worker(self, context: str = ""):
        self._chat_controller.launch_llm_worker(context)

    def _stop_generation(self):
        self._chat_controller.stop_generation()

    def _update_link_hover(self, pos):
        self._chat_controller.update_link_hover(pos)

    def _on_link_clicked(self, url):
        self._chat_controller.on_link_clicked(url)

    def _toggle_thinking(self, tid: int, info: dict):
        self._chat_controller.toggle_thinking(tid, info)

    def _on_thinking_token(self, token: str):
        self._chat_controller.on_thinking_token(token)

    def _on_response_token(self, token: str):
        self._chat_controller.on_response_token(token)

    def _on_generation_done(self):
        self._chat_controller.on_generation_done()

    def _on_generation_error(self, error: str):
        self._chat_controller.on_generation_error(error)

    def _on_compression_done(self, summary: str):
        self._chat_controller.on_compression_done(summary)

    def _on_compression_error(self, error: str):
        self._chat_controller.on_compression_error(error)
