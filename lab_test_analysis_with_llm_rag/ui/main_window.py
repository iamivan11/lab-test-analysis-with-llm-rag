from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from config import (
    APP_NAME,
    list_uploaded_doc_paths,
    is_onboarding_complete,
    save_model_path,
    set_onboarding_complete,
)
from core.chat_store import list_chats, load_chat, new_chat
from core.llm_engine import stop_server
from core.logger import log
from core.security import (
    SecurityError,
    migrate_known_sensitive_files,
    unlock,
)
from ui.chat import ChatController
from ui.chat.view import AutoHideScrollListWidget, ChatDisplayBrowser, PlainTextPasteEdit
from ui.components import CHIP_STYLE, BroadcastLabel, StatsBar, TimedStatusLabel
from ui.documents import DocumentsController
from ui.models import ModelController
from ui.onboarding import (
    DocumentsSetupScreen,
    ModelDownloadScreen,
    ProfileSetupScreen,
    WelcomeScreen,
)
from ui.profile import ProfileController
from ui.screens import HomeScreen
from ui.security import UnlockScreen
from ui.styles import STYLESHEET


def _is_worker_running(worker) -> bool:
    if worker is None:
        return False
    try:
        return bool(worker.isRunning())
    except RuntimeError:
        return False


def _request_worker_stop(worker) -> None:
    stop = getattr(worker, "stop", None)
    if callable(stop):
        stop()


def _wait_worker_stopped(label: str, worker, timeout_ms: int = 3000) -> bool:
    if not _is_worker_running(worker):
        return True
    if worker.wait(timeout_ms):
        return True
    log("UI", f"Worker still running during shutdown: {label}")
    return False


class MainWindow(QMainWindow):
    # Per-worker wait timeout during close. Kept tight: most workers
    # respond to stop_event in <1s; long waits make the app look frozen
    # to macOS Window Server, which then prompts force-quit — and an
    # abrupt force-quit while llama-server is mid-Metal-shutdown has
    # been observed to wedge the GPU subsystem. The retry loop below
    # picks up any worker that didn't make this window, so a tight cap
    # here doesn't lose work — it just unfreezes the UI sooner.
    SHUTDOWN_WAIT_MS = 500
    SHUTDOWN_RETRY_MS = 250

    def __init__(self, *, start_locked: bool = False):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(STYLESHEET)
        self._app_initialized = False

        if start_locked:
            self._show_unlock_screen()
        else:
            self._init_app_ui()

    def _show_unlock_screen(self) -> None:
        self._unlock_screen = UnlockScreen()
        self._unlock_screen.unlock_requested.connect(self._unlock_and_initialize)
        self.setCentralWidget(self._unlock_screen)
        QTimer.singleShot(0, self._unlock_screen.focus_password)
        log("APP", "Unlock screen shown")

    def _unlock_and_initialize(self, password: str) -> None:
        try:
            unlock(password)
            from config import migrate_profile_to_protected_file

            migrate_profile_to_protected_file()
            migrate_known_sensitive_files()
            log("APP", "Protected data unlocked")
            self._init_app_ui()
        except SecurityError:
            self._unlock_screen.show_error("Password is incorrect.")

    def _init_app_ui(self) -> None:
        if self._app_initialized:
            return
        self._app_initialized = True

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
        self._generation_token = 0
        self._compression_token = 0
        self._parsing_active = False
        self._download_worker = None
        self._load_token = 0
        self._hovered_link_range = None
        self._shutdown_retry_scheduled = False

        # Hidden stub kept for compat with ModelController which toggles it.
        self._model_btn = QPushButton()
        self._model_btn.setVisible(False)

        self._chat_controller = ChatController(self)
        self._documents_controller = DocumentsController(self)
        self._model_controller = ModelController(self)
        self._profile_controller = ProfileController(self)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._chat_widget = self._build_chat_screen()
        self._welcome_screen = WelcomeScreen()
        self._profile_setup_screen = ProfileSetupScreen()
        self._model_download_screen = ModelDownloadScreen()
        self._documents_setup_screen = DocumentsSetupScreen()
        self._home_screen = HomeScreen(
            chat_widget=self._chat_widget,
            build_profile_context=self._build_profile_context,
        )

        for screen in (
            self._welcome_screen,
            self._profile_setup_screen,
            self._model_download_screen,
            self._documents_setup_screen,
            self._home_screen,
        ):
            self._stack.addWidget(screen)

        self._wire_screens()
        self._init_chat()

        # DEV: force onboarding on every launch.
        if False and is_onboarding_complete():
            self._stack.setCurrentWidget(self._home_screen)
            self._try_load_saved_model()
        else:
            self._stack.setCurrentWidget(self._welcome_screen)

    # ── Screen wiring ──────────────────────────────────────────────────────

    def _wire_screens(self):
        self._welcome_screen.start_clicked.connect(self._show_profile_setup)
        self._profile_setup_screen.continue_clicked.connect(self._show_model_download)
        self._model_download_screen.proceed_clicked.connect(self._on_model_download_proceed)
        self._model_download_screen.back_clicked.connect(self._show_profile_setup)
        self._documents_setup_screen.proceed_clicked.connect(self._on_documents_setup_proceed)
        self._documents_setup_screen.back_clicked.connect(self._show_model_download)
        self._documents_setup_screen.docs.model_swapped.connect(self._on_docs_model_swapped)
        self._documents_setup_screen.docs.parsing_active_changed.connect(
            self._on_parsing_active_changed
        )
        self._documents_setup_screen.docs.docs_changed.connect(self._update_use_docs_state)

        # Home-embedded sections.
        self._home_screen.documents.model_swapped.connect(self._on_docs_model_swapped)
        self._home_screen.documents.parsing_active_changed.connect(self._on_parsing_active_changed)
        self._home_screen.documents.docs_changed.connect(self._update_use_docs_state)
        self._home_screen.model_hub.load_requested.connect(self._on_hub_load_requested)
        self._home_screen.model_hub.loaded_model_deleted.connect(self._ensure_default_model)
        # Source-of-truth for "is the LLM busy" lives here — every widget
        # that triggers a long LLM call (or wipes shared state) consults
        # this before proceeding, so the user gets an immediate "wait or
        # cancel" message instead of a queued request that looks frozen.
        self._home_screen._settings_form.busy_check = self.is_llm_busy
        self._home_screen.trends.busy_check = self.is_llm_busy
        self._home_screen.health_report.busy_check = self.is_llm_busy
        self._home_screen.documents.busy_check = self.is_llm_busy
        self._documents_setup_screen.docs.busy_check = self.is_llm_busy
        self._home_screen.model_hub.busy_check = self.is_llm_busy
        self.register_stats_bar(self._home_screen.health_report.stats_bar)
        self.register_stats_bar(self._home_screen.trends.stats_bar)
        self.register_general_status_label(self._home_screen.health_report._status_label)
        self.register_general_status_label(self._home_screen.trends._status)
        self._home_screen._settings_form.user_data_cleared.connect(self._on_user_data_cleared)
        self._home_screen._settings_form.default_model_changed.connect(
            self._on_default_model_changed
        )

    def _show_welcome(self):
        self._stack.setCurrentWidget(self._welcome_screen)

    def _show_profile_setup(self):
        self._profile_setup_screen.reload()
        self._stack.setCurrentWidget(self._profile_setup_screen)

    def _show_model_download(self):
        self._stack.setCurrentWidget(self._model_download_screen)
        self._model_download_screen.start()

    def _on_model_download_proceed(self, model_path: str):
        save_model_path(model_path)
        self._load_model(model_path)
        self._stack.setCurrentWidget(self._documents_setup_screen)

    def _on_documents_setup_proceed(self):
        if self._documents_setup_screen.docs.is_busy():
            return
        # Persist completion before showing Home so a crash during
        # _show_home() doesn't drop the user back into onboarding on
        # next launch despite having finished the flow.
        self._mark_onboarding_complete()
        self._show_home()

    def _mark_onboarding_complete(self) -> None:
        try:
            set_onboarding_complete(True)
        except Exception as e:
            # Persistence failed (disk full, settings.json read-only,
            # etc.) — warn the user but do NOT trap them on the docs
            # screen. The in-memory state is fine for this session;
            # they'll just see the welcome flow again next launch.
            from PySide6.QtWidgets import QMessageBox

            log("APP", f"Failed to mark onboarding complete: {e}")
            QMessageBox.warning(
                self,
                "Could not save onboarding state",
                "The app couldn't save its progress. You may see the "
                "welcome screen again on next launch.\n\n"
                f"Reason: {e}",
            )

    def _show_home(self):
        # Resetting Home to its default tile (Profile) keeps the Back button
        # in the chat top bar meaningful: it returns the user to the
        # default Home view.
        self._home_screen.show_profile()
        self._stack.setCurrentWidget(self._home_screen)

    def _on_hub_load_requested(self, model_path: str):
        save_model_path(model_path)
        self._load_model(model_path)
        self._home_screen.show_chat()

    def _on_default_model_changed(self, model_id: str) -> None:
        """Settings → Default Model was changed and saved. Switch the
        running llama-server to the newly-chosen default."""
        from PySide6.QtWidgets import QMessageBox

        from config import APPROVED_MODELS, approved_model_file_path

        model = APPROVED_MODELS.get(model_id)
        if not model:
            log("UI", f"Unknown default model id: {model_id}")
            return
        target_path = approved_model_file_path(model)
        if not target_path.exists():
            # Modal so the user sees this even while on the Settings tile;
            # writing to the chat top-bar status label would be invisible
            # from here.
            log("UI", f"Default model {model_id} not yet downloaded at {target_path}")
            QMessageBox.warning(
                self,
                "Model not installed",
                f"{model['display_name']} isn't downloaded yet. "
                "Open the Models tab and install it to use it as the default.",
            )
            return
        save_model_path(str(target_path))
        # Refresh the My Models table so per-row delete buttons reflect
        # the new default. `_load_model` would normally trigger this on
        # successful start, but it short-circuits when the new default
        # is already the loaded model — leaving the table stale.
        self._home_screen.model_hub.refresh_local()
        self._load_model(str(target_path))

    def _on_user_data_cleared(self) -> None:
        self._current_chat = new_chat()
        self._history = []
        self._chat_display.clear()
        self._chat_search.clear()
        self._refresh_chat_list()
        self._home_screen.documents._refresh_list()
        self._home_screen.health_report.refresh()
        self._home_screen.trends.refresh()
        self._update_use_docs_state()

    # ── Chat screen build ──────────────────────────────────────────────────

    def _build_chat_screen(self) -> QWidget:
        central = QWidget()

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

        self._chat_search = QLineEdit()
        self._chat_search.setPlaceholderText("Search chats...")
        self._chat_search.setClearButtonEnabled(True)
        self._chat_search.setFixedHeight(38)
        self._chat_search.textChanged.connect(self._filter_chat_list)
        sidebar_layout.addWidget(self._chat_search)

        self._chat_list = AutoHideScrollListWidget()
        self._chat_list.setObjectName("chatList")
        self._chat_list.itemClicked.connect(self._on_chat_selected)
        self._chat_list.itemDoubleClicked.connect(self._on_chat_rename)
        self._chat_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._chat_list.customContextMenuRequested.connect(self._on_chat_context_menu)
        sidebar_layout.addWidget(self._chat_list, stretch=1)

        new_chat_btn = QPushButton("New Chat")
        new_chat_btn.setObjectName("attachButton")
        new_chat_btn.setFixedHeight(38)
        new_chat_btn.clicked.connect(self._new_chat)
        sidebar_layout.addWidget(new_chat_btn)

        outer.addWidget(self._sidebar)

        # --- Main content ---
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Top bar — hamburger + Home + status
        top_row = QHBoxLayout()

        self._sidebar_btn = QPushButton("☰")
        self._sidebar_btn.setObjectName("attachButton")
        self._sidebar_btn.setFixedSize(38, 38)
        self._sidebar_btn.clicked.connect(self._toggle_sidebar)
        top_row.addWidget(self._sidebar_btn)

        chip_style = CHIP_STYLE

        # `_status_label` is a BroadcastLabel so peer StatsBars in
        # Health Report / Trends mirror "Model: X" / "Loading..." text
        # without changing the many controller call sites that already
        # write to `_status_label`.
        self._status_label = BroadcastLabel("No model loaded")
        self._status_label.setStyleSheet(chip_style)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_row.addWidget(self._status_label)

        self._mem_chip = QLabel("Memory: ...")
        self._mem_chip.setStyleSheet(chip_style)
        self._mem_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_row.addWidget(self._mem_chip)

        self._cpu_chip = QLabel("CPU: ...")
        self._cpu_chip.setStyleSheet(chip_style)
        self._cpu_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_row.addWidget(self._cpu_chip)

        # Context window utilization (history tokens / configured n_ctx).
        # Lives in the slack to the right of the resource chips so it
        # tells the user how much room they have left to keep chatting.
        self._ctx_chip = QLabel("Context: --")
        self._ctx_chip.setStyleSheet(chip_style)
        self._ctx_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ctx_chip.setToolTip(
            "Approximate context-window usage: history tokens / configured "
            "context size. Increase Context Window in Settings if you're "
            "running out."
        )
        top_row.addWidget(self._ctx_chip)

        # Peer StatsBars (Health Report, Trends) register themselves
        # here so the 2-second stats tick and context refresh push the
        # same values out to every section's chips.
        self._stats_bars: list[StatsBar] = []

        # Peer section-level status labels (Health Report, Trends).
        # `_set_general_status` fans transient messages out to every
        # registered label; each one auto-clears 30 s after its last
        # update via TimedStatusLabel.
        self._general_status_labels: list[TimedStatusLabel] = []

        top_row.addStretch()

        main_layout.addLayout(top_row)

        # Stats timer set up here (after chips exist) so the first
        # _update_stats() call doesn't reference a not-yet-created chip.
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start(2000)
        self._update_ctx_chip()
        self._update_stats()

        # Transient status banner under the chips. All non-model-name
        # messages (model loading, download progress, "wait or cancel",
        # errors) flow through `_set_general_status` and land here as
        # well as in the peer labels in Health Report / Trends.
        self._general_status_label = TimedStatusLabel("")
        main_layout.addWidget(self._general_status_label)
        self._general_status_labels.append(self._general_status_label)

        # Chat display
        self._chat_display = ChatDisplayBrowser()
        self._chat_display.setReadOnly(True)
        self._chat_display.setOpenLinks(False)
        self._chat_display.setOpenExternalLinks(False)
        self._chat_display.anchorClicked.connect(self._on_link_clicked)
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

        self._stop_btn = QPushButton("■")
        self._stop_btn.setObjectName("stopButton")
        self._stop_btn.setFixedSize(BTN_H * 2 + 4, BTN_H)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_generation)
        btn_column.addWidget(self._stop_btn)

        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedSize(BTN_H * 2 + 4, BTN_H)
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._send_message)
        btn_column.addWidget(self._send_btn)

        input_height = BTN_H * 2 + BTN_SPACING
        self._input_field = PlainTextPasteEdit()
        self._input_field.setPlaceholderText("Type your message...")
        self._input_field.setFixedHeight(input_height)
        self._input_field.setViewportMargins(0, 0, 104, 0)

        self._use_docs_btn = QPushButton("Use Docs", self._input_field)
        self._use_docs_btn.setObjectName("useDocsButton")
        self._use_docs_btn.setCheckable(True)
        self._use_docs_btn.setChecked(True)
        self._use_docs_btn.setFixedSize(92, 28)
        self._use_docs_btn.setToolTip("Use uploaded documents when answering")
        self._use_docs_btn.setStyleSheet(
            """
            QPushButton#useDocsButton {
                background-color: #313244;
                color: #6c7086;
                border: 1px solid #45475a;
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton#useDocsButton:enabled:hover {
                background-color: #45475a;
            }
            QPushButton#useDocsButton:enabled:checked {
                color: #cdd6f4;
                border-color: #89b4fa;
            }
            """
        )
        self._update_use_docs_state()

        input_row.addWidget(btn_widget)
        self._input_field.installEventFilter(self)
        self._chat_display.viewport().setMouseTracking(True)
        self._chat_display.viewport().installEventFilter(self)
        input_row.addWidget(self._input_field, stretch=1)
        QTimer.singleShot(0, self._position_use_docs_button)

        main_layout.addLayout(input_row)
        outer.addWidget(main_widget, stretch=1)

        return central

    # ── Chat management ────────────────────────────────────────────────────

    def _init_chat(self):
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

    def _filter_chat_list(self, query: str):
        q = query.strip().lower()
        for i in range(self._chat_list.count()):
            item = self._chat_list.item(i)
            title = item.data(Qt.ItemDataRole.UserRole + 1) or ""
            item.setHidden(bool(q) and q not in str(title).lower())

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
        if obj == self._input_field and event.type() == QEvent.Type.Resize:
            self._position_use_docs_button()

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

    def _position_use_docs_button(self) -> None:
        if not hasattr(self, "_use_docs_btn") or not hasattr(self, "_input_field"):
            return
        margin = 8
        x = self._input_field.width() - self._use_docs_btn.width() - margin
        y = self._input_field.height() - self._use_docs_btn.height() - margin
        self._use_docs_btn.move(max(margin, x), max(margin, y))

    def _update_use_docs_state(self) -> None:
        if not hasattr(self, "_use_docs_btn"):
            return
        has_docs = bool(list_uploaded_doc_paths())
        self._use_docs_btn.setEnabled(has_docs)
        if not has_docs:
            self._use_docs_btn.setChecked(False)
            self._use_docs_btn.setToolTip("Upload documents to enable document search")
        else:
            self._use_docs_btn.setToolTip("Use uploaded documents when answering")

    def closeEvent(self, event):
        log("UI", "Application closing")
        if not self._app_initialized:
            super().closeEvent(event)
            return
        # Stop llama-server FIRST. With a multi-billion-parameter model
        # loaded into unified memory + an active Metal context, this is
        # the single largest resource the OS has to reclaim. Releasing
        # it before the per-worker wait loop frees the GPU/RAM in
        # bounded time (≤2 s), which prevents the laptop-freeze case we
        # saw: workers blocking the main thread → macOS prompts
        # force-quit → user kills the app mid-Metal-shutdown → GPU
        # subsystem wedges.
        stop_server()
        self._save_current_chat()
        self._cancel_transient_ui_work()
        workers = self._background_workers()
        for _, worker in workers:
            _request_worker_stop(worker)
        still_running = []
        for label, worker in workers:
            # EmbedderPrefetchWorker can't be cancelled (snapshot_download
            # has no hook). Don't block shutdown waiting for it — Qt's
            # process exit will reap the daemon thread when the OS sends
            # SIGTERM. Same for any future worker that ignores stop_event.
            if "embedder" in label.lower():
                log("UI", f"Skipping wait on uncancellable worker: {label}")
                continue
            if not _wait_worker_stopped(label, worker, timeout_ms=self.SHUTDOWN_WAIT_MS):
                still_running.append(label)
        if still_running:
            event.ignore()
            self.setEnabled(False)
            labels = ", ".join(still_running)
            log("UI", f"Shutdown deferred; workers still running: {labels}")
            self._schedule_shutdown_retry()
            return
        super().closeEvent(event)

    def _schedule_shutdown_retry(self) -> None:
        if self._shutdown_retry_scheduled:
            return
        self._shutdown_retry_scheduled = True
        QTimer.singleShot(self.SHUTDOWN_RETRY_MS, self._retry_close_after_workers)

    def _retry_close_after_workers(self) -> None:
        self._shutdown_retry_scheduled = False
        self.close()

    def _cancel_transient_ui_work(self) -> None:
        cancel_trends_render = getattr(
            self._home_screen.trends,
            "_cancel_pending_render",
            None,
        )
        if callable(cancel_trends_render):
            cancel_trends_render()

    def _background_workers(self) -> list[tuple[str, object]]:
        workers = [
            ("chat generation", self._worker),
            ("chat compression", self._compression_worker),
            ("model download", self._download_worker),
            ("model server start", self._server_worker),
        ]

        for label, docs in (
            ("onboarding documents", self._documents_setup_screen.docs),
            ("home documents", self._home_screen.documents),
        ):
            workers.extend(
                [
                    (f"{label} vision model", docs._ensure_worker),
                    (f"{label} indexing", docs._index_worker),
                ]
            )

        workers.extend(
            [
                (
                    "onboarding main model download",
                    self._model_download_screen._model_worker,
                ),
                (
                    "onboarding mmproj download",
                    self._model_download_screen._mmproj_worker,
                ),
                (
                    "onboarding embedder prefetch",
                    self._model_download_screen._embedder_worker,
                ),
                ("health report", self._home_screen.health_report._worker),
                ("trends extraction", self._home_screen.trends._worker),
            ]
        )
        # Model Hub now supports multiple concurrent downloads, keyed by
        # token. Wait for all of them.
        for token, worker in self._home_screen.model_hub._download_workers.items():
            workers.append((f"model hub download {token}", worker))
        return [(label, worker) for label, worker in workers if worker is not None]

    # Labels in _background_workers() that talk to llama-server (i.e.
    # whose work breaks if the server is stopped or swapped mid-flight).
    # File downloads, embedder prefetch, and server start/stop itself
    # are intentionally excluded.
    _LLM_WORKER_LABEL_HINTS = (
        "chat generation",
        "chat compression",
        "health report",
        "trends extraction",
        "indexing",
        "vision model",
    )

    def is_llm_busy(self) -> bool:
        """True if any worker actively driving llama-server is running.

        Used to refuse actions that would pull the server out from under
        an in-flight request: model swap, Clear User Data wiping
        DOCS_DIR / FILTERING_OUTPUT_DIR while IndexWorker reads them, etc.
        """
        for label, worker in self._background_workers():
            if not any(hint in label for hint in self._LLM_WORKER_LABEL_HINTS):
                continue
            try:
                if worker.isRunning():
                    return True
            except RuntimeError:
                # QThread was already destroyed — treat as not running.
                pass
        return False

    # ── System stats ────────────────────────────────────────────────────────

    def register_general_status_label(self, label: TimedStatusLabel) -> None:
        """Subscribe a section's status label to general broadcasts."""
        self._general_status_labels.append(label)

    def _set_general_status(self, text: str) -> None:
        """Fan a transient status message out to every section's status label.

        Used for messages that aren't section-specific — model loading,
        download progress, busy refusals ("Wait for the current operation
        to finish..."), server errors. Each label auto-clears 30 s after
        the last update.
        """
        for label in self._general_status_labels:
            label.setText(text)

    def register_stats_bar(self, bar: StatsBar) -> None:
        """Subscribe a peer StatsBar to status / mem / cpu / context updates.

        Status mirroring is wired through BroadcastLabel.add_listener so
        anything that calls `_status_label.setText(...)` (controllers in
        chat / models) reaches the bar with no other code changes.
        """
        self._stats_bars.append(bar)
        self._status_label.add_listener(bar.set_status)
        bar.set_memory(self._mem_chip.text())
        bar.set_cpu(self._cpu_chip.text())
        bar.set_context(self._ctx_chip.text())

    def _update_stats(self):
        import psutil

        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        mem_text = f"Memory: {used_gb:.1f}/{total_gb:.1f} GB"
        cpu_text = f"CPU: {psutil.cpu_percent(interval=None):.0f}%"
        self._mem_chip.setText(mem_text)
        self._cpu_chip.setText(cpu_text)
        for bar in self._stats_bars:
            bar.set_memory(mem_text)
            bar.set_cpu(cpu_text)

    @staticmethod
    def _format_token_count(n: int) -> str:
        if n >= 1000:
            return f"{n / 1000:.1f}K"
        return str(n)

    def _update_ctx_chip(self) -> None:
        """Refresh the context-usage chip in the chat top bar.

        Used = approximate token count of the current chat history (chars / 3
        plus a small system-prompt overhead). This is an estimate, not a
        server tokenize call — fine for a status indicator and avoids
        per-keystroke HTTP roundtrips.

        Total = the per-model `ctx_size` for whatever model is *actually*
        running (not the saved `model_path`, which can lag behind the live
        server after a model swap). Falls back to that model's max ctx if
        no per-model value is saved.
        """
        if not hasattr(self, "_ctx_chip"):
            return
        from config import load_ctx_size, load_model_meta, load_model_path
        from core.llm_engine import get_current_model_path

        # Source of truth is the live llama-server's model. If no server
        # is running, fall back to the saved path (best-effort).
        path = get_current_model_path() or load_model_path()
        meta = load_model_meta(path) if path else None
        model_max = (meta or {}).get("context_length")

        n_ctx = load_ctx_size(path) or model_max or 8192

        history_chars = sum(len(m.get("content", "")) for m in self._history)
        # ~3 chars per token (slightly pessimistic for English clinical text)
        # plus ~500 tokens for the system prompt + per-turn scaffolding.
        used = history_chars // 3 + 500
        used = min(used, n_ctx)

        ctx_text = (
            f"Context: {self._format_token_count(used)}/"
            f"{self._format_token_count(n_ctx)}"
        )
        self._ctx_chip.setText(ctx_text)
        for bar in self._stats_bars:
            bar.set_context(ctx_text)

    # ── Profile / app actions ──────────────────────────────────────────────

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
        self._chat_controller.on_thinking_token(None, token)

    def _on_response_token(self, token: str):
        self._chat_controller.on_response_token(None, token)

    def _on_generation_done(self):
        self._chat_controller.on_generation_done()

    def _on_generation_error(self, error: str):
        self._chat_controller.on_generation_error(None, error)

    def _on_compression_done(self, summary: str):
        self._chat_controller.on_compression_done(None, summary)

    def _on_compression_error(self, error: str):
        self._chat_controller.on_compression_error(None, error)
