"""Settings — global app preferences."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config import (
    APPROVED_MODELS,
    DEFAULT_MODEL_CHOICE_IDS,
    load_answer_detail,
    load_ctx_size,
    load_default_model_id,
    load_model_meta,
    load_model_path,
    save_answer_detail,
    save_ctx_size,
    save_default_model_id,
)
from core.llm_engine import get_current_model_path
from ui.sections import SectionNames
from core.security import (
    SecurityError,
    change_password,
    disable_password,
    is_security_configured,
    migrate_known_sensitive_files,
    setup_password,
)
from core.user_data import clear_user_data

_CTX_OPTIONS = [2048, 4096, 8192, 16384]
_ANSWER_DETAIL_OPTIONS = [
    ("Short", "short"),
    ("Balanced", "balanced"),
    ("Detailed", "detailed"),
]
_FIELD_WIDTH = 220
_BUTTON_FIELD_WIDTH = _FIELD_WIDTH
_FIELD_HEIGHT = 24


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
    return lbl


def _block_header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("settingsBlockHeader")
    lbl.setStyleSheet("font-weight: bold; font-size: 16px; color: #89b4fa;")
    return lbl


def _description(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("statusLabel")
    lbl.setWordWrap(True)
    return lbl


def _setting_block(
    layout: QVBoxLayout,
    *,
    subheader: str,
    control: QWidget,
    description: str,
) -> None:
    layout.addWidget(_section_label(subheader))
    layout.addWidget(control)
    layout.addWidget(_description(description))


def _password_field(placeholder: str) -> QLineEdit:
    field = QLineEdit()
    field.setPlaceholderText(placeholder)
    field.setEchoMode(QLineEdit.EchoMode.Password)
    field.setFixedSize(_BUTTON_FIELD_WIDTH, _FIELD_HEIGHT)
    field.setStyleSheet("padding: 0px; margin: 0px; font-size: 10px;")
    return field


class SettingsForm(QWidget):
    """Settings inputs without any host chrome."""

    submitted = Signal()
    reindex_requested = Signal()
    user_data_cleared = Signal()
    # Emitted on Save when the user changed the Default Model dropdown to
    # a different value than what was loaded into the form. Carries the
    # newly-selected model id (e.g. "qwen35_9b_vision"). MainWindow wires
    # this to its model loader so the chosen default starts running.
    default_model_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(_block_header(SectionNames.MODELS))

        # The dropdown options depend on the currently-loaded model. We
        # build the empty combo here and populate it via reload(), which
        # the host (HomeScreen) calls again whenever the Settings tile is
        # re-activated — so a model swap done after Settings was first
        # constructed actually shows up in the dropdown.
        self._ctx_combo = QComboBox()
        self._ctx_combo.setFixedSize(_FIELD_WIDTH, _FIELD_HEIGHT)
        self._ctx_model_path: str | None = None
        self._populate_ctx_combo()
        _setting_block(
            layout,
            subheader="Context Window",
            control=self._ctx_combo,
            description=(
                "Controls how much text the model can read at once. "
                "Higher values use more memory and apply after model reload."
            ),
        )

        self._default_model_combo = QComboBox()
        self._default_model_combo.setFixedSize(_FIELD_WIDTH, _FIELD_HEIGHT)
        for model_id in DEFAULT_MODEL_CHOICE_IDS:
            self._default_model_combo.addItem(
                APPROVED_MODELS[model_id]["display_name"], model_id
            )
        # Track the value that was active when the form was opened so save()
        # can detect a real change and emit `default_model_changed`.
        self._initial_default_model_id = load_default_model_id()
        idx = self._default_model_combo.findData(self._initial_default_model_id)
        self._default_model_combo.setCurrentIndex(max(idx, 0))
        _setting_block(
            layout,
            subheader="Default Model",
            control=self._default_model_combo,
            description=(
                "Choose default model used for document parsing, health report, "
                "trends and other system functionality."
            ),
        )

        layout.addSpacing(8)
        layout.addWidget(_block_header(SectionNames.CHAT_WITH_DOCUMENTS))

        self._answer_detail_combo = QComboBox()
        self._answer_detail_combo.setFixedSize(_FIELD_WIDTH, _FIELD_HEIGHT)
        for label, value in _ANSWER_DETAIL_OPTIONS:
            self._answer_detail_combo.addItem(label, value)
        idx = self._answer_detail_combo.findData(load_answer_detail())
        self._answer_detail_combo.setCurrentIndex(max(idx, 0))
        _setting_block(
            layout,
            subheader="Answer Detail",
            control=self._answer_detail_combo,
            description=(
                "Controls only chat response length and detail. Does not affect "
                "reports, trends, parsing, or indexing."
            ),
        )

        layout.addSpacing(8)
        layout.addWidget(_block_header(SectionNames.MEDICAL_DOCUMENTS))

        self._reindex_btn = QPushButton("Reindex")
        self._reindex_btn.setObjectName("attachButton")
        self._reindex_btn.setFixedSize(_BUTTON_FIELD_WIDTH, _FIELD_HEIGHT)
        self._reindex_btn.setStyleSheet("padding: 0px; margin: 0px; font-size: 11px;")
        self._reindex_btn.clicked.connect(self.reindex_requested.emit)
        _setting_block(
            layout,
            subheader="Index",
            control=self._reindex_btn,
            description="Rebuild document search from existing processed documents.",
        )

        layout.addSpacing(8)
        layout.addWidget(_block_header("Security"))

        layout.addWidget(_section_label("Password Protection"))
        self._security_status = _description("")
        layout.addWidget(self._security_status)

        self._security_actions = QWidget()
        self._security_actions_layout = QHBoxLayout(self._security_actions)
        self._security_actions_layout.setContentsMargins(0, 0, 0, 0)
        self._security_actions_layout.setSpacing(8)
        layout.addWidget(self._security_actions)

        self._enable_btn = self._security_button("Enable")
        self._enable_btn.clicked.connect(lambda: self._show_security_form("enable"))
        self._change_btn = self._security_button("Change Password")
        self._change_btn.clicked.connect(lambda: self._show_security_form("change"))
        self._disable_btn = self._security_button("Turn Off")
        self._disable_btn.clicked.connect(lambda: self._show_security_form("disable"))

        for btn in (self._enable_btn, self._change_btn, self._disable_btn):
            self._security_actions_layout.addWidget(btn)
        self._security_actions_layout.addStretch()

        self._security_form = QWidget()
        form_layout = QVBoxLayout(self._security_form)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(8)
        self._current_password = _password_field("Current password")
        self._new_password = _password_field("Password")
        self._confirm_password = _password_field("Confirm password")
        form_layout.addWidget(self._current_password)
        form_layout.addWidget(self._new_password)
        form_layout.addWidget(self._confirm_password)

        form_buttons = QHBoxLayout()
        form_buttons.setContentsMargins(0, 0, 0, 0)
        form_buttons.setSpacing(8)
        self._security_save_btn = self._security_button("Save")
        self._security_save_btn.clicked.connect(self._submit_security_form)
        form_buttons.addWidget(self._security_save_btn)
        form_buttons.addStretch()
        form_layout.addLayout(form_buttons)
        self._security_message = _description("")
        form_layout.addWidget(self._security_message)
        layout.addWidget(self._security_form)
        self._security_mode: str | None = None
        self._refresh_security_state()

        # Two-step clear: clicking "Clear Everything" toggles a same-style
        # "Yes, I am sure" button below it. Only the second button performs
        # the actual clear; clicking "Clear Everything" again hides it
        # (so it acts as a cancel for the confirmation).
        layout.addWidget(_section_label("Clear User Data"))

        self._clear_data_btn = QPushButton("Clear Everything")
        self._clear_data_btn.setObjectName("stopButton")
        self._clear_data_btn.setFixedSize(_BUTTON_FIELD_WIDTH, _FIELD_HEIGHT)
        self._clear_data_btn.setStyleSheet("padding: 0px; margin: 0px; font-size: 11px;")
        self._clear_data_btn.clicked.connect(self._toggle_clear_confirm)
        layout.addWidget(self._clear_data_btn)

        self._clear_data_confirm_btn = QPushButton("Yes, I am sure")
        self._clear_data_confirm_btn.setObjectName("stopButton")
        self._clear_data_confirm_btn.setFixedSize(_BUTTON_FIELD_WIDTH, _FIELD_HEIGHT)
        self._clear_data_confirm_btn.setStyleSheet(
            "padding: 0px; margin: 0px; font-size: 11px;"
        )
        self._clear_data_confirm_btn.setVisible(False)
        self._clear_data_confirm_btn.clicked.connect(self._clear_user_data_confirmed)
        layout.addWidget(self._clear_data_confirm_btn)

        layout.addWidget(
            _description("Deletes local user data. Downloaded models are kept.")
        )
        self._clear_data_message = _description("")
        layout.addWidget(self._clear_data_message)

    def reload(self) -> None:
        """Re-query the live model and rebuild model-dependent fields.

        Called by the host whenever the Settings tile is opened, so a
        model swap that happened after the form was first built (the
        dropdown is built once in __init__) is reflected immediately.
        """
        self._populate_ctx_combo()

    def _populate_ctx_combo(self) -> None:
        """Rebuild the Context Window dropdown for the currently-loaded model.

        Source of truth is `get_current_model_path()` (the live server's
        model). Falls back to the saved `model_path` only when no server
        is running.
        """
        model_path = get_current_model_path() or load_model_path()
        meta = load_model_meta(model_path) if model_path else None
        model_max = meta.get("context_length") if meta else None
        model_name = meta.get("name") if meta else None
        # Stash for save() — keeps the dropdown's per-model save target
        # aligned with whichever model the dropdown was just rebuilt for.
        self._ctx_model_path = model_path

        standard = (
            [v for v in _CTX_OPTIONS if v < model_max]
            if model_max
            else list(_CTX_OPTIONS)
        )

        # Per-model default: if the user has saved a value for the
        # currently-loaded model, use it; otherwise pick that model's own
        # max so the dropdown reflects the model's full capability on
        # first open.
        current = load_ctx_size(model_path) or model_max or 8192

        # Block signals so rebuilding the items doesn't fire a spurious
        # change while we repopulate.
        self._ctx_combo.blockSignals(True)
        self._ctx_combo.clear()
        for v in standard:
            self._ctx_combo.addItem(f"{v:,}", v)
        if model_max:
            max_label = f"{model_name or 'Current model'}'s max"
            self._ctx_combo.addItem(f"{model_max:,} ({max_label})", model_max)
        all_options = standard + ([model_max] if model_max else [])
        idx = next(
            (i for i, v in enumerate(all_options) if v == current),
            max(0, len(all_options) - 1),
        )
        self._ctx_combo.setCurrentIndex(idx)
        self._ctx_combo.blockSignals(False)

    def save(self):
        save_ctx_size(self._ctx_combo.currentData(), self._ctx_model_path)

        new_default = self._default_model_combo.currentData()
        save_default_model_id(new_default)
        if new_default and new_default != self._initial_default_model_id:
            self._initial_default_model_id = new_default
            self.default_model_changed.emit(new_default)

        save_answer_detail(self._answer_detail_combo.currentData())

    def _refresh_security_state(self) -> None:
        enabled = is_security_configured()
        self._security_status.setText(
            "Status: Enabled. Sensitive app files are encrypted locally."
            if enabled
            else "Status: Disabled. Sensitive app files are stored normally."
        )
        self._enable_btn.setVisible(not enabled)
        self._change_btn.setVisible(enabled)
        self._disable_btn.setVisible(enabled)
        if self._security_mode is None:
            self._security_form.setVisible(False)

    def _security_button(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("attachButton")
        btn.setFixedSize(_BUTTON_FIELD_WIDTH, _FIELD_HEIGHT)
        btn.setStyleSheet("padding: 0px; margin: 0px; font-size: 11px;")
        return btn

    def _show_security_form(self, mode: str) -> None:
        if self._security_mode == mode:
            self._hide_security_form()
            return

        self._security_mode = mode
        self._security_message.setText("")
        self._enable_btn.setText("Disable" if mode == "enable" else "Enable")
        for field in (self._current_password, self._new_password, self._confirm_password):
            field.clear()

        self._current_password.setVisible(mode in {"change", "disable"})
        self._new_password.setVisible(mode in {"enable", "change"})
        self._confirm_password.setVisible(mode in {"enable", "change"})
        self._new_password.setPlaceholderText(
            "New password" if mode == "change" else "Password"
        )
        self._confirm_password.setPlaceholderText(
            "Confirm new password" if mode == "change" else "Confirm password"
        )
        self._security_save_btn.setText("Disable Protection" if mode == "disable" else "Save")
        self._security_form.setVisible(True)

    def _hide_security_form(self) -> None:
        self._security_mode = None
        self._security_message.setText("")
        self._enable_btn.setText("Enable")
        self._security_form.setVisible(False)

    def _submit_security_form(self) -> None:
        mode = self._security_mode
        if mode is None:
            return
        current = self._current_password.text()
        new = self._new_password.text()
        confirm = self._confirm_password.text()
        if mode in {"enable", "change"} and not new:
            self._security_message.setText("Password cannot be empty.")
            return
        if mode in {"enable", "change"} and new != confirm:
            self._security_message.setText("Passwords do not match.")
            return
        if mode in {"change", "disable"} and not current:
            self._security_message.setText("Current password is required.")
            return

        try:
            if mode == "enable":
                setup_password(new)
                from config import migrate_profile_to_protected_file

                migrate_profile_to_protected_file()
                migrate_known_sensitive_files()
                self._security_message.setText("Password protection enabled.")
            elif mode == "change":
                change_password(current, new)
                self._security_message.setText("Password changed.")
            elif mode == "disable":
                disable_password(current)
                self._security_message.setText("Password protection disabled.")
        except SecurityError as e:
            self._security_message.setText(str(e))
            return
        self._security_mode = None
        self._refresh_security_state()

    def _toggle_clear_confirm(self) -> None:
        """Show or hide the 'Yes, I am sure' confirm button.

        First click of "Clear Everything" reveals the confirm button;
        clicking "Clear Everything" again before confirming hides it
        (treated as a cancellation of the in-flight confirmation).
        """
        new_state = not self._clear_data_confirm_btn.isVisible()
        self._clear_data_confirm_btn.setVisible(new_state)
        # Wipe any prior status text — we're starting a new attempt or
        # cancelling the previous one.
        self._clear_data_message.setText("")

    def _clear_user_data_confirmed(self) -> None:
        """Actually delete user data. Fires only after the user has opted
        in twice (Clear Everything → Yes, I am sure)."""
        self._clear_data_confirm_btn.setVisible(False)
        try:
            clear_user_data()
            self._clear_data_message.setText("User data cleared. Downloaded models were kept.")
            self.user_data_cleared.emit()
        except Exception as e:
            self._clear_data_message.setText(f"Failed to clear user data: {e}")


__all__ = ["SettingsForm"]
