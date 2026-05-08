from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from ui.components import header_label

_TITLE_TEXT = "Enter your app password"
_FIELD_HEIGHT = 38


class UnlockScreen(QWidget):
    unlock_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        layout.addStretch()
        self._title = header_label(_TITLE_TEXT)
        layout.addWidget(self._title)

        description = QLabel("Your protected local data will be unlocked on this device.")
        description.setObjectName("statusLabel")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)

        field_row = QHBoxLayout()
        field_row.addStretch()
        self._password = QLineEdit()
        self._password.setPlaceholderText("Password")
        self._password.setEchoMode(QLineEdit.EchoMode.Password)
        field_width = self._title.fontMetrics().horizontalAdvance(_TITLE_TEXT)
        self._password.setFixedSize(field_width, _FIELD_HEIGHT)
        self._password.returnPressed.connect(self._submit)
        field_row.addWidget(self._password)
        field_row.addStretch()
        layout.addLayout(field_row)

        self._error = QLabel("")
        self._error.setObjectName("statusLabel")
        self._error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._error)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self._continue_btn = QPushButton("Continue")
        self._continue_btn.setFixedSize(100, 38)
        self._continue_btn.clicked.connect(self._submit)
        button_row.addWidget(self._continue_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        layout.addStretch()

    def focus_password(self) -> None:
        self._password.setFocus()

    def show_error(self, message: str) -> None:
        self._error.setText(message)
        self._password.selectAll()
        self._password.setFocus()

    def _submit(self) -> None:
        password = self._password.text()
        if not password:
            self.show_error("Password cannot be empty.")
            return
        self.unlock_requested.emit(password)
