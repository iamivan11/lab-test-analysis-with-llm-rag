STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
}

QTextEdit, QLineEdit {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 8px;
    font-size: 14px;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}

QTextEdit:focus, QLineEdit:focus {
    border: 1px solid #89b4fa;
}

QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 14px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #b4d0fb;
}

QPushButton:pressed {
    background-color: #74c7ec;
}

QPushButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}

QPushButton#attachButton {
    background-color: #313244;
    color: #cdd6f4;
    padding: 8px 12px;
}

QPushButton#attachButton:hover {
    background-color: #45475a;
}

QPushButton#reasoningButton {
    background-color: #313244;
    color: #6c7086;
    padding: 8px 12px;
}

QPushButton#reasoningButton:hover {
    background-color: #45475a;
}

QPushButton#reasoningButton:checked {
    background-color: #313244;
    color: #a6e3a1;
    border: 1px solid #a6e3a1;
}

QLabel {
    color: #cdd6f4;
    font-size: 13px;
}

QLabel#statusLabel {
    color: #6c7086;
    font-size: 12px;
}

QLabel#fileLabel {
    color: #a6e3a1;
    font-size: 12px;
    padding: 4px 8px;
    background-color: #313244;
    border-radius: 4px;
}

QPushButton#stopButton {
    background-color: #313244;
    color: #45475a;
    padding: 8px;
}

QPushButton#stopButton:enabled {
    color: #f38ba8;
}

QPushButton#stopButton:enabled:hover {
    background-color: #45475a;
    border-color: #f38ba8;
}

QPushButton#fileChip {
    background-color: #313244;
    color: #a6e3a1;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: normal;
}

QPushButton#fileChip:hover {
    background-color: #45475a;
    border-color: #f38ba8;
    color: #f38ba8;
}

QPushButton#genderButton {
    background-color: #313244;
    color: #6c7086;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 14px;
    font-weight: normal;
}

QPushButton#genderButton:hover {
    background-color: #45475a;
}

QPushButton#genderButton:checked {
    color: #cdd6f4;
    border: 1px solid #89b4fa;
}

QDialog {
    background-color: #1e1e2e;
}

QWidget#sidebar {
    background-color: #181825;
    border-right: 1px solid #313244;
}

QListWidget {
    background-color: transparent;
    border: none;
    color: #cdd6f4;
    font-size: 13px;
    outline: none;
}

QListWidget::item {
    padding: 6px 8px;
    border-radius: 6px;
    color: #cdd6f4;
    margin: 1px 0px;
}

QListWidget::item:hover {
    background-color: #313244;
}

QListWidget::item:selected {
    background-color: #45475a;
    color: #cdd6f4;
}

QMenu {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 16px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #45475a;
}

QWidget#chatItemWidget {
    background: transparent;
}

QLabel#chatItemLabel {
    color: #cdd6f4;
    font-size: 13px;
    font-weight: normal;
    background: transparent;
}

QPushButton#chatMenuButton {
    background-color: transparent;
    color: #6c7086;
    border: none;
    border-radius: 4px;
    padding: 0px;
    font-size: 16px;
    font-weight: bold;
}

QPushButton#chatMenuButton:hover {
    color: #cdd6f4;
    background-color: rgba(69, 71, 90, 150);
}
"""
