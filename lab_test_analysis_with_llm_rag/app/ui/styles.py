# ruff: noqa: RUF001
STYLESHEET = """
/* ──────────────────────────────────────────────────────────────────────
   Button system
   ──────────────────────────────────────────────────────────────────────
   Shape × Variant.

   Shape (determines size):
     Named  — 38 tall, width ≥ 100 (or 120 for long labels)
     Icon   — 28 × 28 square

   Variant (determines color):
     Primary    — blue filled       (main / confirm)
     Secondary  — dark neutral      (dismiss / navigation / alternative)
     Icon Secondary is a special case: red symbol on dark background
     (destructive / cancel) — follows the Stop-generation color pattern.

   objectName mapping:
     Named Primary    — (default, no objectName)          — e.g. Upload, Send, Save
     Named Secondary  — "secondaryButton" (alias: "attachButton")
                                                          — e.g. Close, Cancel, New Chat
     Icon Primary     — "iconPrimary"                     — e.g. Download  (↓)
     Icon Secondary   — "iconSecondary" (red, destructive) — e.g. Cancel (✕), Delete row (−)

   Specialty (unique behavior, outside the shape × variant grid):
     "stopButton"      — Stop generation (red-when-enabled)
     "chatMenuButton"  — Hover-only ⋯ in chat list items
     "genderButton"    — Toggle for profile gender selection
     "fileChip"        — Removable file attachment pill
   ────────────────────────────────────────────────────────────────────── */

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

/* Named Secondary */
QPushButton#attachButton,
QPushButton#secondaryButton {
    background-color: #313244;
    color: #cdd6f4;
    padding: 8px 12px;
}

QPushButton#attachButton:hover,
QPushButton#secondaryButton:hover {
    background-color: #45475a;
}

/* Load Model — Named Secondary with a disabled state */
QPushButton#loadModelButton {
    background-color: #313244;
    color: #cdd6f4;
    padding: 8px 12px;
}

QPushButton#loadModelButton:hover {
    background-color: #45475a;
}

QPushButton#loadModelButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}

/* Icon Primary — 28×28 blue */
QPushButton#iconPrimary {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 0;
    font-size: 14px;
    font-weight: bold;
}

QPushButton#iconPrimary:hover {
    background-color: #b4d0fb;
}

QPushButton#iconPrimary:pressed {
    background-color: #74c7ec;
}

QPushButton#iconPrimary:disabled {
    background-color: #45475a;
    color: #6c7086;
}

/* Icon Secondary — 28×28 dark background, red symbol (destructive) */
QPushButton#iconSecondary {
    background-color: #313244;
    color: #45475a;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 0;
    font-size: 14px;
    font-weight: bold;
}

QPushButton#iconSecondary:enabled {
    color: #f38ba8;
}

QPushButton#iconSecondary:enabled:hover {
    background-color: #45475a;
    border-color: #f38ba8;
}

QPushButton#iconSecondary:disabled {
    background-color: #45475a;
    color: #6c7086;
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
