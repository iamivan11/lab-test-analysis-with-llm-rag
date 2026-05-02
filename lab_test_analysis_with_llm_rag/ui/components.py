from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QScrollArea, QSizePolicy, QWidget


def header_label(text: str, *, align_center: bool = True) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-size: 24pt; font-weight: bold;")
    if align_center:
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setWordWrap(True)
    return lbl


def profile_scroll_area(content: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setObjectName("profileScrollArea")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    scroll.viewport().setObjectName("profileScrollContent")
    content.setObjectName("profileScrollContent")
    scroll.setWidget(content)
    return scroll
