from collections.abc import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

STATUS_AUTO_CLEAR_MS = 30_000

CHIP_STYLE = (
    "font-size: 11px; color: #cdd6f4; background: #313244;"
    "border: 1px solid #45475a; border-radius: 8px;"
    "padding: 3px 8px;"
)


class BroadcastLabel(QLabel):
    """QLabel that fans setText out to subscribed listeners.

    Used for the chat top-bar status chip so peer StatsBars in other
    sections (Health Report, Trends) mirror "Model: X" / "Loading..."
    text without touching the dozens of controller call sites that
    already write to it.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._listeners: list[Callable[[str], None]] = []

    def add_listener(self, listener: Callable[[str], None]) -> None:
        self._listeners.append(listener)
        listener(self.text())

    def setText(self, text: str) -> None:  # type: ignore[override]
        super().setText(text)
        for listener in self._listeners:
            listener(text)


class TimedStatusLabel(QLabel):
    """QLabel that auto-clears its text after a 30 s idle period.

    Used for transient section-level status messages — "Report ready.",
    "Loading model: X...", "Cancelled.", etc. — that should not linger
    on screen indefinitely. The timer resets on every setText so a
    streaming progress update keeps the message visible as long as
    fresh text arrives at least once per 30 s; the label clears 30 s
    after the last update.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setObjectName("statusLabel")
        self._clear_timer = QTimer(self)
        self._clear_timer.setSingleShot(True)
        self._clear_timer.timeout.connect(lambda: super(TimedStatusLabel, self).setText(""))

    def setText(self, text: str) -> None:  # type: ignore[override]
        super().setText(text)
        if text:
            self._clear_timer.start(STATUS_AUTO_CLEAR_MS)
        else:
            self._clear_timer.stop()


class StatsBar(QWidget):
    """Top-of-section chip row: model status, memory, CPU, context.

    Same four chips — same style, same height, same horizontal
    spacing — as the chat top bar. Owned by each section that wants
    them (Health Report, Trends, ...); MainWindow pushes values in via
    the public setters on its stats tick and on model-status changes.
    """

    # Height of the chat's `☰` sidebar button, which is what forces
    # chat's top-row to render its chips as 38 px-tall pills. Sections
    # that don't have a ☰ next to the chips would otherwise render
    # the same QLabels at their natural ~22 px height, making the
    # chips look noticeably squatter. Pinning the row height here
    # keeps the visual identical across sections.
    _CHIP_ROW_HEIGHT = 38

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        self._status_chip = QLabel("No model loaded")
        self._mem_chip = QLabel("Memory: ...")
        self._cpu_chip = QLabel("CPU: ...")
        self._ctx_chip = QLabel("Context: --")
        self._ctx_chip.setToolTip(
            "Approximate context-window usage: history tokens / configured "
            "context size. Increase Context Window in Settings if you're "
            "running out."
        )

        for chip in (self._status_chip, self._mem_chip, self._cpu_chip, self._ctx_chip):
            chip.setStyleSheet(CHIP_STYLE)
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setFixedHeight(self._CHIP_ROW_HEIGHT)
            row.addWidget(chip)

        row.addStretch()

    def set_status(self, text: str) -> None:
        self._status_chip.setText(text)

    def set_memory(self, text: str) -> None:
        self._mem_chip.setText(text)

    def set_cpu(self, text: str) -> None:
        self._cpu_chip.setText(text)

    def set_context(self, text: str) -> None:
        self._ctx_chip.setText(text)


def icon_button(
    text: str,
    *,
    name: str = "iconSecondary",
    size: int = 28,
    tooltip: str = "",
    on_click: Callable[[], None] | None = None,
) -> QPushButton:
    """Square icon button (×, −, ↓, etc.) with the project's standard
    object-name styling. Default is the destructive red `iconSecondary`
    variant; pass `name="iconPrimary"` for the blue download/action one.
    """
    btn = QPushButton(text)
    btn.setObjectName(name)
    btn.setFixedSize(size, size)
    if tooltip:
        btn.setToolTip(tooltip)
    if on_click is not None:
        # Qt's clicked signal emits a `checked` bool; the helper insulates
        # callers from that so `on_click` can stay zero-arg.
        btn.clicked.connect(lambda _checked=False: on_click())
    return btn


def header_label(text: str, *, align_center: bool = True) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-size: 24pt; font-weight: bold;")
    if align_center:
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setWordWrap(True)
    return lbl


def block_header(text: str, *, margin: str | None = None) -> QLabel:
    """Shared 16 px bold blue heading for top-of-section labels in
    Profile, Settings, and Trends panels. Pass `margin` (CSS shorthand)
    to add per-call vertical breathing room — e.g. "8px 0 4px 0"."""
    lbl = QLabel(text)
    lbl.setObjectName("blockHeader")
    if margin is not None:
        lbl.setStyleSheet(f"margin: {margin};")
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
