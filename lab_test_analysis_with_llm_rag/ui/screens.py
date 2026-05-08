from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from config import ICONS_DIR
from ui.biomarkers import TrendsContent
from ui.components import profile_scroll_area
from ui.documents.view import DocumentsHubWidget
from ui.health_report import HealthReportContent
from ui.models.hub_widget import ModelHubWidget
from ui.onboarding.view import _NAV_BTN_SIZE
from ui.profile.form import ProfileForm
from ui.sections import SectionNames
from ui.settings import SettingsForm

# ── Home ───────────────────────────────────────────────────────────────────


class _SidebarTile(QPushButton):
    """Square sidebar tile: icon centered above a bold label."""

    ICON_SIZE = 40
    SIDE = 90

    def __init__(self, label: str, icon_filename: str, *, checkable: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebarTile")
        self.setCheckable(checkable)
        self.setFixedSize(self.SIDE, self.SIDE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._icon_lbl = QLabel()
        self._icon_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_lbl.setFixedSize(self.ICON_SIZE, self.ICON_SIZE)
        self._source_pixmap = QPixmap(str(ICONS_DIR / icon_filename))
        self._apply_icon()
        layout.addWidget(self._icon_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

        text_lbl = QLabel(label)
        text_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        text_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_lbl.setWordWrap(True)
        text_lbl.setStyleSheet("font-size: 8pt; font-weight: bold; background: transparent;")
        layout.addWidget(text_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

    def _apply_icon(self) -> None:
        if self._source_pixmap.isNull():
            return
        screen = self.screen() or QApplication.primaryScreen()
        dpr = (screen.devicePixelRatio() if screen else self.devicePixelRatioF()) or 1.0
        physical = max(1, round(self.ICON_SIZE * dpr))
        scaled = self._source_pixmap.scaled(
            physical,
            physical,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(dpr)
        self._icon_lbl.setPixmap(scaled)

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_icon()

    def changeEvent(self, event):
        from PySide6.QtCore import QEvent

        if event.type() == QEvent.Type.DevicePixelRatioChange:
            self._apply_icon()
        super().changeEvent(event)


class HomeScreen(QWidget):
    """Sidebar of section tiles on the left, the active section's content on
    the right.

    The chat content is owned by MainWindow (it depends on chat controllers)
    and is injected here. Profile / Model Hub / Documents content widgets are
    built in-place and exposed for MainWindow to wire signals to.
    """

    PROFILE_INDEX = 0
    MODEL_HUB_INDEX = 1
    DOCUMENTS_INDEX = 2
    CHAT_INDEX = 3
    HEALTH_REPORT_INDEX = 4
    TRENDS_INDEX = 5
    SETTINGS_INDEX = 6

    def __init__(self, *, chat_widget: QWidget, build_profile_context, parent=None):
        super().__init__(parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Sidebar ──
        self._sidebar = QWidget()
        self._sidebar.setObjectName("sidebar")
        self._sidebar.setFixedWidth(_SidebarTile.SIDE + 24)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(6)

        # ── Content stack ──
        self._stack = QStackedWidget()

        # Profile content (form + Save).
        self._profile_form = ProfileForm()
        self._stack.addWidget(self._build_profile_content(self._profile_form))

        # Model Hub content.
        self.model_hub = ModelHubWidget()
        hub_wrap = QWidget()
        hub_layout = QVBoxLayout(hub_wrap)
        hub_layout.setContentsMargins(24, 24, 24, 24)
        hub_layout.addWidget(self.model_hub)
        self._stack.addWidget(hub_wrap)

        # Documents content.
        self.documents = DocumentsHubWidget()
        docs_wrap = QWidget()
        docs_layout = QVBoxLayout(docs_wrap)
        docs_layout.setContentsMargins(24, 24, 24, 24)
        docs_layout.addWidget(self.documents)
        self._stack.addWidget(docs_wrap)

        # Chat content (provided by MainWindow).
        self._chat_widget = chat_widget
        self._stack.addWidget(chat_widget)

        # Health Report content.
        self.health_report = HealthReportContent(build_profile_context)
        self._stack.addWidget(self.health_report)

        # Trends content.
        self.trends = TrendsContent()
        self._stack.addWidget(self.trends)

        # Settings content.
        self._settings_form = SettingsForm()
        self._settings_form.reindex_requested.connect(self.documents.reindex_files)
        self._stack.addWidget(self._build_settings_content(self._settings_form))

        # ── Tiles ──
        # Labels come from ui.sections.SectionNames so a rename in one
        # place propagates to the sidebar tiles, Settings block headers,
        # and any other place that names a section.
        tiles_def = [
            (SectionNames.PROFILE, "profile.png", self.PROFILE_INDEX),
            (SectionNames.MODELS, "model_hub.png", self.MODEL_HUB_INDEX),
            (SectionNames.MEDICAL_DOCUMENTS, "documents.png", self.DOCUMENTS_INDEX),
            (SectionNames.CHAT_WITH_DOCUMENTS, "chat_with_documents.png", self.CHAT_INDEX),
            (SectionNames.HEALTH_REPORT, "health_report.png", self.HEALTH_REPORT_INDEX),
            (SectionNames.TRENDS, "trends.png", self.TRENDS_INDEX),
        ]
        self._tiles: list[_SidebarTile] = []

        tiles_wrap = QWidget()
        tiles_layout = QVBoxLayout(tiles_wrap)
        tiles_layout.setContentsMargins(0, 0, 0, 0)
        tiles_layout.setSpacing(6)

        for label, icon, idx in tiles_def:
            tile = _SidebarTile(label, icon)
            tile.clicked.connect(lambda _checked=False, i=idx: self.activate(i))
            tiles_layout.addWidget(tile, alignment=Qt.AlignmentFlag.AlignHCenter)
            self._tiles.append(tile)
        tiles_layout.addStretch(1)

        tiles_scroll = QScrollArea()
        tiles_scroll.setObjectName("sidebarTilesScroll")
        tiles_scroll.setWidgetResizable(True)
        tiles_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        tiles_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tiles_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tiles_scroll.setWidget(tiles_wrap)
        sidebar_layout.addWidget(tiles_scroll, stretch=1)

        settings_tile = _SidebarTile(SectionNames.SETTINGS, "settings.png")
        settings_tile.clicked.connect(
            lambda _checked=False: self.activate(self.SETTINGS_INDEX)
        )
        sidebar_layout.addWidget(settings_tile, alignment=Qt.AlignmentFlag.AlignHCenter)
        self._tiles.append(settings_tile)

        outer.addWidget(self._sidebar)
        outer.addWidget(self._stack, stretch=1)

        self.activate(self.PROFILE_INDEX)

    def _build_save_row(self, save_btn: QPushButton) -> QHBoxLayout:
        save_btn.setFixedSize(*_NAV_BTN_SIZE)

        row = QHBoxLayout()
        row.setSpacing(12)
        row.addStretch()

        spacer = QWidget()
        spacer.setFixedSize(*_NAV_BTN_SIZE)
        row.addWidget(spacer)
        row.addWidget(save_btn)
        return row

    def _build_settings_content(self, form: SettingsForm) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        form.setMinimumWidth(380)
        form.setMaximumWidth(560)

        save_btn = QPushButton("Save")
        save_btn.setObjectName("settingsSaveButton")
        save_btn.clicked.connect(form.save)

        form_column = QWidget()
        form_column.setMinimumWidth(380)
        form_column.setMaximumWidth(560)
        column_layout = QVBoxLayout(form_column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(16)
        column_layout.addWidget(form)

        form_wrap = QHBoxLayout()
        form_wrap.addStretch()
        form_wrap.addWidget(form_column)
        form_wrap.addStretch()

        form_container = QWidget()
        form_container.setLayout(form_wrap)
        layout.addWidget(profile_scroll_area(form_container), stretch=1)
        layout.addLayout(self._build_save_row(save_btn))
        return wrap

    def _build_profile_content(self, form: ProfileForm) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        form_wrap = QHBoxLayout()
        form_wrap.addStretch()
        form.setMinimumWidth(440)
        form.setMaximumWidth(560)

        save_btn = QPushButton("Save")
        save_btn.setObjectName("profileSaveButton")
        save_btn.clicked.connect(form.save)

        form.submitted.connect(form.save)

        form_wrap.addWidget(form)
        form_wrap.addStretch()

        form_container = QWidget()
        form_container.setLayout(form_wrap)
        layout.addWidget(profile_scroll_area(form_container), stretch=1)
        layout.addLayout(self._build_save_row(save_btn))
        return wrap

    def activate(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        for i, tile in enumerate(self._tiles):
            tile.setChecked(i == index)
        if index == self.PROFILE_INDEX:
            self._profile_form.reload()
        elif index == self.MODEL_HUB_INDEX:
            self.model_hub.refresh_local()
        elif index == self.HEALTH_REPORT_INDEX:
            self.health_report.refresh()
        elif index == self.TRENDS_INDEX:
            self.trends.refresh()
        elif index == self.SETTINGS_INDEX:
            # Rebuild model-dependent fields (Context Window dropdown) so a
            # model swap that happened after the form was first built shows
            # up with the live model's max ctx, not the stale one.
            self._settings_form.reload()

    def show_profile(self) -> None:
        self.activate(self.PROFILE_INDEX)

    def show_chat(self) -> None:
        self.activate(self.CHAT_INDEX)


__all__ = [
    "HomeScreen",
]
