"""Trends section content: scrollable line-chart dashboard of biomarker
trends extracted from the user's medical documents."""

from __future__ import annotations

from datetime import datetime
from math import ceil, floor, log10
from typing import Any

from PySide6.QtCharts import (
    QAreaSeries,
    QChart,
    QChartView,
    QDateTimeAxis,
    QLineSeries,
    QScatterSeries,
    QValueAxis,
)
from PySide6.QtCore import QDateTime, QMargins, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.biomarkers import (
    BIOMARKERS_FILE,
    BiomarkerExtractionWorker,
    aggregate_for_dashboard,
    clear_cache,
    docs_pending_for_update,
    has_cache,
)
from core.llm_engine import is_server_running


# Palette
_COLOR_LINE = QColor("#89b4fa")
_COLOR_LINE_LIGHT = QColor("#a4c2fb")
_COLOR_OK = QColor("#a6e3a1")
_COLOR_BAD = QColor("#f38ba8")
_COLOR_BAND = QColor(166, 227, 161, 40)  # translucent green
_COLOR_GRID = QColor("#45475a")
_COLOR_TEXT = QColor("#cdd6f4")
_ACTION_BUTTON_SIZE = (100, 38)


def _iso_to_qdatetime(iso: str) -> QDateTime:
    dt = datetime.strptime(iso, "%Y-%m-%d")
    return QDateTime(dt)


def _nice_number(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = floor(log10(value))
    fraction = value / (10 ** exponent)
    if fraction < 1.5:
        nice_fraction = 1
    elif fraction < 3:
        nice_fraction = 2
    elif fraction < 7:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10 ** exponent)


def _nice_axis_range(
    values: list[float], *, target_intervals: int = 4
) -> tuple[float, float, int, str]:
    raw_min = min(values)
    raw_max = max(values)
    span = max(raw_max - raw_min, abs(raw_max) * 0.1, 1.0)
    step = _nice_number(span / target_intervals)
    nice_min = floor((raw_min - step * 0.5) / step) * step
    nice_max = ceil((raw_max + step * 0.5) / step) * step
    if raw_min >= 0 and nice_min < 0:
        nice_min = 0
    tick_count = int(round((nice_max - nice_min) / step)) + 1
    decimals = max(0, -floor(log10(step))) if step < 1 else 0
    return nice_min, nice_max, tick_count, f"%.{decimals}f"


def _point_key(x: int, y: float) -> tuple[int, float]:
    return x, round(y, 12)


def _format_point_tooltip(name: str, unit: str, point: dict[str, Any], value: float) -> str:
    value_text = f"{value:g}" + (f" {unit}" if unit else "")
    lines = [name, f"Date: {point['date']}", f"Value: {value_text}"]
    source = point.get("source_doc")
    if source:
        lines.append(f"Source: {source}")
    return "\n".join(lines)


def _padded_time_range(timestamps: list[int]) -> tuple[int, int]:
    x_min = min(timestamps)
    x_max = max(timestamps)
    day_ms = 24 * 60 * 60 * 1000
    if x_min == x_max:
        return x_min - day_ms, x_max + day_ms
    padding = max(day_ms, round((x_max - x_min) * 0.06))
    return x_min - padding, x_max + padding


def _show_point_value_label(
    point: Any,
    state: bool,
    tooltips: dict[tuple[int, float], str],
    view: QChartView,
    series: QScatterSeries,
) -> None:
    label = view._point_value_label
    if not state:
        label.hide()
        return
    text = tooltips.get(_point_key(int(round(point.x())), point.y()))
    if not text:
        label.hide()
        return

    label.setText(text)
    label.adjustSize()
    dot_pos = view.mapFromScene(view.chart().mapToPosition(point, series))
    x = dot_pos.x() + 12
    y = dot_pos.y() - label.height() - 8
    x = max(4, min(x, view.viewport().width() - label.width() - 4))
    y = max(4, min(y, view.viewport().height() - label.height() - 4))
    label.move(x, y)
    label.show()


def _build_chart(name: str, slot: dict[str, Any]) -> QChartView:
    """One line+scatter+band chart for a single biomarker time series."""
    points = slot["points"]
    unit = slot.get("unit", "")
    chart = QChart()
    chart.setTitle(f"{name}" + (f"  ({unit})" if unit else ""))
    chart.setBackgroundBrush(QBrush(QColor("#1e1e2e")))
    chart.setTitleBrush(QBrush(_COLOR_TEXT))
    chart.legend().setVisible(False)
    chart.setMargins(QMargins(12, 12, 12, 12))

    line = QLineSeries()
    line.setColor(_COLOR_LINE)
    pen = QPen(_COLOR_LINE)
    pen.setWidth(2)
    line.setPen(pen)

    scatter_ok = QScatterSeries()
    scatter_ok.setColor(_COLOR_OK)
    scatter_ok.setMarkerSize(10.0)
    scatter_ok.setBorderColor(QColor("#1e1e2e"))

    scatter_bad = QScatterSeries()
    scatter_bad.setColor(_COLOR_BAD)
    scatter_bad.setMarkerSize(10.0)
    scatter_bad.setBorderColor(QColor("#1e1e2e"))

    scatter_unknown = QScatterSeries()
    scatter_unknown.setColor(_COLOR_LINE_LIGHT)
    scatter_unknown.setMarkerSize(10.0)
    scatter_unknown.setBorderColor(QColor("#1e1e2e"))

    series_refs: list[Any] = [line, scatter_ok, scatter_bad, scatter_unknown]
    scatter_tooltips: dict[QScatterSeries, dict[tuple[int, float], str]] = {
        scatter_ok: {},
        scatter_bad: {},
        scatter_unknown: {},
    }
    timestamps: list[int] = []
    values: list[float] = []
    for p in points:
        ts = _iso_to_qdatetime(p["date"]).toMSecsSinceEpoch()
        v = float(p["value"])
        timestamps.append(ts)
        values.append(v)
        line.append(ts, v)
        if p["in_range"] is True:
            scatter_ok.append(ts, v)
            scatter = scatter_ok
        elif p["in_range"] is False:
            scatter_bad.append(ts, v)
            scatter = scatter_bad
        else:
            scatter_unknown.append(ts, v)
            scatter = scatter_unknown
        scatter_tooltips[scatter][_point_key(ts, v)] = _format_point_tooltip(
            name, unit, p, v
        )

    # Reference range band: take the latest measurement's range as the
    # representative band (most labs use similar bounds; if they diverge
    # the per-point colors will still tell the story).
    band_low = next(
        (p["ref_low"] for p in reversed(points) if p["ref_low"] is not None), None
    )
    band_high = next(
        (p["ref_high"] for p in reversed(points) if p["ref_high"] is not None), None
    )

    band_added = False
    if band_low is not None and band_high is not None and points:
        upper = QLineSeries()
        lower = QLineSeries()
        for p in points:
            ts = _iso_to_qdatetime(p["date"]).toMSecsSinceEpoch()
            upper.append(ts, band_high)
            lower.append(ts, band_low)
        area = QAreaSeries(upper, lower)
        area.setColor(_COLOR_BAND)
        area.setBorderColor(QColor(0, 0, 0, 0))
        chart.addSeries(area)
        series_refs.extend([upper, lower, area])
        band_added = True

    chart.addSeries(line)
    chart.addSeries(scatter_ok)
    chart.addSeries(scatter_bad)
    chart.addSeries(scatter_unknown)

    axis_x = QDateTimeAxis()
    axis_x.setFormat("yyyy-MM")
    axis_x.setTickCount(min(8, max(2, len(points))))
    axis_x.setLabelsBrush(QBrush(_COLOR_TEXT))
    axis_x.setGridLineColor(_COLOR_GRID)
    axis_x.setLinePenColor(_COLOR_GRID)
    x_min, x_max = _padded_time_range(timestamps)
    axis_x.setRange(
        QDateTime.fromMSecsSinceEpoch(x_min),
        QDateTime.fromMSecsSinceEpoch(x_max),
    )
    chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)

    axis_y = QValueAxis()
    # Pad y-range so points and band aren't flush against the edges.
    candidates = list(values)
    if band_low is not None:
        candidates.append(band_low)
    if band_high is not None:
        candidates.append(band_high)
    y_min, y_max, tick_count, label_format = _nice_axis_range(candidates)
    axis_y.setRange(y_min, y_max)
    axis_y.setTickCount(tick_count)
    axis_y.setLabelFormat(label_format)
    axis_y.setTruncateLabels(False)
    axis_y.setLabelsBrush(QBrush(_COLOR_TEXT))
    axis_y.setGridLineColor(_COLOR_GRID)
    axis_y.setLinePenColor(_COLOR_GRID)
    chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)

    for s in (line, scatter_ok, scatter_bad, scatter_unknown):
        s.attachAxis(axis_x)
        s.attachAxis(axis_y)
    if band_added:
        # Also attach the area series' axes to render correctly.
        chart.series()[0].attachAxis(axis_x)
        chart.series()[0].attachAxis(axis_y)

    view = QChartView(chart)
    view.setRenderHint(QPainter.RenderHint.Antialiasing)
    view.setMinimumHeight(220)
    view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    view.setStyleSheet("background: #1e1e2e; border: 1px solid #313244; border-radius: 8px;")
    view._series_refs = series_refs
    view._point_tooltips = scatter_tooltips
    value_label = QLabel(view.viewport())
    value_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    value_label.setStyleSheet(
        "background-color: #313244; color: #cdd6f4; border: 1px solid #89b4fa; "
        "border-radius: 6px; padding: 4px 6px; font-size: 11px;"
    )
    value_label.hide()
    view._point_value_label = value_label
    for scatter, tooltips in scatter_tooltips.items():
        scatter.hovered.connect(
            lambda point, state, scatter=scatter, tooltips=tooltips: _show_point_value_label(
                point, state, tooltips, view, scatter
            )
        )
    return view


def _panel_header(title: str) -> QLabel:
    lbl = QLabel(title)
    lbl.setStyleSheet(
        "font-size: 16px; font-weight: bold; color: #89b4fa;"
        " margin: 8px 0 4px 0;"
    )
    return lbl


class TrendsContent(QWidget):
    """Trends dashboard. Generate extracts everything from scratch; Update
    extracts only documents new since last extraction."""

    PANEL_ORDER = [
        "Reproductive Hormones",
        "Hormones",
        "Thyroid",
        "Lipid",
        "Glucose",
        "Liver Function",
        "Kidney Function",
        "CBC",
        "Vitamins & Minerals",
        "Inflammation",
        "Coagulation",
        "Spermogram",
        "Imaging",
        "Other",
    ]
    RENDER_BATCH_SIZE = 2
    RENDER_BATCH_DELAY_MS = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: BiomarkerExtractionWorker | None = None
        self._render_generation = 0
        self._render_queue: list[tuple[QGridLayout, int, int, str, dict[str, Any]]] = []
        self._rendered_cache_signature: int | None = None
        self._pending_render_generation = 0
        self._pending_cache_signature: int | None = None
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._continue_render_queue)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        self.setObjectName("trendsContent")

        # ── Action buttons row ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setFixedSize(*_ACTION_BUTTON_SIZE)
        self._generate_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self._generate_btn)

        self._update_btn = QPushButton("Update")
        self._update_btn.setObjectName("attachButton")
        self._update_btn.setFixedSize(*_ACTION_BUTTON_SIZE)
        self._update_btn.clicked.connect(self._on_update)
        btn_row.addWidget(self._update_btn)

        btn_row.addStretch()

        self._delete_all_btn = QPushButton("Delete All")
        self._delete_all_btn.setObjectName("stopButton")
        self._delete_all_btn.setFixedSize(*_ACTION_BUTTON_SIZE)
        self._delete_all_btn.clicked.connect(self._on_delete_all)
        btn_row.addWidget(self._delete_all_btn)

        layout.addLayout(btn_row)

        # ── Status + progress ──
        self._status = QLabel("")
        self._status.setObjectName("statusLabel")
        layout.addWidget(self._status)

        self._progress_widget = QWidget()
        prow = QHBoxLayout(self._progress_widget)
        prow.setContentsMargins(0, 0, 0, 0)
        prow.setSpacing(8)
        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setMaximum(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        prow.addWidget(self._progress, stretch=1)
        self._cancel_btn = QPushButton("✕")
        self._cancel_btn.setObjectName("iconSecondary")
        self._cancel_btn.setFixedSize(28, 28)
        self._cancel_btn.setToolTip("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        prow.addWidget(self._cancel_btn)
        self._progress_widget.setVisible(False)
        layout.addWidget(self._progress_widget)

        # ── Charts area (scrollable) ──
        self._scroll = QScrollArea()
        self._scroll.setObjectName("trendsScrollArea")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._charts_host = QWidget()
        self._charts_host.setObjectName("trendsChartsHost")
        self._charts_layout = QVBoxLayout(self._charts_host)
        self._charts_layout.setContentsMargins(0, 0, 0, 0)
        self._charts_layout.setSpacing(8)
        self._scroll.setWidget(self._charts_host)
        layout.addWidget(self._scroll, stretch=1)

        # Empty-state label
        self._empty = QLabel(
            "No trends yet. Click Generate to extract biomarker values from "
            "your uploaded medical documents and chart their trends over time."
        )
        self._empty.setObjectName("statusLabel")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        layout.addWidget(self._empty, stretch=1)

        self._scroll.setVisible(False)
        self._empty.setVisible(True)
        self._update_button_states()

    # ── Public ────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        self._render_charts()
        self._update_button_states()

    def hideEvent(self, event) -> None:
        self._cancel_pending_render()
        super().hideEvent(event)

    # ── Buttons ───────────────────────────────────────────────────────────

    def _on_generate(self) -> None:
        if self._is_busy():
            return
        if not is_server_running():
            self._status.setText(
                "No AI model is loaded. Open the Models tab and click "
                "Load on the model you want to use, then try again."
            )
            return
        # Generate regenerates from scratch — drop the cache so the worker
        # starts clean.
        clear_cache()
        self._render_charts()  # clears the dashboard while we extract
        self._start_worker(BiomarkerExtractionWorker.MODE_GENERATE)

    def _on_update(self) -> None:
        if self._is_busy():
            return
        if not is_server_running():
            self._status.setText(
                "No AI model is loaded. Open the Models tab and click "
                "Load on the model you want to use, then try again."
            )
            return
        if not docs_pending_for_update():
            self._status.setText("No new documents since last extraction.")
            return
        self._start_worker(BiomarkerExtractionWorker.MODE_UPDATE)

    def _on_delete_all(self) -> None:
        if self._is_busy() or not has_cache():
            return
        clear_cache()
        self._status.setText("Deleted all trends.")
        self.refresh()

    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._cancel_btn.setEnabled(False)
            self._status.setText("Cancelling...")

    # ── Worker plumbing ───────────────────────────────────────────────────

    def _start_worker(self, mode: str) -> None:
        self._set_busy(True)
        self._cancel_btn.setEnabled(True)
        self._status.setText(
            "Generating trends..." if mode == BiomarkerExtractionWorker.MODE_GENERATE
            else "Updating biomarkers..."
        )
        self._progress.setMinimum(0)
        self._progress.setMaximum(0)  # indeterminate until first doc done
        self._worker = BiomarkerExtractionWorker(mode=mode)
        self._worker.progress.connect(self._status.setText)
        self._worker.doc_progress.connect(self._on_doc_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _on_doc_progress(self, done: int, total: int) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(done)

    def _on_finished(self) -> None:
        self._set_busy(False)
        self._status.setText("Trends ready.")
        self.refresh()

    def _on_error(self, msg: str) -> None:
        self._set_busy(False)
        self._status.setText(f"Error: {msg}")
        self._update_button_states()

    def _on_cancelled(self) -> None:
        self._set_busy(False)
        self._status.setText("Cancelled.")
        self._update_button_states()

    # ── State helpers ─────────────────────────────────────────────────────

    def _is_busy(self) -> bool:
        return (self._worker is not None and self._worker.isRunning()) or self._progress_widget.isVisible()

    def _set_busy(self, busy: bool) -> None:
        self._progress_widget.setVisible(busy)
        self._update_button_states()

    def _update_button_states(self) -> None:
        busy = self._is_busy()
        self._generate_btn.setEnabled(not busy)
        self._update_btn.setEnabled(not busy and has_cache() and bool(docs_pending_for_update()))
        self._delete_all_btn.setEnabled(not busy and has_cache())

    # ── Rendering ─────────────────────────────────────────────────────────

    def _render_charts(self) -> None:
        cache_signature = self._cache_signature()
        if (
            self._rendered_cache_signature == cache_signature
            and not self._render_queue
            and self._charts_layout.count() > 0
        ):
            return

        self._render_generation += 1
        generation = self._render_generation
        self._render_queue = []
        self._render_timer.stop()
        self._pending_cache_signature = None

        # Clear existing charts
        while self._charts_layout.count():
            item = self._charts_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        grouped = aggregate_for_dashboard()
        if not grouped:
            self._scroll.setVisible(False)
            self._empty.setVisible(True)
            self._rendered_cache_signature = cache_signature
            return
        self._scroll.setVisible(True)
        self._empty.setVisible(False)

        # Group by panel, sort panels by PANEL_ORDER then alphabetically.
        by_panel: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for name, slot in grouped.items():
            by_panel.setdefault(slot.get("panel", "Other"), []).append((name, slot))

        ordered = sorted(
            by_panel.items(),
            key=lambda kv: (
                self.PANEL_ORDER.index(kv[0]) if kv[0] in self.PANEL_ORDER else 999,
                kv[0],
            ),
        )

        for panel_idx, (panel, items) in enumerate(ordered):
            # Breathing room between panel sections (not above the first one,
            # since the dashboard's outer margin already gives that).
            if panel_idx > 0:
                spacer = QWidget()
                spacer.setFixedHeight(24)
                spacer.setStyleSheet("background: transparent;")
                self._charts_layout.addWidget(spacer)

            self._charts_layout.addWidget(_panel_header(panel))
            grid_wrap = QWidget()
            grid_layout = QGridLayout(grid_wrap)
            grid_layout.setContentsMargins(0, 0, 0, 0)
            grid_layout.setSpacing(8)
            grid_layout.setColumnStretch(0, 1)
            grid_layout.setColumnStretch(1, 1)

            items.sort(key=lambda kv: kv[0].lower())
            for i, (name, slot) in enumerate(items):
                self._render_queue.append((grid_layout, i // 2, i % 2, name, slot))
            self._charts_layout.addWidget(grid_wrap)
        self._charts_layout.addStretch()
        self._render_next_batch(generation, cache_signature)

    def _cache_signature(self) -> int | None:
        try:
            return BIOMARKERS_FILE.stat().st_mtime_ns
        except OSError:
            return None

    def _render_next_batch(self, generation: int, cache_signature: int | None) -> None:
        if generation != self._render_generation:
            return
        self._pending_render_generation = generation
        self._pending_cache_signature = cache_signature

        for _ in range(min(self.RENDER_BATCH_SIZE, len(self._render_queue))):
            grid_layout, row, col, name, slot = self._render_queue.pop(0)
            grid_layout.addWidget(_build_chart(name, slot), row, col)

        if self._render_queue:
            self._render_timer.start(self.RENDER_BATCH_DELAY_MS)
            return

        self._pending_cache_signature = None
        self._rendered_cache_signature = cache_signature

    def _continue_render_queue(self) -> None:
        if not self._render_queue:
            return
        self._render_next_batch(
            self._pending_render_generation,
            self._pending_cache_signature,
        )

    def _cancel_pending_render(self) -> None:
        if not self._render_queue and not self._render_timer.isActive():
            return
        self._render_generation += 1
        self._render_queue = []
        self._render_timer.stop()
        self._pending_cache_signature = None


__all__ = ["TrendsContent"]
