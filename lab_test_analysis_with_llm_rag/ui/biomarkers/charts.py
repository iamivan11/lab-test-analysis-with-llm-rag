"""Chart-rendering helpers for the Trends dashboard.

Pure-presentation code: takes a biomarker time-series dict and returns
a QChartView (optionally wrapped in a _ChartFrame with a remove
button). Lives in its own module so the main TrendsContent widget
stays focused on data flow and user interactions rather than QtCharts
plumbing.
"""

from __future__ import annotations

from collections.abc import Callable
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
from PySide6.QtCore import QDateTime, QMargins, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ui.components import icon_button


# Palette
_COLOR_LINE = QColor("#89b4fa")
_COLOR_LINE_LIGHT = QColor("#a4c2fb")
_COLOR_OK = QColor("#a6e3a1")
_COLOR_BAD = QColor("#f38ba8")
_COLOR_BAND = QColor(166, 227, 161, 40)  # translucent green
_COLOR_GRID = QColor("#45475a")
_COLOR_TEXT = QColor("#cdd6f4")


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
    return "\n".join(
        [name, f"Date: {point['date']}", f"Value: {value_text}"]
        + ([f"Source: {s}"] if (s := point.get("source_doc")) else [])
    )


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
    label.move(int(dot_pos.x()) + 12, int(dot_pos.y()) - label.height() - 8)
    label.show()


class _ChartFrame(QWidget):
    """Container that keeps a QChartView and an absolutely-positioned
    overlay button as siblings. Buttons parented to QChartView.viewport()
    don't receive standard hover/click events reliably (the QGraphicsView
    consumes them at the scene level), so making the button a sibling
    instead of a viewport child is what restores normal behavior."""

    def __init__(self, chart_view: QChartView, overlay_btn: QPushButton) -> None:
        super().__init__()
        chart_view.setParent(self)
        overlay_btn.setParent(self)
        self._chart_view = chart_view
        self._overlay_btn = overlay_btn
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def resizeEvent(self, event) -> None:
        self._chart_view.setGeometry(0, 0, self.width(), self.height())
        self._overlay_btn.move(
            self.width() - self._overlay_btn.width() - 8, 8
        )
        self._overlay_btn.raise_()
        super().resizeEvent(event)


def build_chart(
    name: str,
    slot: dict[str, Any],
    on_remove: Callable[[str], None] | None = None,
) -> QWidget:
    """One line+scatter+band chart for a single biomarker time series.

    Returns the QChartView itself when no remove handler is given, or a
    container widget that holds the chart and an overlay '−' button."""
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

    # Compute axis ranges up front so the band can:
    #  (a) span the full chart x-range — anchoring it to point
    #      timestamps used to leave the right edge un-shaded once
    #      `_padded_time_range` padded past the last point;
    #  (b) fall back to the chart's y bounds when only one side of the
    #      range is known (e.g. Anti-TPO "<34" gives ref_high only —
    #      shade everything below; HDL ">40" gives ref_low only —
    #      shade everything above).
    x_min, x_max = _padded_time_range(timestamps)
    candidates = list(values)
    if band_low is not None:
        candidates.append(band_low)
    if band_high is not None:
        candidates.append(band_high)
    y_min, y_max, tick_count, label_format = _nice_axis_range(candidates)

    band_added = False
    if (band_low is not None or band_high is not None) and points:
        effective_low = band_low if band_low is not None else y_min
        effective_high = band_high if band_high is not None else y_max
        upper = QLineSeries()
        lower = QLineSeries()
        upper.append(x_min, effective_high)
        upper.append(x_max, effective_high)
        lower.append(x_min, effective_low)
        lower.append(x_max, effective_low)
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
    axis_x.setRange(
        QDateTime.fromMSecsSinceEpoch(x_min),
        QDateTime.fromMSecsSinceEpoch(x_max),
    )
    chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)

    axis_y = QValueAxis()
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

    if on_remove is None:
        return view
    remove_btn = icon_button(
        "−",  # noqa: RUF001 - UI glyph, not arithmetic.
        tooltip="Remove the chart (Rebuild restores all)",
        on_click=lambda n=name: on_remove(n),
    )
    return _ChartFrame(view, remove_btn)


__all__ = ["build_chart"]
