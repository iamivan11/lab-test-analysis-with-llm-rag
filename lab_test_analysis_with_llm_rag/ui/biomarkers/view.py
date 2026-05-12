"""Trends section content: scrollable line-chart dashboard of biomarker
trends extracted from the user's medical documents."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
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

from config import (
    add_hidden_biomarker,
    clear_hidden_biomarkers,
    load_hidden_biomarkers,
)
from core.biomarkers import (
    BIOMARKERS_FILE,
    BiomarkerExtractionWorker,
    aggregate_for_dashboard,
    clear_cache,
    has_cache,
    has_pending_refresh,
    list_uploaded_docs,
)
from core.llm_engine import is_server_running
from ui.biomarkers.charts import build_chart
from ui.components import StatsBar, TimedStatusLabel, block_header, icon_button


_ACTION_BUTTON_SIZE = (100, 38)


class TrendsContent(QWidget):
    """Trends dashboard. Generate extracts everything from scratch; Refresh
    extracts only documents new since last extraction; Rebuild restores any
    individually hidden charts without re-extracting."""

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
        # Injected by the host (main_window) so Generate/Refresh can refuse
        # when another LLM-using worker is already on llama-server.
        from collections.abc import Callable

        self.busy_check: Callable[[], bool] | None = None
        self._worker: BiomarkerExtractionWorker | None = None
        self._render_generation = 0
        self._render_queue: list[tuple[QGridLayout, int, int, str, dict[str, Any]]] = []
        self._rendered_cache_signature: tuple | None = None
        self._pending_render_generation = 0
        self._pending_cache_signature: tuple | None = None
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._continue_render_queue)
        # Scroll position to restore once the (async, batched) re-render
        # finishes — preserves the user's place after hide/Rebuild/Refresh.
        self._restore_scroll_y: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        self.setObjectName("trendsContent")

        # Mirrors the chat top-bar chips (model, memory, CPU, context).
        # MainWindow.register_stats_bar pushes updates here.
        self.stats_bar = StatsBar()
        layout.addWidget(self.stats_bar)

        # ── Action buttons row ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setFixedSize(*_ACTION_BUTTON_SIZE)
        self._generate_btn.setToolTip(
            "Extract biomarkers from all documents from scratch"
        )
        self._generate_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self._generate_btn)

        self._rebuild_btn = QPushButton("Rebuild")
        self._rebuild_btn.setObjectName("attachButton")
        self._rebuild_btn.setFixedSize(*_ACTION_BUTTON_SIZE)
        self._rebuild_btn.setToolTip("Restore all removed charts")
        self._rebuild_btn.clicked.connect(self._on_rebuild)
        btn_row.addWidget(self._rebuild_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setObjectName("attachButton")
        self._refresh_btn.setFixedSize(*_ACTION_BUTTON_SIZE)
        self._refresh_btn.setToolTip(
            "Extract biomarkers only from new documents"
        )
        self._refresh_btn.clicked.connect(self._on_refresh)
        btn_row.addWidget(self._refresh_btn)

        btn_row.addStretch()

        self._delete_all_btn = QPushButton("Delete All")
        self._delete_all_btn.setObjectName("stopButton")
        self._delete_all_btn.setFixedSize(*_ACTION_BUTTON_SIZE)
        self._delete_all_btn.setToolTip("Delete all extracted biomarkers")
        self._delete_all_btn.clicked.connect(self._on_delete_all)
        btn_row.addWidget(self._delete_all_btn)

        layout.addLayout(btn_row)

        # ── Status + progress ──
        self._status = TimedStatusLabel("")
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
        self._cancel_btn = icon_button(
            "✕", tooltip="Cancel", on_click=self._on_cancel
        )
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
        # 24 px right gutter so charts stay clear of the macOS scrollbar
        # whether it's overlay (~7-9 px wide, brief) or classic-always-on
        # (~16 px wide, when the user has "Always show scrollbars" set
        # in System Settings → Appearance).
        self._charts_layout.setContentsMargins(0, 0, 24, 0)
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

    def _refuse_if_llm_busy(self) -> bool:
        """Refuse to start a new LLM call while another is in flight.

        Without this, the new request just queues at llama-server
        (single-slot) and the user sees "Generating trends..." stuck
        for minutes while parsing or another extraction finishes —
        looks broken even though it's slow.
        """
        if self.busy_check is not None and self.busy_check():
            self._status.setText(
                "Wait for the current operation to finish or cancel it, "
                "then try again."
            )
            return True
        return False

    def _on_generate(self) -> None:
        if self._is_busy() or self._refuse_if_llm_busy():
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

    def _on_refresh(self) -> None:
        if self._is_busy() or self._refuse_if_llm_busy():
            return
        if not is_server_running():
            self._status.setText(
                "No AI model is loaded. Open the Models tab and click "
                "Load on the model you want to use, then try again."
            )
            return
        if not has_pending_refresh():
            self._status.setText("Trends already in sync with documents.")
            return
        self._start_worker(BiomarkerExtractionWorker.MODE_UPDATE)

    def _on_rebuild(self) -> None:
        if self._is_busy():
            return
        clear_hidden_biomarkers()
        self.refresh()

    def _on_remove_biomarker(self, name: str) -> None:
        if self._is_busy():
            return
        add_hidden_biomarker(name)
        self.refresh()

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

    def _on_finished(self, added: int, removed: int) -> None:
        self._set_busy(False)
        parts = []
        if added > 0:
            parts.append(f"added {added}")
        if removed > 0:
            parts.append(f"removed {removed}")
        if parts:
            self._status.setText("Trends ready — " + ", ".join(parts) + " document(s).")
        else:
            self._status.setText("Trends ready.")
        self.refresh()

    def _on_error(self, msg: str) -> None:
        from core.messages import EXTRACTION_FAILED

        self._set_busy(False)
        self._status.setText(EXTRACTION_FAILED.format(msg=msg))
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
        has_docs = bool(list_uploaded_docs())
        self._generate_btn.setEnabled(not busy and has_docs)
        self._refresh_btn.setEnabled(not busy and has_cache() and has_pending_refresh())
        self._delete_all_btn.setEnabled(not busy and has_cache())
        self._rebuild_btn.setEnabled(not busy and bool(load_hidden_biomarkers()))

    # ── Rendering ─────────────────────────────────────────────────────────

    def _render_charts(self) -> None:
        cache_signature = self._cache_signature()
        if (
            self._rendered_cache_signature == cache_signature
            and not self._render_queue
            and self._charts_layout.count() > 0
        ):
            return

        # Capture the user's current scroll position before we tear the
        # dashboard down. _render_next_batch will restore it after the
        # last batch lands so hide/Rebuild/Refresh feel in-place.
        if self._charts_layout.count() > 0:
            self._restore_scroll_y = self._scroll.verticalScrollBar().value()

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
        hidden = load_hidden_biomarkers()
        if hidden:
            grouped = {n: s for n, s in grouped.items() if n not in hidden}
        if not grouped:
            self._scroll.setVisible(False)
            self._empty.setVisible(True)
            self._rendered_cache_signature = cache_signature
            self._restore_scroll_y = None
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

            self._charts_layout.addWidget(block_header(panel, margin="8px 0 4px 0"))
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

    def _cache_signature(self) -> tuple | None:
        try:
            mtime = BIOMARKERS_FILE.stat().st_mtime_ns
        except OSError:
            return None
        # Include hidden set so hide/Rebuild trigger a re-render even
        # though the cache file itself hasn't changed.
        return (mtime, frozenset(load_hidden_biomarkers()))

    def _render_next_batch(self, generation: int, cache_signature: tuple | None) -> None:
        if generation != self._render_generation:
            return
        self._pending_render_generation = generation
        self._pending_cache_signature = cache_signature

        for _ in range(min(self.RENDER_BATCH_SIZE, len(self._render_queue))):
            grid_layout, row, col, name, slot = self._render_queue.pop(0)
            grid_layout.addWidget(
                build_chart(name, slot, on_remove=self._on_remove_biomarker),
                row,
                col,
            )

        if self._render_queue:
            self._render_timer.start(self.RENDER_BATCH_DELAY_MS)
            return

        self._pending_cache_signature = None
        self._rendered_cache_signature = cache_signature
        if self._restore_scroll_y is not None:
            target = self._restore_scroll_y
            self._restore_scroll_y = None
            # Defer one tick: scrollbar's range is updated after the
            # layout pass, and setValue clamps to current range.
            QTimer.singleShot(
                0, lambda y=target: self._scroll.verticalScrollBar().setValue(y)
            )

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
