from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config import load_max_tokens
from core.health_report import (
    HEALTH_REPORT_MD,
    HEALTH_REPORT_PDF,
    delete_report,
    list_uploaded_documents,
    new_documents_since_last_report,
    report_exists,
)
from core.security import read_protected_bytes, read_protected_text
from ui.components import StatsBar, TimedStatusLabel, icon_button
from ui.health_report.workers import HealthReportWorker

_ACTION_BUTTON_SIZE = (100, 38)


class HealthReportContent(QWidget):
    """Generates, updates, deletes, and renders the synthesized health report."""

    def __init__(self, build_profile_context, parent=None):
        super().__init__(parent)
        # Injected by main_window so Generate can refuse while another
        # LLM-using worker is on llama-server (single-slot — a parallel
        # request just queues for minutes and looks frozen).
        from collections.abc import Callable

        self.busy_check: Callable[[], bool] | None = None
        self._build_profile_context = build_profile_context
        self._worker: HealthReportWorker | None = None
        self._token_count = 0
        self._token_max = 0
        self._gen_start_monotonic: float | None = None
        self._mode_label = ""
        self._loaded_pdf_mtime_ns: int | None = None
        self._pdf_bytes: QByteArray | None = None
        self._pdf_buffer: QBuffer | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        # Mirrors the chat top-bar chips (model, memory, CPU, context).
        # MainWindow.register_stats_bar pushes updates here.
        self.stats_bar = StatsBar()
        layout.addWidget(self.stats_bar)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setFixedSize(*_ACTION_BUTTON_SIZE)
        self._generate_btn.clicked.connect(lambda: self._kick_off("full"))
        btn_row.addWidget(self._generate_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setObjectName("attachButton")
        self._refresh_btn.setFixedSize(*_ACTION_BUTTON_SIZE)
        self._refresh_btn.clicked.connect(lambda: self._kick_off("update"))
        btn_row.addWidget(self._refresh_btn)

        btn_row.addStretch()

        self._download_btn = QPushButton("Download")
        self._download_btn.setObjectName("attachButton")
        self._download_btn.setFixedSize(*_ACTION_BUTTON_SIZE)
        self._download_btn.clicked.connect(self._download)
        btn_row.addWidget(self._download_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setObjectName("stopButton")
        self._delete_btn.setFixedSize(*_ACTION_BUTTON_SIZE)
        self._delete_btn.clicked.connect(self._delete)
        btn_row.addWidget(self._delete_btn)

        layout.addLayout(btn_row)

        self._status_label = TimedStatusLabel("")
        layout.addWidget(self._status_label)

        self._progress_widget = QWidget()
        progress_row = QHBoxLayout(self._progress_widget)
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(8)

        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setMaximum(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        progress_row.addWidget(self._progress, stretch=1)

        self._cancel_btn = icon_button(
            "✕", tooltip="Cancel", on_click=self._on_cancel
        )
        progress_row.addWidget(self._cancel_btn)

        self._progress_widget.setVisible(False)
        layout.addWidget(self._progress_widget)

        self._pdf_doc = QPdfDocument(self)
        self._pdf_view = QPdfView()
        self._pdf_view.setDocument(self._pdf_doc)
        self._pdf_view.setPageMode(QPdfView.PageMode.MultiPage)
        self._pdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
        layout.addWidget(self._pdf_view, stretch=1)

        self._empty_label = QLabel(
            "No report yet. Click Generate to create one from your "
            "uploaded medical documents."
        )
        self._empty_label.setObjectName("statusLabel")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty_label, stretch=1)

        self._pdf_view.setVisible(False)
        self._update_button_states()

    def refresh(self) -> None:
        self._reload_pdf()
        self._update_button_states()

    def _kick_off(self, mode: str) -> None:
        if self._is_busy():
            return
        # Refuse to start while another LLM-using worker (parsing,
        # chat, trends extraction) is on the single-slot llama-server.
        # A queued request would look frozen to the user for minutes.
        if self.busy_check is not None and self.busy_check():
            self._status_label.setText(
                "Wait for the current operation to finish or cancel it, "
                "then try again."
            )
            return
        if mode == "full" and not list_uploaded_documents():
            self._status_label.setText("Upload at least one document first.")
            return
        if mode == "update":
            if not report_exists():
                return
            if not new_documents_since_last_report():
                return

        from core.llm_engine import is_server_running

        if not is_server_running():
            self._status_label.setText(
                "No AI model is loaded. Open the Models tab and click "
                "Load on the model you want to use, then try again."
            )
            return

        self._start(mode)

    def _start(self, mode: str) -> None:
        if mode == "update":
            doc_names = new_documents_since_last_report()
            if not doc_names:
                self._set_busy(False)
                return
            existing_md = read_protected_text(HEALTH_REPORT_MD)
        else:
            doc_names = list_uploaded_documents()
            existing_md = ""

        max_tokens = max(load_max_tokens() or 4096, 16384)
        self._token_count = 0
        self._token_max = max_tokens
        self._gen_start_monotonic = None
        self._mode_label = "Generating report" if mode == "full" else "Updating report"

        self._progress.setMinimum(0)
        self._progress.setMaximum(0)
        self._progress.setValue(0)
        self._set_busy(True)
        self._cancel_btn.setEnabled(True)
        self._status_label.setText(self._mode_label + "...")

        self._worker = HealthReportWorker(
            mode=mode,
            profile_context=self._build_profile_context(),
            document_names=doc_names,
            existing_markdown=existing_md,
            max_tokens=max_tokens,
        )
        self._worker.progress.connect(self._status_label.setText)
        self._worker.token.connect(self._on_token)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _delete(self) -> None:
        if self._is_busy() or not report_exists():
            return
        self._pdf_doc.close()
        self._close_pdf_buffer()
        self._loaded_pdf_mtime_ns = None
        delete_report()
        self._status_label.setText("Report deleted.")
        self.refresh()

    def _download(self) -> None:
        if not HEALTH_REPORT_PDF.exists():
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Health Report", "health_report.pdf", "PDF Files (*.pdf)"
        )
        if dest:
            try:
                dest_path = Path(dest)
                dest_path.write_bytes(read_protected_bytes(HEALTH_REPORT_PDF))
                self._status_label.setText(f"Saved to {dest}")
            except OSError as e:
                self._status_label.setText(f"Save failed: {e}")

    def _on_token(self, _tok: str) -> None:
        import time

        if self._gen_start_monotonic is None:
            self._gen_start_monotonic = time.monotonic()
            self._progress.setMaximum(self._token_max)
            self._progress.setValue(0)

        self._token_count += 1
        self._progress.setValue(min(self._token_count, self._token_max))

        if self._token_count > 1 and self._token_count % 10 != 0:
            return

        elapsed = max(time.monotonic() - self._gen_start_monotonic, 1e-3)
        rate = self._token_count / elapsed
        self._status_label.setText(
            f"{self._mode_label}: {self._token_count} tokens • {rate:.1f} tok/s"
        )

    def _on_finished(self) -> None:
        self._set_busy(False)
        self._status_label.setText("Report ready.")
        self.refresh()

    def _on_error(self, msg: str) -> None:
        self._set_busy(False)
        self._status_label.setText(f"Error: {msg}")
        self._update_button_states()

    def _on_cancelled(self) -> None:
        self._set_busy(False)
        self._status_label.setText("Cancelled.")
        self._update_button_states()

    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._cancel_btn.setEnabled(False)
            self._status_label.setText("Cancelling...")

    def _is_busy(self) -> bool:
        worker_running = self._worker is not None and self._worker.isRunning()
        return worker_running or self._progress_widget.isVisible()

    def _set_busy(self, busy: bool) -> None:
        self._progress_widget.setVisible(busy)
        self._update_button_states()

    def _update_button_states(self) -> None:
        busy = self._is_busy()
        has_docs = bool(list_uploaded_documents())
        has_report = report_exists()
        has_new = bool(new_documents_since_last_report())

        self._generate_btn.setEnabled(not busy and has_docs)
        self._refresh_btn.setEnabled(not busy and has_report and has_new)
        self._delete_btn.setEnabled(not busy and has_report)
        self._download_btn.setEnabled(not busy and has_report)

    def _reload_pdf(self) -> None:
        if HEALTH_REPORT_PDF.exists():
            mtime_ns = HEALTH_REPORT_PDF.stat().st_mtime_ns
            if self._loaded_pdf_mtime_ns != mtime_ns:
                self._pdf_doc.close()
                self._close_pdf_buffer()
                self._pdf_bytes = QByteArray(read_protected_bytes(HEALTH_REPORT_PDF))
                self._pdf_buffer = QBuffer(self._pdf_bytes, self)
                self._pdf_buffer.open(QIODevice.OpenModeFlag.ReadOnly)
                self._pdf_doc.load(self._pdf_buffer)
                self._loaded_pdf_mtime_ns = mtime_ns
            self._pdf_view.setVisible(True)
            self._empty_label.setVisible(False)
        else:
            self._pdf_doc.close()
            self._close_pdf_buffer()
            self._loaded_pdf_mtime_ns = None
            self._pdf_view.setVisible(False)
            self._empty_label.setVisible(True)

    def _close_pdf_buffer(self) -> None:
        if self._pdf_buffer is not None:
            self._pdf_buffer.close()
            self._pdf_buffer.deleteLater()
        self._pdf_buffer = None
        self._pdf_bytes = None
