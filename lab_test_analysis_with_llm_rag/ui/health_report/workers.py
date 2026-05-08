"""UI workers for health-report generation."""

import threading
from datetime import datetime

from PySide6.QtCore import QThread, Signal

from core.health_report import (
    HEALTH_REPORT_MD,
    HEALTH_REPORT_PDF,
    HEALTH_REPORT_SYSTEM_PROMPT,
    build_full_report_history,
    build_update_report_history,
    gather_document_texts,
    load_metadata,
    save_metadata,
    write_report_pdf,
)
from core.llm_engine import generate_stream
from core.logger import log
from core.security import write_protected_text


class HealthReportWorker(QThread):
    """Streams report markdown, writes PDF, and reports progress to the UI."""

    progress = Signal(str)
    token = Signal(str)
    finished_ok = Signal()
    error_occurred = Signal(str)
    cancelled = Signal()

    MODE_FULL = "full"
    MODE_UPDATE = "update"

    def __init__(
        self,
        *,
        mode: str,
        profile_context: str,
        document_names: list[str],
        existing_markdown: str = "",
        max_tokens: int | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.profile_context = profile_context
        self.document_names = document_names
        self.existing_markdown = existing_markdown
        self.max_tokens = max_tokens
        self._stop_event = threading.Event()
        self._collected: list[str] = []
        self._thinking_observed = False

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        log(
            "HEALTH",
            f"HealthReportWorker: mode={self.mode}, docs={len(self.document_names)}",
        )
        try:
            self.progress.emit("Reading documents...")
            doc_texts = gather_document_texts(self.document_names)
            if self._stop_event.is_set():
                self.cancelled.emit()
                return
            if not doc_texts:
                self.error_occurred.emit(
                    "No parsed document text found. Re-upload or reindex documents."
                )
                return

            if self.mode == self.MODE_UPDATE:
                history, ctx = build_update_report_history(
                    self.profile_context, self.existing_markdown, doc_texts
                )
            else:
                history, ctx = build_full_report_history(self.profile_context, doc_texts)

            if self._stop_event.is_set():
                self.cancelled.emit()
                return

            self.progress.emit("Generating report...")
            for kind, tok in generate_stream(
                history,
                context=ctx,
                stop_event=self._stop_event,
                max_tokens=self.max_tokens,
                use_rag=False,
                enable_thinking=False,
                system_prompt_override=HEALTH_REPORT_SYSTEM_PROMPT,
            ):
                if self._stop_event.is_set():
                    log("HEALTH", "HealthReportWorker: stopped")
                    self.cancelled.emit()
                    return
                if kind == "thinking":
                    self._thinking_observed = True
                    continue
                self._collected.append(tok)
                self.token.emit(tok)

            if self._stop_event.is_set():
                self.cancelled.emit()
                return

            markdown = "".join(self._collected).strip()
            if not markdown:
                if self._thinking_observed:
                    self.error_occurred.emit(
                        "The model spent its entire output budget on "
                        "internal reasoning and produced no report. "
                        "Shorten the uploaded documents and try again."
                    )
                else:
                    self.error_occurred.emit("Model produced no report content.")
                return

            self.progress.emit("Writing PDF...")
            from ui.chat.view import render_message_html

            html = (
                "<h1>Health Report</h1>"
                f"<p style='color:#6c7086;font-size:9pt;'>Generated "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>"
                + render_message_html(markdown)
            )
            write_report_pdf(html, HEALTH_REPORT_PDF)
            write_protected_text(HEALTH_REPORT_MD, markdown)

            used_documents = (
                [name for name, _ in doc_texts]
                if self.mode == self.MODE_FULL
                else sorted(
                    set(load_metadata().get("used_documents", []))
                    | {name for name, _ in doc_texts}
                )
            )
            save_metadata(used_documents)

            self.finished_ok.emit()
        except Exception as e:
            log("HEALTH", f"HealthReportWorker: ERROR {e}")
            self.error_occurred.emit(str(e))
