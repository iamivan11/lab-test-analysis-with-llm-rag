from PySide6.QtCore import Signal

from core.llm_engine import generate_stream, summarize_history
from core.logger import log, log_exception
from core.qthread_utils import StoppableQThread


class LLMWorker(StoppableQThread):
    thinking_token = Signal(str)
    response_token = Signal(str)
    finished_generation = Signal()
    cancelled_generation = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        history: list[dict],
        context: str = "",
        max_tokens: int | None = None,
        use_rag: bool = True,
        answer_detail: str | None = None,
    ):
        super().__init__()
        self.history = history
        self.context = context
        self.max_tokens = max_tokens
        self.use_rag = use_rag
        self.answer_detail = answer_detail

    def run(self):
        log(
            "WORKER",
            f"LLMWorker: starting, history={len(self.history)} msgs, "
            f"context={len(self.context)} chars",
        )
        try:
            token_count = 0
            cancelled = False
            for kind, token in generate_stream(
                self.history,
                context=self.context,
                stop_event=self._stop_event,
                max_tokens=self.max_tokens,
                use_rag=self.use_rag,
                answer_detail=self.answer_detail,
            ):
                if self._stop_event.is_set():
                    log("WORKER", "LLMWorker: stopped")
                    cancelled = True
                    break
                token_count += 1
                if kind == "thinking":
                    self.thinking_token.emit(token)
                else:
                    self.response_token.emit(token)
            if cancelled or self._stop_event.is_set():
                log("WORKER", f"LLMWorker: cancelled, {token_count} tokens emitted")
                self.cancelled_generation.emit()
            else:
                log("WORKER", f"LLMWorker: finished, {token_count} tokens emitted")
                self.finished_generation.emit()
        except Exception as e:
            if self._stop_event.is_set():
                log("WORKER", f"LLMWorker: cancelled via exception ({e})")
                self.cancelled_generation.emit()
                return
            log_exception("WORKER", "LLMWorker failed")
            self.error_occurred.emit(str(e))


class CompressionWorker(StoppableQThread):
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, history: list[dict]):
        super().__init__()
        self.history = history

    def run(self):
        log("WORKER", f"CompressionWorker: compressing {len(self.history)} messages")
        try:
            summary = summarize_history(self.history, stop_event=self._stop_event)
            if self._stop_event.is_set():
                log("WORKER", "CompressionWorker: cancelled after summary, skipping emit")
                return
            log("WORKER", f"CompressionWorker: done, summary={len(summary)} chars")
            self.finished.emit(summary)
        except Exception as e:
            if self._stop_event.is_set():
                log("WORKER", f"CompressionWorker: cancelled, swallowing error: {e}")
                return
            log_exception("WORKER", "CompressionWorker failed")
            self.error_occurred.emit(str(e))
