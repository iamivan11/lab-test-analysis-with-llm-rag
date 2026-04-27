import threading

from PySide6.QtCore import QThread, Signal

from core.llm_engine import generate_stream, summarize_history
from core.logger import log


class LLMWorker(QThread):
    thinking_token = Signal(str)
    response_token = Signal(str)
    finished_generation = Signal()
    error_occurred = Signal(str)

    def __init__(self, history: list[dict], context: str = "", max_tokens: int | None = None):
        super().__init__()
        self.history = history
        self.context = context
        self.max_tokens = max_tokens
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        log(
            "WORKER",
            f"LLMWorker: starting, history={len(self.history)} msgs, "
            f"context={len(self.context)} chars",
        )
        try:
            token_count = 0
            for kind, token in generate_stream(
                self.history,
                context=self.context,
                stop_event=self._stop_event,
                max_tokens=self.max_tokens,
            ):
                if self._stop_event.is_set():
                    log("WORKER", "LLMWorker: stopped by user")
                    break
                token_count += 1
                if kind == "thinking":
                    self.thinking_token.emit(token)
                else:
                    self.response_token.emit(token)
            log("WORKER", f"LLMWorker: finished, {token_count} tokens emitted")
            self.finished_generation.emit()
        except Exception as e:
            log("WORKER", f"LLMWorker: ERROR {e}")
            self.error_occurred.emit(str(e))


class CompressionWorker(QThread):
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, history: list[dict]):
        super().__init__()
        self.history = history

    def run(self):
        log("WORKER", f"CompressionWorker: compressing {len(self.history)} messages")
        try:
            summary = summarize_history(self.history)
            log("WORKER", f"CompressionWorker: done, summary={len(summary)} chars")
            self.finished.emit(summary)
        except Exception as e:
            log("WORKER", f"CompressionWorker: ERROR {e}")
            self.error_occurred.emit(str(e))
