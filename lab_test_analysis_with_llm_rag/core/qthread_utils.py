"""Shared QThread primitives for cooperative cancellation."""

import threading

from PySide6.QtCore import QThread


class StoppableQThread(QThread):
    """QThread with a thread-safe `_stop_event` and a `stop()` method
    that sets it. Workers either poll `self._stop_event.is_set()` between
    units of work or pass the event into long-running helpers (HTTP
    streams, downloads, parsers) so cancellation is cooperative."""

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()
