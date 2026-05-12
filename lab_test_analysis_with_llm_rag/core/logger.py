"""Application logging with levels, rotation, and tag-prefixed messages."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import LOGGER_BACKUP_COUNT, LOGGER_LEVEL, LOGGER_MAX_BYTES

_LOGGER_NAME = "lab_test_analyzer"
_logger = logging.getLogger(_LOGGER_NAME)
_logger.setLevel(logging.DEBUG)
_logger.propagate = False

_console_handler: logging.Handler | None = None
_file_handler: logging.Handler | None = None


class _TagFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "tag"):
            record.tag = "APP"
        return True


class _TagAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("tag", self.extra["tag"])
        return msg, kwargs


def _formatter() -> logging.Formatter:
    return logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(tag)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    parsed = logging.getLevelName(level.upper())
    return parsed if isinstance(parsed, int) else logging.INFO


def _ensure_console_handler(level: int = logging.INFO) -> None:
    global _console_handler
    if _console_handler is not None:
        _console_handler.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(_formatter())
    handler.addFilter(_TagFilter())
    _logger.addHandler(handler)
    _console_handler = handler


def enable_file_logging(
    path: Path,
    *,
    level: int | str = LOGGER_LEVEL,
    max_bytes: int = LOGGER_MAX_BYTES,
    backup_count: int = LOGGER_BACKUP_COUNT,
) -> None:
    """Enable rotating file logging and stderr logging at the requested level."""
    global _file_handler

    parsed_level = _parse_level(level)
    _ensure_console_handler(parsed_level)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    except OSError as e:
        print(
            f"[LOGGER] Failed to open log file {path}: {e}",
            file=sys.stderr,
            flush=True,
        )
        _file_handler = None
        return

    handler.setLevel(parsed_level)
    handler.setFormatter(_formatter())
    handler.addFilter(_TagFilter())

    if _file_handler is not None:
        _logger.removeHandler(_file_handler)
        _file_handler.close()

    _logger.addHandler(handler)
    _file_handler = handler


def get_logger(tag: str) -> logging.LoggerAdapter:
    _ensure_console_handler()
    return _TagAdapter(_logger, {"tag": tag})


def log(tag: str, msg: str, level: int | str = "INFO") -> None:
    get_logger(tag).log(_parse_level(level), msg)


def log_exception(tag: str, msg: str) -> None:
    """Log an in-flight exception with full traceback at ERROR level.

    Use inside `except` blocks instead of `log(tag, f"... ERROR {e}")`:
    bare `str(e)` lines hide the call site (we just saw a silent crash
    where the log ended mid-task with no clue what raised). The traceback
    is rendered by the stdlib `Logger.exception()` machinery.
    """
    get_logger(tag).exception(msg)


def install_global_excepthooks() -> None:
    """Route every escaped exception — main thread, QThread, native
    signals (SIGSEGV/SIGBUS/SIGABRT) — into the same rotating log so
    post-mortem analysis is possible without re-running under a debugger.
    """
    import faulthandler
    import sys
    import threading

    # Native crashes (PyTorch/Metal/ggml/chromadb C extensions) bypass
    # Python's exception machinery entirely; faulthandler dumps every
    # thread's Python stack to stderr on a fatal signal. We also tee
    # stderr into the file handler set up by enable_file_logging, so
    # the dump survives the process death.
    faulthandler.enable(file=sys.stderr, all_threads=True)

    def _main_thread_hook(exc_type, exc_value, exc_tb):
        # KeyboardInterrupt is the user's intent; preserve default exit.
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        get_logger("CRASH").error(
            "Unhandled exception on main thread",
            exc_info=(exc_type, exc_value, exc_tb),
        )

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        if issubclass(args.exc_type, SystemExit):
            return
        thread_name = args.thread.name if args.thread else "<unknown>"
        get_logger("CRASH").error(
            f"Unhandled exception on thread {thread_name!r}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _main_thread_hook
    threading.excepthook = _thread_hook
