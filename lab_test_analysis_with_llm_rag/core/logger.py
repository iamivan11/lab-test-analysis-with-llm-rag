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


def set_log_level(level: int | str) -> None:
    """Update the log level for configured handlers."""
    parsed_level = _parse_level(level)
    _ensure_console_handler(parsed_level)
    if _console_handler is not None:
        _console_handler.setLevel(parsed_level)
    if _file_handler is not None:
        _file_handler.setLevel(parsed_level)


def get_logger(tag: str) -> logging.LoggerAdapter:
    _ensure_console_handler()
    return _TagAdapter(_logger, {"tag": tag})


def log(tag: str, msg: str, level: int | str = "INFO") -> None:
    get_logger(tag).log(_parse_level(level), msg)
