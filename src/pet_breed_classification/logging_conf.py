"""JSON logging configuration shared by command-line and API entry points."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime

from .config import cfg

correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-"
)

_STANDARD_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    """Format each log record as one structured JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a JSON representation of a log record."""
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def setup_logging() -> None:
    """Configure one JSON stdout handler and propagate uvicorn logs."""
    level = logging.getLevelNamesMapping().get(cfg.LOG_LEVEL.upper(), logging.INFO)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
