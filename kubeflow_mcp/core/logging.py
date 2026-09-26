# Copyright The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structured logging with correlation IDs."""

import json
import logging
import re
import sys
import uuid
from collections import deque
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from kubeflow_mcp.core.security import is_sensitive_key, mask_sensitive_data

correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
request_context: ContextVar[dict[str, Any] | None] = ContextVar("request_context", default=None)

_log_buffer: deque[dict[str, Any]] = deque(maxlen=1000)

# A key needs a real ``=`` or ``:`` after it, otherwise prose such as
# ``bearer auth`` reads as a key and ordinary log lines get mangled.
_REDACT_KEY = re.compile(r"(?<!\w)(\w+)\s*[=:]\s*")
# Taking the scheme word too covers ``Authorization: Bearer <jwt>``.
_REDACT_VALUE = re.compile(r"(?:(?:bearer|basic|digest|token)\s+)?\S+", re.IGNORECASE)


def _redact_text(text: str) -> str:
    """Redact credential values in free text.

    Every log path routes through here. Keys are judged by ``is_sensitive_key``,
    the same check ``mask_sensitive_data`` uses. A value that follows a harmless
    key is still scanned, which is how ``user: password=hunter2`` gets caught.
    """
    pieces: list[str] = []
    pos = 0
    for key in _REDACT_KEY.finditer(text):
        if key.start() < pos or not is_sensitive_key(key.group(1)):
            continue
        value = _REDACT_VALUE.match(text, key.end())
        if value is None:
            continue
        pieces.append(text[pos : key.end()])
        pieces.append("***")
        pos = value.end()
    pieces.append(text[pos:])
    return "".join(pieces)


def _redact_dict(d: Any) -> Any:
    """Recursively redact sensitive data for logging."""

    if isinstance(d, dict):
        return _apply_pattern(mask_sensitive_data(d))
    if isinstance(d, list):
        return [_redact_dict(item) for item in d]

    return d


def _apply_pattern(d: Any) -> Any:
    """Apply _redact_text to string leaves in an already key-masked structure."""
    if isinstance(d, dict):
        return {k: _apply_pattern(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_apply_pattern(v) for v in d]
    if isinstance(d, str):
        return _redact_text(d)

    return d


class StructuredFormatter(logging.Formatter):
    """JSON formatter for production."""

    def format(self, record: logging.LogRecord) -> str:
        log_dict: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _redact_text(record.getMessage()),
            "correlation_id": correlation_id.get() or None,
        }

        if record.exc_info:
            log_dict["exception"] = _redact_text(self.formatException(record.exc_info))

        ctx = request_context.get()
        if ctx is not None:
            log_dict["context"] = _redact_dict(ctx)

        extra_keys = {"audit", "tool", "parameters", "success", "duration_ms", "tracing_enabled"}
        for key in extra_keys:
            if hasattr(record, key):
                value = getattr(record, key)
                log_dict[key] = _redact_dict(value) if isinstance(value, (dict, list)) else value

        return json.dumps(log_dict, default=str)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        cid = correlation_id.get()
        cid_str = f" [{cid[:8]}]" if cid else ""
        return (
            f"{color}{record.levelname:8}{self.RESET}{cid_str} "
            f"{record.name}: {_redact_text(record.getMessage())}"
        )


class BufferingHandler(logging.Handler):
    """Handler that stores logs in memory buffer with sensitive data redacted."""

    def emit(self, record: logging.LogRecord) -> None:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _redact_text(record.getMessage()),
        }
        _log_buffer.append(log_entry)


def setup_logging(
    level: str = "INFO",
    format: str | None = None,
) -> logging.Logger:
    """Configure logging for kubeflow-mcp.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Log format (json, console). Auto-detects if None.
    """
    if format is None:
        format = "console" if sys.stderr.isatty() else "json"

    formatter: logging.Formatter
    if format == "json":
        formatter = StructuredFormatter()
    else:
        formatter = ConsoleFormatter()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    buffer_handler = BufferingHandler()
    buffer_handler.setLevel(logging.DEBUG)

    root = logging.getLogger("kubeflow_mcp")
    root.setLevel(getattr(logging, level.upper()))
    root.handlers.clear()
    root.addHandler(handler)
    root.addHandler(buffer_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the kubeflow_mcp prefix."""
    return logging.getLogger(f"kubeflow_mcp.{name}")


def with_correlation_id() -> str:
    """Generate and set a new correlation ID."""
    cid = str(uuid.uuid4())
    correlation_id.set(cid)
    return cid


def get_log_buffer() -> list[dict[str, Any]]:
    """Get recent log entries from buffer."""
    return list(_log_buffer)
