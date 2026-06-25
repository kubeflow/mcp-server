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

correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
request_context: ContextVar[dict[str, Any] | None] = ContextVar("request_context", default=None)

_log_buffer: deque[dict[str, Any]] = deque(maxlen=1000)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for production."""

    def format(self, record: logging.LogRecord) -> str:
        log_dict: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id.get() or None,
        }

        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        ctx = request_context.get()
        if ctx is not None:
            log_dict["context"] = ctx

        extra_keys = {"audit", "tool", "parameters", "success", "duration_ms", "tracing_enabled"}
        for key in extra_keys:
            if hasattr(record, key):
                log_dict[key] = getattr(record, key)

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
            f"{color}{record.levelname:8}{self.RESET}{cid_str} {record.name}: {record.getMessage()}"
        )


_REDACT_PATTERNS = re.compile(
    r"(token|password|secret|bearer|authorization|credential)[=: ]+\S+",
    re.IGNORECASE,
)


class BufferingHandler(logging.Handler):
    """Handler that stores logs in memory buffer with sensitive data redacted."""

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        message = _REDACT_PATTERNS.sub(
            lambda m: (
                m.group().split("=")[0] + "=***"
                if "=" in m.group()
                else m.group().split(":")[0] + ": ***"
                if ":" in m.group()
                else m.group().split()[0] + " ***"
            ),
            message,
        )
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
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
