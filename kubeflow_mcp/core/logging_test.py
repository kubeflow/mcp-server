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
"""Tests for structured logging module."""

import json
import logging

from kubeflow_mcp.core.logging import (
    BufferingHandler,
    ConsoleFormatter,
    StructuredFormatter,
    correlation_id,
    get_log_buffer,
    get_logger,
    request_context,
    setup_logging,
    with_correlation_id,
)


class TestWithCorrelationId:
    def test_generates_uuid(self):
        cid = with_correlation_id()
        assert len(cid) == 36
        assert cid.count("-") == 4

    def test_sets_context_var(self):
        cid = with_correlation_id()
        assert correlation_id.get() == cid

    def test_generates_unique_ids(self):
        ids = {with_correlation_id() for _ in range(10)}
        assert len(ids) == 10


class TestGetLogger:
    def test_returns_prefixed_logger(self):
        logger = get_logger("test")
        assert logger.name == "kubeflow_mcp.test"

    def test_returns_logging_instance(self):
        logger = get_logger("core")
        assert isinstance(logger, logging.Logger)


class TestStructuredFormatter:
    def setup_method(self):
        self.formatter = StructuredFormatter()
        correlation_id.set("")
        request_context.set(None)

    def test_basic_format_is_json(self):
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        output = self.formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "hello"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert "timestamp" in data

    def test_includes_correlation_id(self):
        correlation_id.set("test-cid-123")
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        data = json.loads(self.formatter.format(record))
        assert data["correlation_id"] == "test-cid-123"

    def test_null_correlation_id_when_empty(self):
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        data = json.loads(self.formatter.format(record))
        assert data["correlation_id"] is None

    def test_includes_request_context(self):
        request_context.set({"user": "admin"})
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        data = json.loads(self.formatter.format(record))
        assert data["context"] == {"user": "admin"}

    def test_excludes_context_when_none(self):
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        data = json.loads(self.formatter.format(record))
        assert "context" not in data

    def test_includes_exception(self):
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            record = logging.LogRecord("test", logging.ERROR, "", 0, "err", (), sys.exc_info())

        data = json.loads(self.formatter.format(record))
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_includes_audit_extra_fields(self):
        record = logging.LogRecord("test", logging.INFO, "", 0, "tool_call", (), None)
        record.audit = True  # type: ignore[attr-defined]
        record.tool = "fine_tune"  # type: ignore[attr-defined]
        record.success = True  # type: ignore[attr-defined]
        record.duration_ms = 42  # type: ignore[attr-defined]
        data = json.loads(self.formatter.format(record))
        assert data["audit"] is True
        assert data["tool"] == "fine_tune"
        assert data["success"] is True
        assert data["duration_ms"] == 42

    def test_ignores_unknown_extra_keys(self):
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.random_field = "val"  # type: ignore[attr-defined]
        data = json.loads(self.formatter.format(record))
        assert "random_field" not in data


class TestConsoleFormatter:
    def setup_method(self):
        self.formatter = ConsoleFormatter()
        correlation_id.set("")

    def test_basic_format(self):
        record = logging.LogRecord("test.logger", logging.INFO, "", 0, "hello", (), None)
        output = self.formatter.format(record)
        assert "INFO" in output
        assert "test.logger" in output
        assert "hello" in output

    def test_includes_correlation_id(self):
        correlation_id.set("abcdef12-3456-7890-abcd-ef1234567890")
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        output = self.formatter.format(record)
        assert "[abcdef12]" in output

    def test_no_correlation_id(self):
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        output = self.formatter.format(record)
        stripped = output.replace("\033[32m", "").replace("\033[0m", "")
        assert "[" not in stripped

    def test_color_codes_present(self):
        record = logging.LogRecord("test", logging.ERROR, "", 0, "err", (), None)
        output = self.formatter.format(record)
        assert "\033[31m" in output
        assert "\033[0m" in output


class TestBufferingHandler:
    def test_stores_log_entries(self):
        handler = BufferingHandler()
        record = logging.LogRecord("test", logging.INFO, "", 0, "buffered", (), None)
        handler.emit(record)

        from kubeflow_mcp.core.logging import _log_buffer

        entries = list(_log_buffer)
        assert any(e["message"] == "buffered" for e in entries)


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "kubeflow_mcp"

    def test_json_format(self):
        logger = setup_logging(format="json")
        handler = next(h for h in logger.handlers if isinstance(h, logging.StreamHandler))
        assert isinstance(handler.formatter, StructuredFormatter)

    def test_console_format(self):
        logger = setup_logging(format="console")
        handler = next(h for h in logger.handlers if isinstance(h, logging.StreamHandler))
        assert isinstance(handler.formatter, ConsoleFormatter)

    def test_sets_log_level(self):
        logger = setup_logging(level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_adds_buffer_handler(self):
        logger = setup_logging()
        assert any(isinstance(h, BufferingHandler) for h in logger.handlers)

    def test_clears_previous_handlers(self):
        setup_logging()
        setup_logging()
        logger = logging.getLogger("kubeflow_mcp")
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) <= 2


class TestGetLogBuffer:
    def test_returns_list(self):
        result = get_log_buffer()
        assert isinstance(result, list)

    def test_captures_logged_messages(self):
        setup_logging(level="DEBUG", format="console")
        logger = logging.getLogger("kubeflow_mcp.buffer_test")
        logger.info("buffer_test_msg")

        buffer = get_log_buffer()
        assert any("buffer_test_msg" in e["message"] for e in buffer)
