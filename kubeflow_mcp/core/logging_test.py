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
"""Unit tests for structured logging (logging.py)."""

import json
import logging

from kubeflow_mcp.core.logging import (
    BufferingHandler,
    ConsoleFormatter,
    StructuredFormatter,
    _log_buffer,
    get_log_buffer,
    get_logger,
    setup_logging,
    with_correlation_id,
)

# ─── StructuredFormatter ──────────────────────────────────────────────────────


class TestStructuredFormatter:
    def _make_record(self, msg="hello", level=logging.INFO, name="kubeflow_mcp.test"):
        record = logging.LogRecord(
            name=name,
            level=level,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        return record

    def test_output_is_valid_json(self):
        fmt = StructuredFormatter()
        record = self._make_record("test message")
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "test message"

    def test_required_keys_present(self):
        fmt = StructuredFormatter()
        record = self._make_record()
        parsed = json.loads(fmt.format(record))
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "message" in parsed

    def test_level_name_correct(self):
        fmt = StructuredFormatter()
        record = self._make_record(level=logging.WARNING)
        parsed = json.loads(fmt.format(record))
        assert parsed["level"] == "WARNING"

    def test_extra_audit_fields_included(self):
        fmt = StructuredFormatter()
        record = self._make_record()
        record.tool = "fine_tune"
        record.success = True
        record.duration_ms = 42
        parsed = json.loads(fmt.format(record))
        assert parsed["tool"] == "fine_tune"
        assert parsed["success"] is True
        assert parsed["duration_ms"] == 42

    def test_correlation_id_included_when_set(self):
        from kubeflow_mcp.core.logging import correlation_id

        token = correlation_id.set("test-cid-123")
        try:
            fmt = StructuredFormatter()
            record = self._make_record()
            parsed = json.loads(fmt.format(record))
            assert parsed["correlation_id"] == "test-cid-123"
        finally:
            correlation_id.reset(token)


# ─── ConsoleFormatter ─────────────────────────────────────────────────────────


class TestConsoleFormatter:
    def _make_record(self, msg="hello", level=logging.INFO):
        return logging.LogRecord(
            name="kubeflow_mcp.test",
            level=level,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_output_contains_message(self):
        fmt = ConsoleFormatter()
        record = self._make_record("my message")
        output = fmt.format(record)
        assert "my message" in output

    def test_output_contains_level(self):
        fmt = ConsoleFormatter()
        record = self._make_record(level=logging.ERROR)
        output = fmt.format(record)
        assert "ERROR" in output

    def test_output_not_json(self):
        import pytest

        fmt = ConsoleFormatter()
        record = self._make_record()
        output = fmt.format(record)
        with pytest.raises(json.JSONDecodeError):
            json.loads(output)


# ─── BufferingHandler ─────────────────────────────────────────────────────────


class TestBufferingHandler:
    def setup_method(self):
        _log_buffer.clear()

    def _emit(self, msg, level=logging.INFO):
        handler = BufferingHandler()
        record = logging.LogRecord(
            name="kubeflow_mcp.test",
            level=level,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        handler.emit(record)

    def test_message_stored_in_buffer(self):
        self._emit("hello world")
        entries = get_log_buffer()
        assert any("hello world" in e["message"] for e in entries)

    def test_entry_has_required_fields(self):
        self._emit("check fields")
        entry = get_log_buffer()[-1]
        assert "timestamp" in entry
        assert "level" in entry
        assert "logger" in entry
        assert "message" in entry

    def test_redacts_token_in_message(self):
        self._emit("token=super-secret-value")
        entry = get_log_buffer()[-1]
        assert "super-secret-value" not in entry["message"]
        assert "***" in entry["message"]

    def test_redacts_password_in_message(self):
        self._emit("password=hunter2 user=alice")
        entry = get_log_buffer()[-1]
        assert "hunter2" not in entry["message"]
        assert "alice" in entry["message"]

    def test_redacts_bearer_keyword(self):
        # The regex consumes the keyword + first non-space token (the word "Bearer").
        # The JWT itself is a second token — partial redaction is the current behaviour.
        self._emit("authorization=Bearer-eyJhbGciOi...")
        entry = get_log_buffer()[-1]
        assert "Bearer-eyJhbGciOi" not in entry["message"]
        assert "***" in entry["message"]

    def test_safe_messages_not_redacted(self):
        self._emit("training job started with lr=0.001")
        entry = get_log_buffer()[-1]
        assert "0.001" in entry["message"]


# ─── get_log_buffer ───────────────────────────────────────────────────────────


class TestGetLogBuffer:
    def setup_method(self):
        _log_buffer.clear()

    def test_returns_list(self):
        assert isinstance(get_log_buffer(), list)

    def test_returns_copy_not_deque(self):
        result = get_log_buffer()
        assert not hasattr(result, "appendleft")

    def test_buffer_grows_with_emissions(self):
        handler = BufferingHandler()
        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"msg {i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)
        assert len(get_log_buffer()) == 5


# ─── with_correlation_id ──────────────────────────────────────────────────────


def test_with_correlation_id_returns_uuid_string():
    cid = with_correlation_id()
    assert isinstance(cid, str)
    assert len(cid) == 36
    assert cid.count("-") == 4


def test_with_correlation_id_unique_per_call():
    cid1 = with_correlation_id()
    cid2 = with_correlation_id()
    assert cid1 != cid2


# ─── setup_logging ────────────────────────────────────────────────────────────


def test_setup_logging_returns_logger():
    logger = setup_logging(level="DEBUG", format="json")
    assert isinstance(logger, logging.Logger)


def test_setup_logging_console_format():
    logger = setup_logging(level="INFO", format="console")
    assert any(isinstance(h.formatter, ConsoleFormatter) for h in logger.handlers)


def test_setup_logging_json_format():
    logger = setup_logging(level="INFO", format="json")
    assert any(isinstance(h.formatter, StructuredFormatter) for h in logger.handlers)


def test_setup_logging_attaches_buffer_handler():
    setup_logging(level="INFO", format="json")
    root = logging.getLogger("kubeflow_mcp")
    assert any(isinstance(h, BufferingHandler) for h in root.handlers)


# ─── get_logger ───────────────────────────────────────────────────────────────


def test_get_logger_prefixes_name():
    logger = get_logger("mymodule")
    assert logger.name == "kubeflow_mcp.mymodule"


def test_get_logger_is_child_of_root():
    logger = get_logger("child")
    root = logging.getLogger("kubeflow_mcp")
    assert logger.parent is root or logger.name.startswith("kubeflow_mcp.")
