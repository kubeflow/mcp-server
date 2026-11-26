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

"""Unit tests for structured logging redaction."""

import io
import json
import logging
import re
import sys
import uuid

import pytest

from kubeflow_mcp.core.logging import (
    BufferingHandler,
    ConsoleFormatter,
    StructuredFormatter,
    _redact_dict,
    _redact_text,
    correlation_id,
    get_log_buffer,
    get_logger,
    request_context,
    setup_logging,
    with_correlation_id,
)

_JWT = "eyJhbGciOiJIUzI1NiJ9.SECRETPAYLOAD.sig"


class TestRedactDict:
    """Tests for the _redact_dict helper."""

    def test_redacts_top_level_sensitive_key(self):
        result = _redact_dict({"password": "hunter2"})
        assert result == {"password": "***"}

    def test_redacts_nested_dict(self):
        result = _redact_dict({"user": {"token": "abc123"}})
        assert result == {"user": {"token": "***"}}

    def test_redacts_dict_inside_list(self):
        result = _redact_dict({"steps": [{"password": "hunter2"}, {"name": "ok"}]})
        assert result == {"steps": [{"password": "***"}, {"name": "ok"}]}

    def test_redacts_non_string_sensitive_values(self):
        # Sensitive keys should be redacted regardless of value type.
        result = _redact_dict({"token": 12345, "secret": None})
        assert result == {"token": "***", "secret": "***"}

    def test_redacts_pattern_inside_non_sensitive_key_value(self):
        # A sensitive-looking substring embedded in an otherwise ordinary
        # string value should still be caught by the regex pass. Only the
        # matched portion is replaced; surrounding text is preserved.
        result = _redact_dict({"note": "password=hunter2 was used"})
        assert result["note"] == "password=*** was used"
        assert "hunter2" not in result["note"]

    def test_leaves_non_sensitive_values_untouched(self):
        result = _redact_dict({"user_id": "abc-123", "count": 5, "ok": True})
        assert result == {"user_id": "abc-123", "count": 5, "ok": True}

    def test_case_insensitive_key_match(self):
        result = _redact_dict({"Password": "hunter2", "API_KEY": "xyz"})
        assert result == {"Password": "***", "API_KEY": "***"}

    def test_substring_key_match(self):
        # auth_token matches via the "_token" suffix rule; client_secret
        # matches via the "secret" substring rule in mask_sensitive_data.
        result = _redact_dict({"auth_token": "abc", "client_secret": "xyz"})
        assert result == {"auth_token": "***", "client_secret": "***"}

    def test_tokenizer_key_not_falsely_redacted(self):
        # mask_sensitive_data deliberately does not treat "tokenizer" as
        # sensitive despite containing "token" as a substring — this was
        # the false positive with the old local key-matching logic.
        result = _redact_dict({"tokenizer": "bert-base-uncased"})
        assert result == {"tokenizer": "bert-base-uncased"}

    def test_glued_token_key_redacted(self):
        # No underscore before "token", but the suffix still marks these as credentials.
        result = _redact_dict({"accessToken": "abc", "authtoken": "xyz"})
        assert result == {"accessToken": "***", "authtoken": "***"}

    def test_safe_keys_not_redacted(self):
        result = _redact_dict({"public_key": "ssh-rsa AAAA...", "key_name": "prod"})
        assert result["public_key"] == "ssh-rsa AAAA..."
        assert result["key_name"] == "prod"

    def test_exact_sensitive_key_redacted(self):
        result = _redact_dict({"access_token": "abc", "secret_access_key": "xyz"})
        assert result == {"access_token": "***", "secret_access_key": "***"}

    def test_non_dict_non_list_input_returned_unchanged(self):
        assert _redact_dict("just a string") == "just a string"
        assert _redact_dict(42) == 42
        assert _redact_dict(None) is None

    def test_empty_dict_and_list(self):
        assert _redact_dict({}) == {}
        assert _redact_dict([]) == []

    def test_deeply_nested_mixed_structure(self):
        payload = {
            "request": {
                "headers": {"authorization": "Bearer abc.def.ghi"},
                "items": [
                    {"credential": "xyz"},
                    {"safe": "value"},
                ],
            }
        }
        result = _redact_dict(payload)
        assert result["request"]["headers"]["authorization"] == "***"
        assert result["request"]["items"][0]["credential"] == "***"
        assert result["request"]["items"][1]["safe"] == "value"


class TestStructuredFormatterRedaction:
    """Integration-style tests: redaction applied via the actual formatter."""

    @pytest.fixture(autouse=True)
    def _reset_context(self):
        token = request_context.set(None)
        yield
        request_context.reset(token)

    def _make_record(self, **extra):
        record = logging.LogRecord(
            name="kubeflow_mcp.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        for key, value in extra.items():
            setattr(record, key, value)
        return record

    def test_context_is_redacted_in_output(self):
        request_context.set({"password": "hunter2", "user": "alice"})
        formatter = StructuredFormatter()
        record = self._make_record()

        output = json.loads(formatter.format(record))

        assert output["context"]["password"] == "***"
        assert output["context"]["user"] == "alice"

    def test_parameters_extra_is_redacted_in_output(self):
        formatter = StructuredFormatter()
        record = self._make_record(parameters={"api_key": "sk-abc123", "limit": 10})

        output = json.loads(formatter.format(record))

        assert output["parameters"]["api_key"] == "***"
        assert output["parameters"]["limit"] == 10

    def test_non_dict_extra_passes_through_unmodified(self):
        formatter = StructuredFormatter()
        record = self._make_record(success=True, duration_ms=42)

        output = json.loads(formatter.format(record))

        assert output["success"] is True
        assert output["duration_ms"] == 42

    def test_no_context_omits_context_key(self):
        formatter = StructuredFormatter()
        record = self._make_record()

        output = json.loads(formatter.format(record))

        assert "context" not in output


class TestRedactText:
    """The one helper every log path shares."""

    @pytest.mark.parametrize(
        ("text", "leaked"),
        [
            (f"calling API with Authorization: Bearer {_JWT}", _JWT),
            (f"header was authorization: Basic {_JWT}", _JWT),
            ("kubeconfig has token: abc123def", "abc123def"),
            ("auth failed for credential=hunter2", "hunter2"),
            ("password=hunter2 was used", "hunter2"),
            ("upload with secret_access_key=AKIA123", "AKIA123"),
            ("loaded private_key=pk-live-123", "pk-live-123"),
            ("client api_key=sk-abc123", "sk-abc123"),
            ("retry s3_secret_access_key: s3-abc999", "s3-abc999"),
            ("refresh with accessToken=at-abc999", "at-abc999"),
            ("login failed user: password=hunter2", "hunter2"),
            ('{"token": "my-json-token-123"}', "my-json-token-123"),
            ("{'token': 'my-dict-token-456'}", "my-dict-token-456"),
            ('{"password": "secret_pass_789"}', "secret_pass_789"),
            ("{'password': 'dict_pass_abc'}", "dict_pass_abc"),
            ('{"api_key": "sk-live-999"}', "sk-live-999"),
            ('{"client_secret": "cs-test-000"}', "cs-test-000"),
            ('{"authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.SECRETPAYLOAD.sig"}', _JWT),
            ('{"token": 12345}', "12345"),
            (r'{"password": "secret\"with\"quotes"}', "secret"),
            (r'{"password": "secret\"with\"quotes"}', "quotes"),
            (r"{'password': 'secret\'with\'quotes'}", "quotes"),
            ("password=,dummy-value", "dummy-value"),
            ("password=abc,def", "def"),
            ("password=;dummy-value", "dummy-value"),
            ("password=abc;def", "def"),
            ("password=abc}def", "def"),
            ("password=abc]def", "def"),
        ],
    )
    def test_credential_is_removed(self, text, leaked):
        result = _redact_text(text)
        assert leaked not in result
        assert "***" in result

    def test_bare_credential_with_leading_comma_redacted(self):
        text = "password=,dummy-value"
        assert _redact_text(text) == "password=***"
        assert "dummy-value" not in _redact_text(text)

    def test_bare_credential_with_embedded_comma_redacted(self):
        text = "password=abc,def"
        assert _redact_text(text) == "password=***"
        assert "abc" not in _redact_text(text)
        assert "def" not in _redact_text(text)

    def test_bare_credential_with_leading_and_embedded_semicolon_redacted(self):
        assert _redact_text("password=;dummy-value") == "password=***"
        assert "dummy-value" not in _redact_text("password=;dummy-value")
        assert _redact_text("password=abc;def") == "password=***"
        assert "abc" not in _redact_text("password=abc;def")
        assert "def" not in _redact_text("password=abc;def")

    def test_bare_credential_with_embedded_brackets_redacted(self):
        assert _redact_text("password=abc}def") == "password=***"
        assert "abc" not in _redact_text("password=abc}def")
        assert "def" not in _redact_text("password=abc}def")
        assert _redact_text("password=abc]def") == "password=***"
        assert "abc" not in _redact_text("password=abc]def")
        assert "def" not in _redact_text("password=abc]def")

    def test_bare_credential_preserves_trailing_comma_separator(self):
        text = "token: abc, next=1"
        assert _redact_text(text) == "token: ***, next=1"

    def test_bare_credential_preserves_trailing_separators_and_delimiters(self):
        assert _redact_text("token: abc; next=1") == "token: ***; next=1"
        assert _redact_text("token: abc;") == "token: ***;"
        assert _redact_text('{"token": abc}') == '{"token": ***}'
        assert _redact_text("[token: abc]") == "[token: ***]"
        assert _redact_text('{"data": {"token": abc}}') == '{"data": {"token": ***}}'

    def test_escaped_quotes_inside_json_value_redacted(self):
        text = r'{"password": "secret\"with\"quotes", "user": "alice"}'
        expected = '{"password": "***", "user": "alice"}'
        assert _redact_text(text) == expected
        assert "secret" not in _redact_text(text)
        assert "quotes" not in _redact_text(text)

    def test_escaped_quotes_inside_single_quoted_dict_redacted(self):
        text = r"{'password': 'secret\'with\'quotes', 'user': 'alice'}"
        expected = "{'password': '***', 'user': 'alice'}"
        assert _redact_text(text) == expected
        assert "secret" not in _redact_text(text)
        assert "quotes" not in _redact_text(text)

        escaped_dict = r"{\'password\': \'alpha beta\'}"
        result = _redact_text(escaped_dict)
        assert result == r"{\'password\': '***'}"
        assert "alpha" not in result
        assert "beta" not in result

    def test_quoted_json_structure_preserved(self):
        text = '{"user": "alice", "token": "secret123", "role": "admin"}'
        expected = '{"user": "alice", "token": "***", "role": "admin"}'
        assert _redact_text(text) == expected

    def test_single_quoted_dict_structure_preserved(self):
        text = "{'user': 'alice', 'password': 'hunter2', 'role': 'admin'}"
        expected = "{'user': 'alice', 'password': '***', 'role': 'admin'}"
        assert _redact_text(text) == expected

    def test_plain_message_untouched(self):
        assert _redact_text("tool_call completed in 12ms") == "tool_call completed in 12ms"

    def test_word_ending_in_keyword_not_matched(self):
        # image_pull_secrets is a parameter name, not a credential assignment.
        assert _redact_text("image_pull_secrets not set") == "image_pull_secrets not set"

    def test_prose_keyword_not_matched(self):
        # The cli.py auth warning names flags and env vars, not credential values.
        warning = (
            "Set --auth-token or KUBEFLOW_MCP_AUTH_TOKEN for bearer auth, "
            "or KUBEFLOW_MCP_JWKS_URI with KUBEFLOW_MCP_JWT_ISSUER and "
            "KUBEFLOW_MCP_JWT_AUDIENCE for JWT verification."
        )
        assert _redact_text(warning) == warning

    @pytest.mark.parametrize(
        "text",
        [
            "public_key=ssh-rsa AAAA",
            "key_name=prod",
            "keyword=llama",
            "key_format=pem",
            "tokenizer: bert-base-uncased",
        ],
    )
    def test_safe_key_not_redacted(self, text):
        assert _redact_text(text) == text


class TestLogPathsRedactConsistently:
    """Buffer, stderr JSON, and console must agree on the same line."""

    _LINE = f"calling API with Authorization: Bearer {_JWT}"

    def _record(self, msg, exc_info=None):
        return logging.LogRecord(
            name="kubeflow_mcp.test",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=(),
            exc_info=exc_info,
        )

    def test_buffer_redacts_bearer_credential(self):
        BufferingHandler().emit(self._record(self._LINE))
        assert _JWT not in get_log_buffer()[-1]["message"]

    def test_structured_formatter_redacts_message(self):
        output = json.loads(StructuredFormatter().format(self._record(self._LINE)))
        assert _JWT not in output["message"]

    def test_console_formatter_redacts_message(self):
        assert _JWT not in ConsoleFormatter().format(self._record(self._LINE))

    def test_structured_formatter_redacts_exception_text(self):
        try:
            raise RuntimeError(f"boom: token={_JWT}")
        except RuntimeError:
            record = self._record("tool_call_failed", exc_info=sys.exc_info())

        output = json.loads(StructuredFormatter().format(record))
        assert _JWT not in output["exception"]

    def test_console_formatter_redacts_exception_text(self):
        try:
            raise RuntimeError(f"boom: token={_JWT}")
        except RuntimeError:
            record = self._record("tool_call_failed", exc_info=sys.exc_info())

        output = ConsoleFormatter().format(record)
        assert _JWT not in output
        assert "token=***" in output
        assert "RuntimeError: boom" in output
        assert "Traceback (most recent call last)" in output

    def test_all_paths_produce_the_same_message(self):
        record = self._record(self._LINE)
        BufferingHandler().emit(record)

        buffered = get_log_buffer()[-1]["message"]
        structured = json.loads(StructuredFormatter().format(record))["message"]

        assert buffered == "calling API with Authorization: ***"
        assert buffered == structured
        assert buffered in ConsoleFormatter().format(record)


class TestCorrelationId:
    @pytest.fixture(autouse=True)
    def _reset_correlation_id(self):
        token = correlation_id.set("")
        try:
            yield
        finally:
            correlation_id.reset(token)

    def test_with_correlation_id_sets_and_returns_uuid(self):
        cid = with_correlation_id()
        assert isinstance(cid, str)
        assert uuid.UUID(cid).version == 4
        assert correlation_id.get() == cid

    def test_structured_formatter_includes_correlation_id(self):
        cid = with_correlation_id()
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", (), None)
        output = json.loads(StructuredFormatter().format(record))
        assert output["correlation_id"] == cid

    def test_structured_formatter_correlation_id_none_when_empty(self):
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", (), None)
        output = json.loads(StructuredFormatter().format(record))
        assert output["correlation_id"] is None

    def test_console_formatter_includes_correlation_id_prefix(self):
        cid = with_correlation_id()
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", (), None)
        formatted = ConsoleFormatter().format(record)
        assert f"[{cid[:8]}]" in formatted

    def test_console_formatter_omits_correlation_id_when_empty(self):
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", (), None)
        formatted = ConsoleFormatter().format(record)
        assert not re.search(r"\[[0-9a-f]{8}\]", formatted)
        assert f"{ConsoleFormatter.RESET} test: hello" in formatted


class TestRedactTextEdgeCases:
    def test_sensitive_key_with_trailing_empty_value(self):
        assert _redact_text("password=") == "password="
        assert _redact_text("secret: ") == "secret: "


class TestGetLogger:
    def test_get_logger_prefixes_name(self):
        logger = get_logger("my_component")
        assert logger.name == "kubeflow_mcp.my_component"


class TestConsoleFormatterColors:
    @pytest.mark.parametrize(
        ("level", "color"),
        [
            (logging.DEBUG, ConsoleFormatter.COLORS["DEBUG"]),
            (logging.INFO, ConsoleFormatter.COLORS["INFO"]),
            (logging.WARNING, ConsoleFormatter.COLORS["WARNING"]),
            (logging.ERROR, ConsoleFormatter.COLORS["ERROR"]),
            (logging.CRITICAL, ConsoleFormatter.COLORS["CRITICAL"]),
        ],
    )
    def test_console_formatter_colors(self, level, color):
        record = logging.LogRecord("test", level, __file__, 1, "test message", (), None)
        formatted = ConsoleFormatter().format(record)
        assert color in formatted
        assert ConsoleFormatter.RESET in formatted

    def test_console_formatter_unknown_level(self):
        record = logging.LogRecord("test", 99, __file__, 1, "custom level", (), None)
        record.levelname = "CUSTOM"
        formatted = ConsoleFormatter().format(record)
        assert formatted.startswith(ConsoleFormatter.RESET)


class TestConsoleFormatterTraceback:
    """Tests for ConsoleFormatter exception and stack info handling."""

    def test_renders_and_redacts_traceback(self):
        formatter = ConsoleFormatter()
        try:
            raise ValueError(f"db crash password=hunter2 token={_JWT}")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="operation failed",
                args=(),
                exc_info=sys.exc_info(),
            )

        formatted = formatter.format(record)
        assert "operation failed" in formatted
        assert "Traceback (most recent call last)" in formatted
        assert "ValueError: db crash" in formatted
        assert "password=***" in formatted
        assert "hunter2" not in formatted
        assert _JWT not in formatted
        assert "token=***" in formatted

    def test_renders_and_redacts_stack_info(self):
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="warning check",
            args=(),
            exc_info=None,
        )
        record.stack_info = (
            f"Stack (most recent call last):\n  File 'test.py', line 1\n    secret_key={_JWT}"
        )

        formatted = formatter.format(record)
        assert "warning check" in formatted
        assert "Stack (most recent call last)" in formatted
        assert _JWT not in formatted
        assert "secret_key=***" in formatted

    def test_single_line_when_no_exception_or_stack(self):
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="simple message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "\n" not in formatted

    def test_renders_and_redacts_json_exception_payload(self):
        formatter = ConsoleFormatter()
        try:
            raise RuntimeError('{"token": "super_secret_token", "error": "unauthorized"}')
        except RuntimeError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="api error",
                args=(),
                exc_info=sys.exc_info(),
            )

        formatted = formatter.format(record)
        assert "super_secret_token" not in formatted
        assert '{"token": "***", "error": "unauthorized"}' in formatted
        assert "Traceback (most recent call last)" in formatted

    def test_renders_and_redacts_quoted_dict_in_stack_info(self):
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="stack error",
            args=(),
            exc_info=None,
        )
        record.stack_info = (
            "Stack (most recent call last):\n"
            "  File 'handler.py', line 12\n"
            "    headers = {'api_key': 'secret-key-123'}"
        )

        formatted = formatter.format(record)
        assert "secret-key-123" not in formatted
        assert "{'api_key': '***'}" in formatted
        assert "Stack (most recent call last)" in formatted

    def test_renders_and_redacts_exception_with_escaped_quotes(self):
        formatter = ConsoleFormatter()
        try:
            raise RuntimeError(r'{"password": "leak_xyz\"nested_abc\"val", "status": "failed"}')
        except RuntimeError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="auth error",
                args=(),
                exc_info=sys.exc_info(),
            )

        formatted = formatter.format(record)
        assert "leak_xyz" not in formatted
        assert "nested_abc" not in formatted
        assert '{"password": "***", "status": "failed"}' in formatted
        assert "Traceback (most recent call last)" in formatted

    def test_renders_and_redacts_bare_credential_with_comma_in_traceback(self):
        formatter = ConsoleFormatter()
        try:
            raise RuntimeError("failed with password=lead,trail_secret in job")
        except RuntimeError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="auth error",
                args=(),
                exc_info=sys.exc_info(),
            )

        formatted = formatter.format(record)
        assert "lead" not in formatted
        assert "trail_secret" not in formatted
        assert "password=***" in formatted
        assert "Traceback (most recent call last)" in formatted


class TestSetupLogging:
    """Tests for setup_logging handler and level configuration."""

    @pytest.fixture(autouse=True)
    def _restore_logger(self):
        root = logging.getLogger("kubeflow_mcp")
        handlers, level = root.handlers[:], root.level
        yield
        root.handlers[:] = handlers
        root.setLevel(level)

    @staticmethod
    def _stream_formatter(root: logging.Logger) -> logging.Formatter:
        stream = [h for h in root.handlers if not isinstance(h, BufferingHandler)]
        assert len(stream) == 1
        return stream[0].formatter

    @pytest.mark.parametrize(
        ("fmt", "formatter_cls"),
        [("json", StructuredFormatter), ("console", ConsoleFormatter)],
    )
    def test_uses_requested_format(self, fmt, formatter_cls):
        root = setup_logging(format=fmt)
        assert isinstance(self._stream_formatter(root), formatter_cls)

    @pytest.mark.parametrize(
        ("is_tty", "formatter_cls"),
        [(True, ConsoleFormatter), (False, StructuredFormatter)],
    )
    def test_auto_detects_format_from_terminal(self, monkeypatch, is_tty, formatter_cls):
        class _Stream(io.StringIO):
            def isatty(self) -> bool:
                return is_tty

        monkeypatch.setattr(sys, "stderr", _Stream())
        root = setup_logging()
        assert isinstance(self._stream_formatter(root), formatter_cls)

    def test_sets_log_level(self):
        root = setup_logging(level="debug", format="json")
        assert root.level == logging.DEBUG

    @pytest.mark.parametrize(
        ("level", "expected_level"),
        [
            ("WARN", logging.WARNING),
            ("warn", logging.WARNING),
            ("FATAL", logging.CRITICAL),
            ("fatal", logging.CRITICAL),
        ],
    )
    def test_normalizes_legacy_log_level_aliases(self, level, expected_level):
        root = setup_logging(level=level, format="json")
        assert root.level == expected_level

    @pytest.mark.parametrize("level", ["TRACE", "NOT_A_LEVEL", ""])
    def test_rejects_invalid_log_level(self, level):
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(level=level, format="json")

    def test_calling_twice_does_not_duplicate_handlers(self):
        setup_logging(format="json")
        root = setup_logging(format="json")

        assert len(root.handlers) == 2
        assert sum(isinstance(h, BufferingHandler) for h in root.handlers) == 1
