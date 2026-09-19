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

import json
import logging
import sys

import pytest

from kubeflow_mcp.core.logging import (
    BufferingHandler,
    ConsoleFormatter,
    StructuredFormatter,
    _redact_dict,
    _redact_text,
    get_log_buffer,
    request_context,
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
        ],
    )
    def test_credential_is_removed(self, text, leaked):
        result = _redact_text(text)
        assert leaked not in result
        assert "***" in result

    def test_plain_message_untouched(self):
        assert _redact_text("tool_call completed in 12ms") == "tool_call completed in 12ms"

    def test_word_ending_in_keyword_not_matched(self):
        # image_pull_secrets is a parameter name, not a credential assignment.
        assert _redact_text("image_pull_secrets not set") == "image_pull_secrets not set"

    def test_prose_keyword_not_matched(self):
        # The cli.py auth warning names flags and env vars, not credential values.
        warning = (
            "Set --auth-token or KUBEFLOW_MCP_AUTH_TOKEN for bearer auth, "
            "or KUBEFLOW_MCP_JWKS_URI for JWT verification."
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

    def test_all_paths_produce_the_same_message(self):
        record = self._record(self._LINE)
        BufferingHandler().emit(record)

        buffered = get_log_buffer()[-1]["message"]
        structured = json.loads(StructuredFormatter().format(record))["message"]

        assert buffered == "calling API with Authorization: ***"
        assert buffered == structured
        assert buffered in ConsoleFormatter().format(record)
