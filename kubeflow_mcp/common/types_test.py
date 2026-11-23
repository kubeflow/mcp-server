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

"""Tests for common/types.py — K8s error detection, exception details, response models."""

from __future__ import annotations

import logging
from unittest.mock import patch

from kubeflow_mcp.common.types import (
    PreviewResponse,
    ToolError,
    ToolResponse,
    exception_details,
    has_k8s_status,
    is_k8s_forbidden,
    is_k8s_not_found,
)


class TestIsK8sNotFound:
    def test_returns_true_for_api_404(self):
        try:
            from kubernetes.client.exceptions import ApiException

            exc = ApiException(status=404)
            assert is_k8s_not_found(exc) is True
        except ImportError:
            pass

    def test_returns_false_for_api_403(self):
        try:
            from kubernetes.client.exceptions import ApiException

            exc = ApiException(status=403)
            assert is_k8s_not_found(exc) is False
        except ImportError:
            pass

    def test_returns_false_for_generic_exception(self):
        assert is_k8s_not_found(RuntimeError("not found")) is False

    def test_returns_false_for_module_not_found(self):
        assert is_k8s_not_found(ModuleNotFoundError("not found")) is False

    def test_chained_cause_with_api_404(self):
        from kubernetes.client.exceptions import ApiException

        cause = ApiException(status=404)
        exc = RuntimeError("wrapper")
        exc.__cause__ = cause
        assert is_k8s_not_found(exc) is True

    def test_chained_context_with_api_404(self):
        from kubernetes.client.exceptions import ApiException

        context = ApiException(status=404)
        exc = RuntimeError("wrapper")
        exc.__context__ = context
        assert is_k8s_not_found(exc) is True


class TestIsK8sForbidden:
    """401/403 mean the caller lacks access; they must not be read as an outage."""

    def test_returns_true_for_api_403(self):
        from kubernetes.client.exceptions import ApiException

        assert is_k8s_forbidden(ApiException(status=403)) is True

    def test_returns_true_for_api_401(self):
        from kubernetes.client.exceptions import ApiException

        assert is_k8s_forbidden(ApiException(status=401)) is True

    def test_returns_false_for_api_404_and_500(self):
        from kubernetes.client.exceptions import ApiException

        assert is_k8s_forbidden(ApiException(status=404)) is False
        assert is_k8s_forbidden(ApiException(status=500)) is False

    def test_returns_false_for_text_that_mentions_forbidden(self):
        assert is_k8s_forbidden(RuntimeError("403 Forbidden")) is False

    def test_finds_403_wrapped_more_than_one_level_deep(self):
        """The SDK can wrap an ApiException twice; one level is not enough."""
        from kubernetes.client.exceptions import ApiException

        inner = RuntimeError("SDK wrapper")
        inner.__cause__ = ApiException(status=403)
        outer = RuntimeError("Failed to list SparkConnect sessions")
        outer.__context__ = inner
        assert is_k8s_forbidden(outer) is True

    def test_cyclic_chain_terminates(self):
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert is_k8s_forbidden(a) is False


class TestHasK8sStatus:
    def test_matches_any_of_several_statuses(self):
        from kubernetes.client.exceptions import ApiException

        assert has_k8s_status(ApiException(status=409), 403, 409) is True
        assert has_k8s_status(ApiException(status=500), 403, 409) is False


class TestExceptionDetails:
    def test_includes_exception_type_and_message(self):
        details = exception_details(RuntimeError("boom"))
        assert details["exception"] == "RuntimeError"
        assert details["message"] == "boom"

    def test_includes_cause_when_present(self):
        try:
            try:
                raise ValueError("inner")
            except ValueError:
                raise RuntimeError("outer") from ValueError("inner")
        except RuntimeError as e:
            details = exception_details(e)
            assert "cause" in details
            assert "ValueError" in details["cause"]

    def test_excludes_traceback_at_info_level(self):
        root = logging.getLogger("kubeflow_mcp")
        original = root.level
        root.setLevel(logging.INFO)
        try:
            details = exception_details(RuntimeError("boom"))
            assert "traceback" not in details
        finally:
            root.setLevel(original)

    def test_includes_traceback_at_debug_level(self):
        root = logging.getLogger("kubeflow_mcp")
        original = root.level
        root.setLevel(logging.DEBUG)
        try:
            try:
                raise RuntimeError("boom")
            except RuntimeError as e:
                details = exception_details(e)
                assert "traceback" in details
                assert "RuntimeError" in details["traceback"]
        finally:
            root.setLevel(original)

    @patch("kubeflow_mcp.common.types.traceback.format_exc", return_value="NoneType: None")
    def test_excludes_traceback_when_format_exc_returns_nonetype(self, _mock_fmt):
        root = logging.getLogger("kubeflow_mcp")
        original = root.level
        root.setLevel(logging.DEBUG)
        try:
            details = exception_details(RuntimeError("boom"))
            assert "traceback" not in details
        finally:
            root.setLevel(original)


class TestResponseModels:
    def test_tool_response_serialization(self):
        r = ToolResponse(data={"job": "abc"})
        d = r.model_dump()
        assert d["success"] is True
        assert d["data"]["job"] == "abc"

    def test_tool_error_serialization(self):
        e = ToolError(error="bad input", error_code="VALIDATION_ERROR")
        d = e.model_dump()
        assert d["success"] is False
        assert d["error"] == "bad input"

    def test_preview_response_has_status_preview(self):
        p = PreviewResponse(config={"model": "gemma"})
        d = p.model_dump()
        assert d["status"] == "preview"
        assert d["success"] is True

    def test_tool_error_with_hint(self):
        e = ToolError(
            error="job not found",
            error_code="RESOURCE_NOT_FOUND",
            hint="Use list_training_jobs to find available jobs",
        )
        d = e.model_dump()
        assert d["hint"] == "Use list_training_jobs to find available jobs"

    def test_tool_error_with_details(self):
        e = ToolError(
            error="sdk failure",
            error_code="SDK_ERROR",
            details={"exception": "RuntimeError", "message": "connection reset"},
        )
        d = e.model_dump()
        assert d["details"]["exception"] == "RuntimeError"
        assert d["details"]["message"] == "connection reset"
