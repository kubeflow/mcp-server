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

from kubeflow_mcp.common.types import (
    PreviewResponse,
    ToolError,
    ToolResponse,
    exception_details,
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

    # TODO(test): test chained exception (__cause__) with ApiException 404
    # TODO(test): test chained exception (__context__) with ApiException 404


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

    # TODO(test): test traceback included at DEBUG level
    # TODO(test): test traceback excluded when format_exc returns "NoneType: None"


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

    # TODO(test): test ToolError with hint field
    # TODO(test): test ToolError with details dict
