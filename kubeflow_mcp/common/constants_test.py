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

"""Tests for common/constants.py — error classification, phase maps."""

from kubeflow_mcp.common.constants import (
    ErrorCode,
    is_infrastructure_error,
)


class TestIsInfrastructureError:
    def test_kubernetes_error_is_infrastructure(self):
        result = {"error": "api failed", "error_code": ErrorCode.KUBERNETES_ERROR}
        assert is_infrastructure_error(result) is True

    def test_timeout_is_infrastructure(self):
        result = {"error": "timed out", "error_code": ErrorCode.TIMEOUT}
        assert is_infrastructure_error(result) is True

    def test_validation_error_is_not_infrastructure(self):
        result = {"error": "bad input", "error_code": ErrorCode.VALIDATION_ERROR}
        assert is_infrastructure_error(result) is False

    def test_none_result_is_not_infrastructure(self):
        assert is_infrastructure_error(None) is False

    def test_success_result_is_not_infrastructure(self):
        assert is_infrastructure_error({"data": {"ok": True}}) is False

    # TODO(test): test SDK_ERROR classification
    # TODO(test): test CIRCUIT_OPEN classification
    # TODO(test): test RATE_LIMITED classification
    # TODO(test): test TOOL_PHASES contains all expected phases
    # TODO(test): test TOOL_TO_PHASE covers all registered tools
    # TODO(test): test TOOL_NEXT_HINTS provides hints for training tools
