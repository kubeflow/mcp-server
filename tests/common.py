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

"""Shared test utilities and types for kubeflow-mcp tests."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

SUCCESS = "success"
FAILED = "failed"
PREVIEW = "preview"
DEFAULT_NAMESPACE = "default"

# Common tool error codes (mirrors kubeflow_mcp.common.constants.ErrorCode)
VALIDATION_ERROR = "VALIDATION_ERROR"
RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
PERMISSION_DENIED = "PERMISSION_DENIED"
KUBERNETES_ERROR = "KUBERNETES_ERROR"
SDK_ERROR = "SDK_ERROR"
CIRCUIT_OPEN = "CIRCUIT_OPEN"
RATE_LIMITED = "RATE_LIMITED"
TIMEOUT = "TIMEOUT"


@dataclass
class TestCase:
    """Parametrized test case for table-driven tests.

    Usage::

        @pytest.mark.parametrize("test_case", [
            TestCase(
                name="valid name",
                expected_status=SUCCESS,
                config={"name": "train-gemma"},
            ),
            TestCase(
                name="empty name rejected",
                expected_status=FAILED,
                config={"name": ""},
                expected_error_code=VALIDATION_ERROR,
            ),
        ])
        def test_validate_name(test_case):
            result = validate_k8s_name(test_case.config["name"])
            if test_case.expected_status == SUCCESS:
                assert result is None
            else:
                assert result is not None
    """

    name: str
    expected_status: str = SUCCESS
    config: dict[str, Any] = field(default_factory=dict)
    expected_output: Any | None = None
    expected_error: type[Exception] | None = None
    expected_error_code: str | None = None

    # Prevent pytest from collecting this dataclass as a test.
    __test__ = False

    def __repr__(self) -> str:
        return self.name


def assert_test_case(
    test_case: TestCase,
    fn: Callable[..., Any],
    **extra_kwargs: Any,
) -> Any:
    """Run *fn* with test_case.config and assert against expectations.

    Supports tool dict responses (``success`` / ``status`` keys) and validator
    functions that return ``None`` on success or a ``ToolError`` on failure.
    """
    try:
        result = fn(**test_case.config, **extra_kwargs)
    except Exception as e:
        assert test_case.expected_error is not None, f"Unexpected exception: {e}"
        assert isinstance(e, test_case.expected_error)
        return None

    assert test_case.expected_error is None, (
        f"Expected {test_case.expected_error.__name__}, got result: {result!r}"
    )

    if isinstance(result, dict):
        if test_case.expected_status == SUCCESS:
            assert result.get("success") is True, f"Expected success response, got: {result!r}"
        elif test_case.expected_status == FAILED:
            assert result.get("success") is False
            if test_case.expected_error_code:
                assert result.get("error_code") == test_case.expected_error_code
        elif test_case.expected_status == PREVIEW:
            assert result.get("status") == "preview"
    elif result is None:
        assert test_case.expected_status == SUCCESS
    else:
        assert test_case.expected_status == FAILED

    if test_case.expected_output is not None:
        _assert_expected_output(result, test_case.expected_output)

    return result


def _assert_expected_output(result: Any, expected_output: Any) -> None:
    if not isinstance(expected_output, dict):
        assert result == expected_output
        return

    for key, value in expected_output.items():
        assert (
            result.get(key) == value
            or result.get("config", {}).get(key) == value
            or result.get("data", {}).get(key) == value
        ), f"Expected {key}={value!r} in result/config/data, got: {result!r}"
