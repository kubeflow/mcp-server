# Copyright The Kubeflow Authors.
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

"""Shared input validation helpers for Spark tools."""

from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError


def validate_int_range(
    value: Any,
    field: str,
    minimum: int,
    maximum: int | None = None,
) -> ToolError | None:
    """Require *value* to be a real ``int`` (never ``bool``) within bounds."""
    bound = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
    if not isinstance(value, int) or isinstance(value, bool):
        return ToolError(
            error=f"{field} must be an integer {bound}, got {value!r}",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    if value < minimum or (maximum is not None and value > maximum):
        return ToolError(
            error=f"{field} must be {bound}, got {value}",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    return None
