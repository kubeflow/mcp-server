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

"""Unit tests for discovery tool input validation."""

from unittest.mock import patch

import pytest

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.trainer.api.discovery import (
    get_runtime,
    get_training_job,
    list_training_jobs,
)


@pytest.mark.parametrize(
    ("tool", "kwargs", "client_path"),
    [
        (
            get_training_job,
            {"name": "INVALID_NAME"},
            "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        ),
        (
            get_runtime,
            {"name": "INVALID_NAME"},
            "kubeflow_mcp.trainer.api.discovery.get_trainer_client",
        ),
        (
            list_training_jobs,
            {"runtime": "INVALID_NAME"},
            "kubeflow_mcp.trainer.api.discovery.get_trainer_client_for_namespace",
        ),
    ],
)
def test_rejects_invalid_resource_name_before_calling_sdk(tool, kwargs, client_path):
    with patch(client_path) as mock_client:
        result = tool(**kwargs)

    assert result["success"] is False
    assert result["error_code"] == ErrorCode.VALIDATION_ERROR
    mock_client.assert_not_called()
