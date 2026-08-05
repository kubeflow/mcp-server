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

"""Tests for trainer/api/discovery.py — list/get training jobs and runtimes.

Covers input validation and status filter aliasing.
K8s API interaction tests require mocking the SDK and are marked as TODOs.
"""

from __future__ import annotations

from tests.common import RESOURCE_NOT_FOUND

from kubeflow_mcp.conftest import (
    NOT_FOUND_NAME,
    VALID_JOB_NAME,
    verify_tool_error,
    verify_tool_success,
)
from kubeflow_mcp.trainer.api.discovery import (
    _JOB_STATUS_FILTER_ALIASES,
    _trainjob_runtime_to_mcp,
    get_training_job,
)


class TestJobStatusFilterAliases:
    def test_succeeded_maps_to_complete(self):
        assert _JOB_STATUS_FILTER_ALIASES["Succeeded"] == "Complete"


class TestTrainjobRuntimeToMcp:
    def test_none_returns_none(self):
        assert _trainjob_runtime_to_mcp(None) is None

    def test_serializes_name(self):
        class FakeRuntime:
            name = "torchtune-llama"

        assert _trainjob_runtime_to_mcp(FakeRuntime()) == {"name": "torchtune-llama"}

    def test_empty_name_returns_none(self):
        class FakeRuntime:
            name = ""

        assert _trainjob_runtime_to_mcp(FakeRuntime()) is None


def test_get_training_job_returns_details(mock_trainer_client):
    result = get_training_job(name=VALID_JOB_NAME)
    data = verify_tool_success(result)
    assert data["name"] == VALID_JOB_NAME
    assert data["status"] == "Running"


def test_get_training_job_not_found(mock_trainer_client):
    result = get_training_job(name=NOT_FOUND_NAME)
    verify_tool_error(result, error_code=RESOURCE_NOT_FOUND)


# TODO(test): list_training_jobs — returns formatted job list with mock SDK
# TODO(test): list_training_jobs — status filter applies alias
# TODO(test): list_training_jobs — runtime filter
# TODO(test): list_training_jobs — namespace policy enforcement
# TODO(test): get_training_job — returns job details with mock SDK
# TODO(test): get_training_job — invalid name rejected
# TODO(test): list_runtimes — returns runtime list
# TODO(test): get_runtime — returns runtime details
# TODO(test): get_runtime — include_packages=True spawns pod
# TODO(test): get_runtime — not found returns RESOURCE_NOT_FOUND
