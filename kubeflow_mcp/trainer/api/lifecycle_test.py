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

"""Tests for trainer/api/lifecycle.py — delete, suspend, resume.

Covers input validation. K8s API interaction tests require mocking and are
marked as TODOs.
"""

from __future__ import annotations

import pytest
from tests.common import FAILED, VALIDATION_ERROR, TestCase, assert_test_case

from kubeflow_mcp.trainer.api.lifecycle import delete_training_job, update_training_job


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="invalid name rejected",
            expected_status=FAILED,
            config={"name": "INVALID"},
            expected_error_code=VALIDATION_ERROR,
        ),
        TestCase(
            name="empty name rejected",
            expected_status=FAILED,
            config={"name": ""},
            expected_error_code=VALIDATION_ERROR,
        ),
    ],
)
def test_delete_training_job_validation(test_case):
    assert_test_case(test_case, delete_training_job)


# TODO(test): test preview (confirmed=False) returns job details
# TODO(test): test confirmed=True with mock SDK deletes job
# TODO(test): test not found returns RESOURCE_NOT_FOUND
# TODO(test): test namespace policy enforcement
# TODO(test): test non-admin persona cannot delete non-MCP jobs


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="invalid name rejected",
            expected_status=FAILED,
            config={"name": "INVALID", "action": "suspend"},
            expected_error_code=VALIDATION_ERROR,
        ),
        TestCase(
            name="invalid action rejected",
            expected_status=FAILED,
            config={"name": "valid-job", "action": "restart"},
            expected_error_code=VALIDATION_ERROR,
        ),
    ],
)
def test_update_training_job_validation(test_case):
    assert_test_case(test_case, update_training_job)


# TODO(test): test suspend action with mock SDK
# TODO(test): test resume action with mock SDK
# TODO(test): test not found returns RESOURCE_NOT_FOUND
# TODO(test): test namespace policy enforcement
# TODO(test): test non-admin persona cannot update non-MCP jobs
