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

"""Tests for trainer/api/platform.py — CRD inspection, runtime CRUD.

Covers input validation for platform admin tools.
K8s API interaction tests require mocking and are marked as TODOs.
"""

from __future__ import annotations

import pytest
from tests.common import FAILED, SUCCESS, VALIDATION_ERROR, TestCase, assert_test_case

from kubeflow_mcp.trainer.api.platform import create_runtime, patch_runtime


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="rejects invalid top-level patch keys",
            expected_status=FAILED,
            config={
                "name": "torchtune-llama",
                "patch": {"status": {"phase": "Ready"}},
                "confirmed": False,
            },
            expected_error_code=VALIDATION_ERROR,
        ),
        TestCase(
            name="preview accepts valid patch keys",
            expected_status=SUCCESS,
            config={
                "name": "torchtune-llama",
                "patch": {"spec": {"template": {}}},
                "confirmed": False,
            },
        ),
    ],
)
def test_patch_runtime_validation(test_case):
    assert_test_case(test_case, patch_runtime)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="rejects invalid top-level spec keys",
            expected_status=FAILED,
            config={
                "name": "torchtune-llama",
                "spec": {"replicas": 1},
                "confirmed": False,
            },
            expected_error_code=VALIDATION_ERROR,
        ),
        TestCase(
            name="preview accepts valid spec keys",
            expected_status=SUCCESS,
            config={
                "name": "torchtune-llama",
                "spec": {"template": {"spec": {"numNodes": 1}}},
                "confirmed": False,
            },
        ),
    ],
)
def test_create_runtime_validation(test_case):
    assert_test_case(test_case, create_runtime)


# TODO(test): test inspect_crd — lists all Trainer CRDs
# TODO(test): test inspect_crd(name) — returns CRD schema and conditions
# TODO(test): test inspect_crd — invalid CRD name
# TODO(test): test inspect_controller(view="logs") — returns controller logs
# TODO(test): test inspect_controller(view="events") — returns controller events
# TODO(test): test patch_runtime — confirmed=True applies strategic merge patch
# TODO(test): test patch_runtime — invalid runtime name rejected
# TODO(test): test create_runtime — confirmed=True creates runtime
# TODO(test): test create_runtime — name collision error
# TODO(test): test delete_runtime — preview lists dependent TrainJobs
# TODO(test): test delete_runtime — confirmed=True removes runtime
# TODO(test): test delete_runtime — non-admin persona rejected
