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

Covers input validation and the MCP ownership gate. The remaining K8s API
interaction tests are marked as TODOs.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException
from tests.common import FAILED, VALIDATION_ERROR, TestCase, assert_test_case

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.trainer.api.lifecycle import delete_training_job, update_training_job

_UTILS = "kubeflow_mcp.common.utils"
_LIFECYCLE = "kubeflow_mcp.trainer.api.lifecycle"


@contextmanager
def cluster(ownership=mcp_utils.OWNERSHIP_MANAGED, persona="data-scientist"):
    """Patch namespace resolution, the ownership lookup and the K8s APIs."""
    api = MagicMock()
    client = MagicMock()
    with (
        patch(f"{_UTILS}.get_trainer_effective_namespace", return_value="default"),
        patch(f"{_UTILS}.get_trainer_custom_objects_api", return_value=api),
        patch(f"{_UTILS}.get_trainer_client_for_namespace", return_value=client),
        patch(f"{_UTILS}.get_trainer_ownership", return_value=ownership) as lookup,
        patch(f"{_LIFECYCLE}.get_effective_persona", return_value=persona),
        patch(f"{_LIFECYCLE}.check_namespace_allowed", return_value=None),
    ):
        yield api, client, lookup


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
# TODO(test): test namespace policy enforcement


# ─── MCP ownership enforcement ─────────────────────────────────────────────


@pytest.mark.parametrize("persona", ["data-scientist", "ml-engineer", "platform-admin"])
@pytest.mark.parametrize(
    ("tool", "call"),
    [
        ("delete", lambda: delete_training_job("ghost-job", confirmed=True)),
        ("update", lambda: update_training_job("ghost-job", "suspend")),
    ],
)
def test_missing_job_reports_not_found_for_every_persona(persona, tool, call):
    """A name that does not exist is a not-found, whoever asks.

    The ownership gate used to collapse "missing" into "not created by MCP",
    so a typo sent non-admins after RBAC that was never the problem, and the
    same job answered differently depending on the caller's persona.
    """
    with cluster(ownership=mcp_utils.OWNERSHIP_MISSING, persona=persona) as (api, client, _):
        # platform-admin skips the ownership lookup and reaches the cluster,
        # which returns its own 404.
        client.delete_job.side_effect = ApiException(status=404, reason="Not Found")
        api.patch_namespaced_custom_object.side_effect = ApiException(
            status=404, reason="Not Found"
        )
        result = call()

    assert result["success"] is False
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "not created by MCP" not in result["error"]


@pytest.mark.parametrize(
    ("tool", "call", "attr"),
    [
        ("delete", lambda: delete_training_job("external", confirmed=True), "delete_job"),
        ("update", lambda: update_training_job("external", "suspend"), "delete_job"),
    ],
)
def test_non_admin_blocked_from_mutating_external_job(tool, call, attr):
    with cluster(ownership=mcp_utils.OWNERSHIP_UNMANAGED) as (api, client, _):
        result = call()
    assert result["success"] is False
    assert result["error_code"] == VALIDATION_ERROR
    assert "was not created by MCP" in result["error"]
    getattr(client, attr).assert_not_called()
    api.patch_namespaced_custom_object.assert_not_called()


def test_non_admin_allowed_for_mcp_created_job():
    with cluster(ownership=mcp_utils.OWNERSHIP_MANAGED) as (_, client, _lookup):
        result = delete_training_job("ours", confirmed=True)
    assert result["success"] is True
    client.delete_job.assert_called_once()


def test_admin_bypasses_ownership_check_entirely():
    """platform-admin must not even perform the lookup."""
    with cluster(ownership=mcp_utils.OWNERSHIP_UNMANAGED, persona="platform-admin") as (
        _,
        client,
        lookup,
    ):
        result = delete_training_job("external", confirmed=True)
    assert result["success"] is True
    client.delete_job.assert_called_once()
    lookup.assert_not_called()


@pytest.mark.parametrize(
    ("tool", "call"),
    [
        ("delete", lambda: delete_training_job("ghost-job", confirmed=True)),
        ("update", lambda: update_training_job("ghost-job", "suspend")),
    ],
)
def test_missing_job_end_to_end_from_a_real_404(tool, call):
    """End to end through the real ownership helper, mocking only Kubernetes.

    The tests above stub ``get_trainer_ownership``, so they pin the gate's
    branching but cannot see how a genuine 404 is classified. This one closes
    that gap: it is the test that fails if the 404 is ever folded back into
    "unmanaged".
    """
    k8s = MagicMock()
    k8s.get_namespaced_custom_object.side_effect = ApiException(status=404, reason="Not Found")
    client = MagicMock()

    with (
        patch(f"{_UTILS}.get_trainer_effective_namespace", return_value="default"),
        patch(f"{_UTILS}.get_custom_objects_api", return_value=k8s),
        patch(f"{_UTILS}.get_trainer_custom_objects_api", return_value=k8s),
        patch(f"{_UTILS}.get_trainer_client_for_namespace", return_value=client),
        patch(f"{_LIFECYCLE}.get_effective_persona", return_value="data-scientist"),
        patch(f"{_LIFECYCLE}.check_namespace_allowed", return_value=None),
    ):
        result = call()

    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "not created by MCP" not in result["error"]
    client.delete_job.assert_not_called()


def test_ownership_lookup_failure_does_not_mutate():
    with cluster(ownership=None) as (_, client, _lookup):
        result = delete_training_job("unknown", confirmed=True)
    assert result["error_code"] == "SDK_ERROR"
    client.delete_job.assert_not_called()


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
