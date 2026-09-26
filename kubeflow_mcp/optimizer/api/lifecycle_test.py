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

"""Unit tests for optimizer lifecycle tools (confirmation, ownership, suspend/resume)."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.optimizer.api import lifecycle as lc

_UTILS = "kubeflow_mcp.common.utils"
_COMMON = "kubeflow_mcp.optimizer.api._common"


def _not_found():
    """The shape the SDK raises: RuntimeError chaining a 404 ApiException."""
    err = RuntimeError("Failed to get OptimizationJob: kubeflow/missing")
    err.__cause__ = ApiException(status=404, reason="Not Found")
    return err


@contextmanager
def cluster(api: MagicMock | None = None, persona="platform-admin", managed=True, client=None):
    """``managed`` accepts True/False/None, or an ownership string directly."""
    api = api or MagicMock()
    client = client or MagicMock()
    ownership = {
        True: mcp_utils.OWNERSHIP_MANAGED,
        False: mcp_utils.OWNERSHIP_UNMANAGED,
        None: None,
    }.get(managed, managed)
    with (
        patch(f"{_UTILS}.get_optimizer_effective_namespace", return_value="kubeflow"),
        patch(f"{_UTILS}.get_custom_objects_api", return_value=api),
        patch(f"{_UTILS}.get_optimizer_client_for_namespace", return_value=client),
        patch(f"{_UTILS}.get_optimizer_ownership", return_value=ownership),
        patch(f"{_COMMON}.get_effective_persona", return_value=persona),
    ):
        yield api, client


def _experiment(parallel=4, annotations=None):
    return {
        "metadata": {"annotations": annotations or {}},
        "spec": {"parallelTrialCount": parallel},
    }


# ─── delete_experiment ─────────────────────────────────────────────────────


def test_delete_preview_does_not_delete():
    with cluster() as (_, client):
        result = lc.delete_experiment("demo")
    assert result["status"] == "preview"
    assert result["config"] == {"experiment": "demo", "namespace": "kubeflow"}
    client.delete_job.assert_not_called()


def test_delete_confirmed_calls_sdk():
    with cluster() as (_, client):
        result = lc.delete_experiment("demo", confirmed=True)
    assert result["data"]["deleted"] is True
    client.delete_job.assert_called_once_with(name="demo")


def test_delete_invalid_name():
    result = lc.delete_experiment("Bad_Name!")
    assert result["error_code"] == "VALIDATION_ERROR"


def test_delete_not_found():
    client = MagicMock()
    client.delete_job.side_effect = _not_found()
    with cluster(client=client):
        result = lc.delete_experiment("missing", confirmed=True)
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


# ─── MCP ownership enforcement ─────────────────────────────────────────────


def test_non_admin_blocked_from_deleting_external_experiment():
    with cluster(persona="data-scientist", managed=False) as (_, client):
        result = lc.delete_experiment("external", confirmed=True)
    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    client.delete_job.assert_not_called()


def test_non_admin_allowed_for_mcp_created_experiment():
    with cluster(persona="data-scientist", managed=True) as (_, client):
        result = lc.delete_experiment("ours", confirmed=True)
    assert result["success"] is True
    client.delete_job.assert_called_once()


def test_admin_bypasses_ownership_check():
    with cluster(persona="platform-admin", managed=False) as (_, client):
        result = lc.delete_experiment("external", confirmed=True)
    assert result["success"] is True
    client.delete_job.assert_called_once()


def test_ownership_check_failure_is_reported():
    """managed=None means the label lookup itself failed — must not delete."""
    with cluster(persona="data-scientist", managed=None) as (_, client):
        result = lc.delete_experiment("unknown", confirmed=True)
    assert result["error_code"] == "SDK_ERROR"
    client.delete_job.assert_not_called()


@pytest.mark.parametrize(
    ("call", "attr"),
    [
        (lambda: lc.delete_experiment("typo", confirmed=True), "delete_job"),
        (lambda: lc.update_experiment("typo", "suspend"), "delete_job"),
    ],
)
def test_missing_experiment_reports_not_found_not_ownership(call, attr):
    """A name that does not exist must not be blamed on ownership: that sends
    the caller after RBAC when the real problem is usually a typo."""
    with cluster(persona="data-scientist", managed=mcp_utils.OWNERSHIP_MISSING) as (_, client):
        result = call()
    assert result["success"] is False
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "not created by MCP" not in result["error"]
    getattr(client, attr).assert_not_called()


# ─── update_experiment: suspend / resume ───────────────────────────────────


def test_suspend_zeroes_parallelism_and_stashes_previous():
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = _experiment(parallel=4)
    with cluster(api):
        result = lc.update_experiment("demo", "suspend")

    assert result["data"]["parallel_trial_count"] == 0
    body = api.patch_namespaced_custom_object.call_args.kwargs["body"]
    assert body["spec"]["parallelTrialCount"] == 0
    assert body["metadata"]["annotations"][lc.PARALLELISM_ANNOTATION] == "4"
    # KEP text also mentions resumePolicy; we deliberately do not touch it.
    assert "resumePolicy" not in body["spec"]


def test_resume_restores_stashed_parallelism_and_clears_annotation():
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = _experiment(
        parallel=0, annotations={lc.PARALLELISM_ANNOTATION: "4"}
    )
    with cluster(api):
        result = lc.update_experiment("demo", "resume")

    assert result["data"]["parallel_trial_count"] == 4
    body = api.patch_namespaced_custom_object.call_args.kwargs["body"]
    assert body["spec"]["parallelTrialCount"] == 4
    assert body["metadata"]["annotations"][lc.PARALLELISM_ANNOTATION] is None


def test_resume_without_stash_falls_back_to_default():
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = _experiment(parallel=0)
    with cluster(api):
        result = lc.update_experiment("demo", "resume")
    assert result["data"]["parallel_trial_count"] == lc.DEFAULT_PARALLELISM


def test_resume_never_restores_zero():
    """A corrupt stash of '0' would silently leave the experiment paused."""
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = _experiment(
        parallel=0, annotations={lc.PARALLELISM_ANNOTATION: "0"}
    )
    with cluster(api):
        result = lc.update_experiment("demo", "resume")
    assert result["data"]["parallel_trial_count"] >= lc.DEFAULT_PARALLELISM


def test_suspend_is_idempotent():
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = _experiment(parallel=0)
    with cluster(api):
        result = lc.update_experiment("demo", "suspend")
    assert result["success"] is True
    assert "already suspended" in result["data"]["message"]
    api.patch_namespaced_custom_object.assert_not_called()


def test_suspend_resume_round_trip_preserves_concurrency():
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = _experiment(parallel=7)
    with cluster(api):
        lc.update_experiment("demo", "suspend")
        stashed = api.patch_namespaced_custom_object.call_args.kwargs["body"]["metadata"][
            "annotations"
        ]
        api.get_namespaced_custom_object.return_value = _experiment(parallel=0, annotations=stashed)
        result = lc.update_experiment("demo", "resume")
    assert result["data"]["parallel_trial_count"] == 7


@pytest.mark.parametrize("action", ["pause", "", "SUSPEND"])
def test_invalid_action_rejected(action):
    result = lc.update_experiment("demo", action)
    assert result["error_code"] == "VALIDATION_ERROR"


def test_update_not_found():
    api = MagicMock()
    api.get_namespaced_custom_object.side_effect = _not_found()
    with cluster(api):
        result = lc.update_experiment("missing", "suspend")
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


def test_non_admin_blocked_from_updating_external_experiment():
    api = MagicMock()
    with cluster(api, persona="ml-engineer", managed=False):
        result = lc.update_experiment("external", "suspend")
    assert result["success"] is False
    api.patch_namespaced_custom_object.assert_not_called()
