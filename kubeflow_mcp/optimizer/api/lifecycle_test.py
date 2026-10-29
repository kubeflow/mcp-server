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

"""Unit tests for optimizer lifecycle tools (confirmation, ownership, budget edits)."""

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


def _experiment(parallel=3, max_trials=12, max_failed=None, trials=4, conditions=(), **spec):
    """An Experiment CR as the API returns it, with Katib's defaults applied."""
    body = {"parallelTrialCount": parallel, "maxTrialCount": max_trials, **spec}
    if max_failed is not None:
        body["maxFailedTrialCount"] = max_failed
    return {
        "metadata": {"name": "demo"},
        "spec": body,
        "status": {"trials": trials, "conditions": list(conditions)},
    }


def _succeeded(reason):
    return {"type": "Succeeded", "status": "True", "reason": reason}


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
        (lambda: lc.update_experiment("typo", parallel_trial_count=1), "delete_job"),
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


# ─── update_experiment: budget and concurrency edits ───────────────────────


def _update(api, **kwargs):
    with cluster(api):
        return lc.update_experiment("demo", **kwargs)


def _api(**experiment):
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = _experiment(**experiment)
    return api


def test_update_preview_shows_the_change_and_does_not_patch():
    api = _api(parallel=3)
    result = _update(api, parallel_trial_count=1)
    assert result["status"] == "preview"
    assert result["config"]["changes"] == {"parallel_trial_count": {"from": 3, "to": 1}}
    assert "parallel_trial_count 3 -> 1" in result["message"]
    api.patch_namespaced_custom_object.assert_not_called()


def test_update_confirmed_patches_only_changed_spec_fields():
    api = _api(parallel=3, max_trials=12)
    result = _update(api, parallel_trial_count=1, max_trial_count=12, confirmed=True)
    assert result["success"] is True
    assert result["data"]["changes"] == {"parallel_trial_count": {"from": 3, "to": 1}}
    kwargs = api.patch_namespaced_custom_object.call_args.kwargs
    # Nothing else is touched: no annotations, no resumePolicy.
    assert kwargs["body"] == {"spec": {"parallelTrialCount": 1}}
    assert kwargs["_request_timeout"] == mcp_utils.K8S_WRITE_TIMEOUT


def test_update_with_current_values_is_a_no_op():
    api = _api(parallel=3)
    result = _update(api, parallel_trial_count=3, confirmed=True)
    assert result["success"] is True
    assert result["data"]["changes"] == {}
    api.patch_namespaced_custom_object.assert_not_called()


def test_update_requires_at_least_one_field():
    api = MagicMock()
    result = _update(api)
    assert result["error_code"] == "VALIDATION_ERROR"
    api.get_namespaced_custom_object.assert_not_called()


def test_zero_parallelism_is_rejected_with_a_pause_hint():
    """Katib's webhook rejects parallelTrialCount <= 0, so this cannot pause."""
    api = MagicMock()
    result = _update(api, parallel_trial_count=0)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "cannot pause" in result["hint"]
    api.get_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parallel_trial_count": -1},
        {"parallel_trial_count": 101},
        {"parallel_trial_count": True},
        {"max_trial_count": 0},
        {"max_trial_count": 1001},
        {"max_trial_count": "20"},
        {"max_failed_trial_count": -1},
    ],
)
def test_out_of_range_values_are_rejected_before_any_call(kwargs):
    api = MagicMock()
    result = _update(api, **kwargs)
    assert result["error_code"] == "VALIDATION_ERROR"
    api.get_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize(
    ("experiment", "kwargs", "message"),
    [
        ({"max_trials": 5}, {"parallel_trial_count": 6}, "must not exceed max_trial_count"),
        ({"parallel": 4}, {"max_trial_count": 3, "parallel_trial_count": 4}, "must not exceed"),
        ({"max_trials": 10}, {"max_failed_trial_count": 11}, "max_failed_trial_count (11)"),
        ({"trials": 8}, {"max_trial_count": 8}, "greater than the 8 trials"),
        # Any spec edit is refused once every budgeted trial exists.
        ({"trials": 12, "max_trials": 12}, {"parallel_trial_count": 1}, "greater than"),
    ],
)
def test_edits_katib_would_reject_are_refused(experiment, kwargs, message):
    api = _api(**experiment)
    result = _update(api, confirmed=True, **kwargs)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert message in result["error"]
    api.patch_namespaced_custom_object.assert_not_called()


def test_raising_the_budget_above_created_trials_is_allowed():
    api = _api(trials=8, max_trials=8, parallel=2)
    result = _update(api, max_trial_count=20)
    assert result["status"] == "preview"


@pytest.mark.parametrize(
    ("conditions", "spec"),
    [
        ([_succeeded("ExperimentGoalReached")], {"resumePolicy": "LongRunning"}),
        ([_succeeded("ExperimentMaxTrialsReached")], {"resumePolicy": "Never"}),
        ([_succeeded("ExperimentMaxTrialsReached")], {}),  # defaulted to Never
        ([{"type": "Failed", "status": "True", "reason": "ExperimentFailed"}], {}),
    ],
)
def test_finished_experiment_that_katib_cannot_restart_is_refused(conditions, spec):
    api = _api(trials=12, max_trials=12, conditions=conditions, **spec)
    result = _update(api, max_trial_count=20, confirmed=True)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "has finished" in result["error"]
    api.patch_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize("policy", ["LongRunning", "FromVolume"])
def test_finished_experiment_with_a_restartable_policy_can_be_extended(policy):
    api = _api(
        trials=12,
        max_trials=12,
        conditions=[_succeeded("ExperimentMaxTrialsReached")],
        resumePolicy=policy,
    )
    result = _update(api, max_trial_count=20, confirmed=True)
    assert result["success"] is True
    body = api.patch_namespaced_custom_object.call_args.kwargs["body"]
    assert body == {"spec": {"maxTrialCount": 20}}


def test_update_not_found():
    api = MagicMock()
    api.get_namespaced_custom_object.side_effect = _not_found()
    with cluster(api):
        result = lc.update_experiment("missing", parallel_trial_count=1)
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


def test_non_admin_blocked_from_updating_external_experiment():
    api = MagicMock()
    with cluster(api, persona="ml-engineer", managed=False):
        result = lc.update_experiment("external", parallel_trial_count=1, confirmed=True)
    assert result["success"] is False
    assert "update experiments created through MCP" in result["hint"]
    api.patch_namespaced_custom_object.assert_not_called()
