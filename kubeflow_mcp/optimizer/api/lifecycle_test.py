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

"""Unit tests for optimizer lifecycle tools (confirmation, ownership, budget edits)."""

from unittest.mock import patch

import pytest
from kubernetes.client.exceptions import ApiException

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.optimizer.api import lifecycle as lc
from kubeflow_mcp.optimizer.conftest import experiment_cr, trial_cr

_PERSONA = "kubeflow_mcp.optimizer.api._common.get_effective_persona"


@pytest.fixture(autouse=True)
def admin():
    """Run as platform-admin unless a test says otherwise."""
    with patch(_PERSONA, return_value="platform-admin") as persona:
        yield persona


def _succeeded(reason):
    return {"type": "Succeeded", "status": "True", "reason": reason}


def _writes(katib):
    return [name for name, _ in katib.custom.writes()]


# ─── delete_hpo_experiment ─────────────────────────────────────────────────


def test_delete_preview_does_not_delete(katib):
    katib.add_experiment(experiment_cr("demo"))
    result = lc.delete_hpo_experiment("demo")
    assert result["status"] == "preview"
    assert result["config"] == {"experiment": "demo", "namespace": "kubeflow"}
    assert _writes(katib) == []


def test_delete_confirmed_deletes_with_a_finite_timeout(katib):
    """The SDK's delete_job sends no timeout and could block forever."""
    katib.add_experiment(experiment_cr("demo"))
    result = lc.delete_hpo_experiment("demo", confirmed=True)
    assert result["data"]["deleted"] is True
    ((name, kw),) = katib.custom.writes()
    assert name == "delete"
    assert kw["name"] == "demo"
    assert kw["_request_timeout"] == mcp_utils.K8S_WRITE_TIMEOUT


def test_delete_invalid_name(katib):
    assert lc.delete_hpo_experiment("Bad_Name")["error_code"] == "VALIDATION_ERROR"
    assert katib.custom.calls == []


def test_delete_not_found(katib):
    result = lc.delete_hpo_experiment("missing", confirmed=True)
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


# ─── MCP ownership enforcement ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "call",
    [
        lambda: lc.delete_hpo_experiment("external", confirmed=True),
        lambda: lc.update_hpo_experiment("external", max_trial_count=20, confirmed=True),
    ],
)
def test_non_admin_cannot_change_an_experiment_mcp_did_not_create(katib, admin, call):
    admin.return_value = "ml-engineer"
    katib.add_experiment(experiment_cr("external", managed=False))
    result = call()
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "was not created by MCP" in result["error"]
    assert _writes(katib) == []


def test_non_admin_can_change_an_experiment_mcp_created(katib, admin):
    admin.return_value = "data-scientist"
    katib.add_experiment(experiment_cr("ours"))
    assert lc.delete_hpo_experiment("ours", confirmed=True)["success"] is True


def test_admin_bypasses_ownership_check(katib):
    katib.add_experiment(experiment_cr("external", managed=False))
    assert lc.delete_hpo_experiment("external", confirmed=True)["success"] is True


def test_ownership_check_failure_is_reported(katib, admin):
    """An ownership lookup that fails must refuse the change, not allow it."""
    admin.return_value = "data-scientist"
    with patch.object(mcp_utils, "get_optimizer_ownership", return_value=None):
        result = lc.delete_hpo_experiment("unknown", confirmed=True)
    assert result["error_code"] == "SDK_ERROR"
    assert _writes(katib) == []


@pytest.mark.parametrize(
    "call",
    [
        lambda: lc.delete_hpo_experiment("typo", confirmed=True),
        lambda: lc.update_hpo_experiment("typo", parallel_trial_count=1),
    ],
)
def test_missing_experiment_reports_not_found_not_ownership(katib, admin, call):
    """A typo must not be blamed on ownership, which sends callers after RBAC."""
    admin.return_value = "data-scientist"
    result = call()
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "not created by MCP" not in result["error"]


# ─── update_hpo_experiment: budget and concurrency edits ───────────────────


def test_update_preview_shows_the_change_and_does_not_patch(katib):
    katib.add_experiment(experiment_cr("demo", max_trials=12))
    result = lc.update_hpo_experiment("demo", max_trial_count=20)
    assert result["status"] == "preview"
    assert result["config"]["changes"] == {"max_trial_count": {"from": 12, "to": 20}}
    assert result["config"]["trials_to_delete"] == []
    assert "max_trial_count 12 -> 20" in result["message"]
    assert "delete" not in result["message"]
    assert _writes(katib) == []


def test_update_confirmed_patches_only_changed_fields_guarded_by_resource_version(katib):
    katib.add_experiment(experiment_cr("demo", parallel=3, max_trials=12, resource_version="42"))
    result = lc.update_hpo_experiment(
        "demo", parallel_trial_count=4, max_trial_count=12, confirmed=True
    )
    assert result["data"]["changes"] == {"parallel_trial_count": {"from": 3, "to": 4}}
    ((name, kw),) = katib.custom.writes()
    assert name == "patch"
    # Nothing else is touched: no annotations, no resumePolicy. The resource
    # version makes the API server refuse the patch if the experiment changed.
    assert kw["body"] == {"spec": {"parallelTrialCount": 4}, "metadata": {"resourceVersion": "42"}}
    assert kw["_request_timeout"] == mcp_utils.K8S_WRITE_TIMEOUT


def test_update_refuses_a_stale_read(katib):
    """If the experiment changed after it was read, nothing is applied."""
    katib.add_experiment(experiment_cr("demo", parallel=3))
    katib.custom.patch_error = ApiException(status=409, reason="Conflict")
    result = lc.update_hpo_experiment("demo", parallel_trial_count=4, confirmed=True)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "changed while this update was being applied" in result["error"]
    assert "already exists" not in result["error"]


def test_update_with_current_values_is_a_no_op(katib):
    katib.add_experiment(experiment_cr("demo", parallel=3))
    result = lc.update_hpo_experiment("demo", parallel_trial_count=3, confirmed=True)
    assert result["success"] is True
    assert result["data"]["changes"] == {}
    assert _writes(katib) == []


def test_update_requires_at_least_one_field(katib):
    assert lc.update_hpo_experiment("demo")["error_code"] == "VALIDATION_ERROR"
    assert katib.custom.calls == []


def test_zero_parallelism_is_rejected_with_a_pause_hint(katib):
    """Katib's webhook rejects parallelTrialCount <= 0, so this cannot pause."""
    result = lc.update_hpo_experiment("demo", parallel_trial_count=0)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "cannot pause" in result["hint"]
    assert katib.custom.calls == []


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
def test_out_of_range_values_are_rejected_before_any_call(katib, kwargs):
    assert lc.update_hpo_experiment("demo", **kwargs)["error_code"] == "VALIDATION_ERROR"
    assert katib.custom.calls == []


@pytest.mark.parametrize(
    ("experiment", "kwargs", "message"),
    [
        ({"max_trials": 5}, {"parallel_trial_count": 6}, "must not exceed max_trial_count"),
        ({"parallel": 4}, {"max_trial_count": 3, "parallel_trial_count": 4}, "must not exceed"),
        ({"max_trials": 10}, {"max_failed_trial_count": 11}, "max_failed_trial_count (11)"),
        ({"status": {"trials": 8}}, {"max_trial_count": 8}, "greater than the 8 trials"),
        # Any spec edit is refused once every budgeted trial exists.
        ({"status": {"trials": 10}}, {"parallel_trial_count": 1}, "greater than"),
    ],
)
def test_edits_katib_would_reject_are_refused(katib, experiment, kwargs, message):
    katib.add_experiment(experiment_cr("demo", **experiment))
    result = lc.update_hpo_experiment("demo", confirmed=True, **kwargs)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert message in result["error"]
    assert _writes(katib) == []


def test_raising_the_budget_above_created_trials_is_allowed(katib):
    katib.add_experiment(experiment_cr("demo", max_trials=8, status={"trials": 8}))
    assert lc.update_hpo_experiment("demo", max_trial_count=20)["status"] == "preview"


@pytest.mark.parametrize(
    ("conditions", "policy"),
    [
        ([_succeeded("ExperimentGoalReached")], "LongRunning"),
        ([_succeeded("ExperimentMaxTrialsReached")], "Never"),
        ([{"type": "Failed", "status": "True", "reason": "ExperimentFailed"}], "LongRunning"),
    ],
)
def test_finished_experiment_that_katib_cannot_restart_is_refused(katib, conditions, policy):
    katib.add_experiment(
        experiment_cr(
            "demo", max_trials=12, conditions=conditions, resumePolicy=policy, status={"trials": 12}
        )
    )
    result = lc.update_hpo_experiment("demo", max_trial_count=20, confirmed=True)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "has finished" in result["error"]
    assert _writes(katib) == []


@pytest.mark.parametrize("policy", ["LongRunning", "FromVolume"])
def test_finished_experiment_with_a_restartable_policy_can_be_extended(katib, policy):
    katib.add_experiment(
        experiment_cr(
            "demo",
            max_trials=12,
            conditions=[_succeeded("ExperimentMaxTrialsReached")],
            resumePolicy=policy,
            status={"trials": 12},
        )
    )
    result = lc.update_hpo_experiment("demo", max_trial_count=20, confirmed=True)
    assert result["success"] is True
    assert katib.custom.writes()[0][1]["body"]["spec"] == {"maxTrialCount": 20}


def test_update_not_found(katib):
    result = lc.update_hpo_experiment("missing", parallel_trial_count=1)
    assert result["error_code"] == "RESOURCE_NOT_FOUND"


# ─── update_hpo_experiment: trials Katib deletes when parallelism drops ────
# Katib's controller deletes the excess of active trials from ALL of the
# experiment's trials, newest first, whatever their state, and repeats on the
# next reconcile until the active trials fit.


def _demo(katib, *trials, parallel=2):
    katib.add_experiment(
        experiment_cr("demo", parallel=parallel, max_trials=10, status={"trials": len(trials)})
    )
    katib.add_trials(*(trial_cr(n, s, experiment="demo", created=c) for n, s, c in trials))


def test_preview_lists_the_running_trial_katib_deletes(katib):
    """The first live-cluster case: two running, parallelism lowered to 1."""
    _demo(
        katib,
        ("old", "Running", "2026-11-01T10:00:00Z"),
        ("new", "Running", "2026-11-01T10:01:00Z"),
    )
    result = lc.update_hpo_experiment("demo", parallel_trial_count=1)
    assert result["config"]["trials_to_delete"] == [{"name": "new", "status": "Running"}]
    assert result["config"]["trials_to_delete_exact"] is True
    assert "it would delete 1: new (Running)" in result["message"]
    assert _writes(katib) == []


def test_preview_counts_a_newer_completed_trial_katib_deletes_first(katib):
    """Regression from the live repro: two older trials running, the newest
    already succeeded, parallelism lowered to 1. Katib deleted the succeeded
    trial and then one running trial, while the preview had said one."""
    _demo(
        katib,
        ("newest-k99wfnvd", "Running", "2026-11-05T10:05:10Z"),
        ("newest-kqr52hs9", "Running", "2026-11-05T10:05:20Z"),
        ("newest-2f99rkd2", "Complete", "2026-11-05T10:05:25Z"),
    )
    result = lc.update_hpo_experiment("demo", max_trial_count=4, parallel_trial_count=1)

    assert result["config"]["trials_to_delete"] == [
        {"name": "newest-2f99rkd2", "status": "Complete"},
        {"name": "newest-kqr52hs9", "status": "Running"},
    ]
    assert "it would delete 2" in result["message"]
    assert "including 1 that already finished, whose results are lost" in result["message"]


def test_a_trial_without_conditions_counts_as_pending(katib):
    """Regression: a running trial, a conditionless one and a newer completed one
    gave an empty preview, though Katib counts the conditionless trial as pending
    and so deletes the completed trial first, then the conditionless one."""
    _demo(
        katib,
        ("running", "Running", "2026-11-01T00:00:01Z"),
        ("fresh", "Created", "2026-11-01T00:00:02Z"),
        ("done", "Complete", "2026-11-01T00:00:03Z"),
    )
    katib.custom.trials[1]["status"]["conditions"] = []
    result = lc.update_hpo_experiment("demo", parallel_trial_count=1)
    assert result["config"]["trials_to_delete"] == [
        {"name": "done", "status": "Complete"},
        {"name": "fresh", "status": "Created"},
    ]
    assert result["config"]["trials_to_delete_exact"] is True


def test_preview_is_flagged_inexact_when_a_cut_splits_same_second_trials(katib):
    """Katib sorts by creation time alone and timestamps have one-second
    resolution, so which tied trial it deletes is undetermined."""
    same = "2026-11-01T00:00:01Z"
    _demo(
        katib, ("a", "Running", same), ("b", "Running", same), ("c", "Complete", same), parallel=3
    )
    result = lc.update_hpo_experiment("demo", parallel_trial_count=1)
    assert result["config"]["trials_to_delete_exact"] is False
    assert "undetermined" in result["message"]


def test_pending_trials_count_as_active(katib):
    _demo(
        katib,
        ("a", "Running", "2026-11-01T00:00:01Z"),
        ("b", "Created", "2026-11-01T00:00:02Z"),
        ("c", "Created", "2026-11-01T00:00:03Z"),
        parallel=3,
    )
    result = lc.update_hpo_experiment("demo", parallel_trial_count=1)
    assert [t["name"] for t in result["config"]["trials_to_delete"]] == ["c", "b"]


def test_confirmed_response_reports_the_trials_to_delete(katib):
    _demo(
        katib,
        ("a", "Running", "2026-11-01T00:00:01Z"),
        ("b", "Running", "2026-11-01T00:00:02Z"),
        ("c", "Running", "2026-11-01T00:00:03Z"),
        parallel=3,
    )
    result = lc.update_hpo_experiment("demo", parallel_trial_count=1, confirmed=True)
    assert [t["name"] for t in result["data"]["trials_to_delete"]] == ["c", "b"]
    assert "not counted as killed" in result["data"]["message"]


def test_every_reduction_warns_even_when_nothing_is_expected_to_go(katib):
    """The count is an estimate: a trial started before the patch lands can still go."""
    _demo(katib, ("a", "Running", "2026-11-01T00:00:01Z"), parallel=3)
    result = lc.update_hpo_experiment("demo", parallel_trial_count=2)
    assert result["config"]["trials_to_delete"] == []
    assert "none would be deleted now" in result["message"]


@pytest.mark.parametrize(
    "kwargs", [{"parallel_trial_count": 4}, {"max_trial_count": 20}], ids=["raised", "untouched"]
)
def test_no_deletion_warning_unless_parallelism_drops(katib, kwargs):
    _demo(katib, ("a", "Running", "2026-11-01T00:00:01Z"), ("b", "Running", "2026-11-01T00:00:02Z"))
    result = lc.update_hpo_experiment("demo", **kwargs)
    assert result["config"]["trials_to_delete"] == []
    assert "delete" not in result["message"]
    # Trials are only listed when parallelism drops.
    assert not [kw for name, kw in katib.custom.calls if kw.get("plural") == "trials"]
