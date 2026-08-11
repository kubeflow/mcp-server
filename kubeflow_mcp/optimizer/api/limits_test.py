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

"""Result-limit enforcement on optimizer collection tools.

Unbounded responses are an MCP hazard: a large experiment could otherwise emit
every trial into the agent's context window. Every collection tool must bound
what it returns while still reporting the true total.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kubeflow_mcp.optimizer.api import discovery, monitoring
from kubeflow_mcp.optimizer.api._common import DEFAULT_LIST_LIMIT, MAX_LIST_LIMIT, clamp_limit

_DISC = "kubeflow_mcp.optimizer.api.discovery"
_MON = "kubeflow_mcp.optimizer.api.monitoring"


def _trial(i):
    return SimpleNamespace(
        name=f"t{i}",
        parameters={"lr": "0.01"},
        trainjob=SimpleNamespace(name=f"t{i}-job", status="Complete"),
        metrics=[],
    )


def _job(trial_count=0):
    return SimpleNamespace(
        name="exp",
        status="Running",
        creation_timestamp=None,
        trials=[_trial(i) for i in range(trial_count)],
        objectives=[],
        algorithm=SimpleNamespace(random_state=None),
        search_space={},
        trial_config=SimpleNamespace(num_trials=1, parallel_trials=1, max_failed_trials=None),
    )


# ─── clamp_limit ───────────────────────────────────────────────────────────


def test_clamp_limit_caps_at_maximum():
    assert clamp_limit(10_000) == (MAX_LIST_LIMIT, None)


def test_clamp_limit_passes_through_valid_values():
    assert clamp_limit(7) == (7, None)


@pytest.mark.parametrize("bad", [0, -1])
def test_clamp_limit_rejects_non_positive(bad):
    value, err = clamp_limit(bad)
    assert err is not None
    assert err.error_code == "VALIDATION_ERROR"


# ─── every collection tool bounds its response ─────────────────────────────


def test_list_experiments_truncates_but_reports_true_total():
    jobs = [_job() for _ in range(120)]
    client = MagicMock(list_jobs=MagicMock(return_value=jobs))
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.list_experiments(limit=5)
    assert len(result["data"]["experiments"]) == 5
    assert result["data"]["total"] == 120


def test_get_experiment_truncates_trials_and_flags_it():
    client = MagicMock(get_job=MagicMock(return_value=_job(trial_count=120)))
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.get_experiment("exp", limit=5)
    data = result["data"]
    assert len(data["trials"]) == 5
    assert data["trials_truncated"] is True
    # Counts are derived from the full set, not the truncated slice.
    assert data["total_trials"] == 120


def test_get_experiment_does_not_flag_truncation_when_all_fit():
    client = MagicMock(get_job=MagicMock(return_value=_job(trial_count=3)))
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.get_experiment("exp", limit=50)
    assert result["data"]["trials_truncated"] is False


def test_get_successful_trials_truncates():
    client = MagicMock(get_job=MagicMock(return_value=_job(trial_count=80)))
    with patch(f"{_DISC}.get_optimizer_client_for_namespace", return_value=client):
        result = discovery.get_successful_trials("exp", limit=10)
    assert len(result["data"]["trials"]) == 10
    assert result["data"]["total"] == 80


def test_list_suggestions_truncates():
    api = MagicMock()
    api.list_namespaced_custom_object.return_value = {
        "items": [{"metadata": {"name": f"s{i}"}, "spec": {}, "status": {}} for i in range(70)]
    }
    with (
        patch(f"{_DISC}.get_custom_objects_api", return_value=api),
        patch(f"{_DISC}.get_optimizer_effective_namespace", return_value="default"),
    ):
        result = discovery.list_suggestions(limit=4)
    assert len(result["data"]["suggestions"]) == 4
    assert result["data"]["total"] == 70


def test_get_experiment_trials_truncates():
    client = MagicMock(get_job=MagicMock(return_value=_job(trial_count=90)))
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.get_experiment_trials("exp", limit=6)
    assert len(result["data"]["trials"]) == 6
    assert result["data"]["total"] == 90
    # Summary counts still reflect every trial.
    assert result["data"]["summary"]["total_trials"] == 90


def test_get_experiment_events_truncates():
    events = [
        SimpleNamespace(
            involved_object_kind="Pod",
            involved_object_name=f"p{i}",
            reason="Scheduled",
            message="m",
            event_time=None,
        )
        for i in range(75)
    ]
    client = MagicMock(get_job_events=MagicMock(return_value=events))
    with patch(f"{_MON}.get_optimizer_client_for_namespace", return_value=client):
        result = monitoring.get_experiment_events("exp", limit=3)
    assert len(result["data"]["events"]) == 3
    assert result["data"]["total"] == 75


@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [
        (discovery.list_experiments, {}),
        (discovery.get_experiment, {"name": "exp"}),
        (discovery.get_successful_trials, {"name": "exp"}),
        (discovery.list_suggestions, {}),
        (monitoring.get_experiment_trials, {"name": "exp"}),
        (monitoring.get_experiment_events, {"name": "exp"}),
    ],
)
def test_every_collection_tool_rejects_invalid_limit(tool, kwargs):
    result = tool(limit=0, **kwargs)
    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [
        (discovery.list_experiments, {}),
        (discovery.get_experiment, {"name": "exp"}),
        (discovery.get_successful_trials, {"name": "exp"}),
        (discovery.list_suggestions, {}),
        (monitoring.get_experiment_trials, {"name": "exp"}),
        (monitoring.get_experiment_events, {"name": "exp"}),
    ],
)
def test_default_limit_is_consistent(tool, kwargs):
    import inspect

    assert inspect.signature(tool).parameters["limit"].default == DEFAULT_LIST_LIMIT
