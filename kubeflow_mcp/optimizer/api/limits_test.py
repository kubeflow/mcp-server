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

from unittest.mock import MagicMock, patch

import pytest

from kubeflow_mcp.optimizer.api import discovery, monitoring
from kubeflow_mcp.optimizer.api._common import DEFAULT_LIST_LIMIT, MAX_LIST_LIMIT, clamp_limit
from kubeflow_mcp.optimizer.conftest import event, experiment_cr, trial_cr

_DISC = "kubeflow_mcp.optimizer.api.discovery"


def _experiment_with_trials(katib, count, state="Complete"):
    katib.add_experiment(experiment_cr("exp", status={"trials": count}))
    katib.add_trials(
        *(
            trial_cr(
                f"t{i:03d}",
                state,
                experiment="exp",
                created=f"2026-11-01T00:{i // 60:02d}:{i % 60:02d}Z",
            )
            for i in range(count)
        )
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
    items = [{"metadata": {"name": f"exp-{i}"}} for i in range(120)]
    api = MagicMock()
    api.list_namespaced_custom_object.return_value = {"items": items}
    with (
        patch(f"{_DISC}.get_custom_objects_api", return_value=api),
        patch(f"{_DISC}.get_optimizer_effective_namespace", return_value="default"),
    ):
        result = discovery.list_hpo_experiments(limit=5)
    assert len(result["data"]["experiments"]) == 5
    assert result["data"]["total"] == 120


def test_get_experiment_truncates_trials_and_flags_it(katib):
    _experiment_with_trials(katib, 120)
    data = discovery.get_hpo_experiment("exp", limit=5)["data"]
    assert len(data["trials"]) == 5
    assert data["trials_truncated"] is True
    # Counts describe the whole experiment, not the truncated slice.
    assert data["total_trials"] == 120


def test_get_experiment_does_not_flag_truncation_when_all_fit(katib):
    _experiment_with_trials(katib, 3)
    assert discovery.get_hpo_experiment("exp", limit=50)["data"]["trials_truncated"] is False


def test_get_successful_trials_truncates(katib):
    _experiment_with_trials(katib, 80)
    data = discovery.get_successful_trials("exp", limit=10)["data"]
    assert len(data["trials"]) == 10
    assert data["total"] == 80


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


def test_get_experiment_trials_truncates(katib):
    _experiment_with_trials(katib, 90)
    data = monitoring.get_hpo_experiment_trials("exp", limit=6)["data"]
    assert len(data["trials"]) == 6
    assert data["total"] == 90
    # Summary counts still reflect every trial.
    assert data["summary"]["total_trials"] == 90


def test_get_experiment_events_truncates(katib):
    katib.add_experiment(experiment_cr("exp"))
    katib.core.events.extend(
        event("exp", "Experiment", f"R{i}", f"2026-11-01T00:{i // 60:02d}:{i % 60:02d}Z")
        for i in range(75)
    )
    data = monitoring.get_hpo_experiment_events("exp", limit=3)["data"]
    assert len(data["events"]) == 3
    assert data["total"] == 75


@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [
        (discovery.list_hpo_experiments, {}),
        (discovery.get_hpo_experiment, {"name": "exp"}),
        (discovery.get_successful_trials, {"name": "exp"}),
        (discovery.list_suggestions, {}),
        (monitoring.get_hpo_experiment_trials, {"name": "exp"}),
        (monitoring.get_hpo_experiment_events, {"name": "exp"}),
    ],
)
def test_every_collection_tool_rejects_invalid_limit(tool, kwargs):
    result = tool(limit=0, **kwargs)
    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [
        (discovery.list_hpo_experiments, {}),
        (discovery.get_hpo_experiment, {"name": "exp"}),
        (discovery.get_successful_trials, {"name": "exp"}),
        (discovery.list_suggestions, {}),
        (monitoring.get_hpo_experiment_trials, {"name": "exp"}),
        (monitoring.get_hpo_experiment_events, {"name": "exp"}),
    ],
)
def test_default_limit_is_consistent(tool, kwargs):
    import inspect

    assert inspect.signature(tool).parameters["limit"].default == DEFAULT_LIST_LIMIT
