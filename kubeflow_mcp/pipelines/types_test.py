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

"""Tests for the pipelines SDK-object serialization helpers."""

import json
from datetime import datetime, timezone
from enum import Enum
from types import SimpleNamespace

from kubeflow_mcp.pipelines.conftest import RUN_ID, make_pipeline, make_run, make_task
from kubeflow_mcp.pipelines.types import (
    MAX_CHILD_PODS,
    MAX_PARAMETER_CHARS,
    MAX_PARAMETERS,
    MAX_TASKS,
    pipeline_to_dict,
    run_pod_names,
    run_to_dict,
)


def test_pipeline_is_json_safe_and_drops_the_spec():
    data = pipeline_to_dict(make_pipeline())

    json.dumps(data)
    assert data["created_at"] == "2026-10-20T09:30:00+00:00"
    assert "pipeline_spec" not in data


def test_run_summary_has_no_details():
    data = run_to_dict(make_run("SUCCEEDED"))

    json.dumps(data)
    assert data["run_id"] == RUN_ID
    assert data["state"] == "succeeded"
    assert data["pipeline_id"] == "p-1"
    assert "tasks" not in data and "parameters" not in data


def test_run_times_are_iso_strings():
    run = make_run("SUCCEEDED")
    run.finished_at = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)

    data = run_to_dict(run)

    assert data["created_at"] == "2026-10-20T09:30:00+00:00"
    assert data["finished_at"] == "2026-10-20T10:00:00+00:00"


def test_unfinished_run_has_no_finish_time():
    assert run_to_dict(make_run("RUNNING"))["finished_at"] is None


def test_enum_state_is_read_by_value():
    class State(Enum):
        FAILED = "FAILED"

    assert run_to_dict(make_run(State.FAILED))["state"] == "failed"


def test_run_details_cap_the_task_list():
    tasks = [make_task(f"t{i}") for i in range(MAX_TASKS + 5)]

    data = run_to_dict(make_run(tasks=tasks), include_details=True)

    assert len(data["tasks"]) == MAX_TASKS
    assert data["tasks_truncated"] is True


def test_run_details_bound_child_pods_and_parameters():
    children = [f"child-{i}" for i in range(MAX_CHILD_PODS + 5)]
    params = {f"p{i}": i for i in range(MAX_PARAMETERS + 5)}
    data = run_to_dict(
        make_run(tasks=[make_task("fanout", children=children)], params=params),
        include_details=True,
    )

    assert len(data["tasks"][0]["child_pods"]) == MAX_CHILD_PODS
    assert data["tasks"][0]["child_pods_truncated"] is True
    assert len(data["parameters"]) == MAX_PARAMETERS
    assert data["parameters_truncated"] is True


def test_run_details_skip_parameter_values_over_response_budget():
    data = run_to_dict(
        make_run(params={"too_large": "x" * MAX_PARAMETER_CHARS}),
        include_details=True,
    )

    assert data["parameters"] == {}
    assert data["parameters_truncated"] is True


def test_missing_fields_do_not_raise():
    data = run_to_dict(SimpleNamespace(), include_details=True)

    assert data["run_id"] is None
    assert data["tasks"] == [] and data["parameters"] == {}
    assert data["parameters_truncated"] is False


def test_pod_names_include_children_once():
    run = make_run(
        tasks=[
            make_task("a", pod="pod-a"),
            make_task("b", pod=None, children=["pod-b1", "pod-b2"]),
            make_task("a", pod="pod-a"),
        ]
    )

    assert run_pod_names(run) == [("a", "pod-a"), ("b", "pod-b1"), ("b", "pod-b2")]
