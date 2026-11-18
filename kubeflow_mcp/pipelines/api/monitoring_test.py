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

"""Tests for list_pipeline_runs, get_pipeline_run and get_pipeline_logs."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kubeflow_mcp.pipelines.api import monitoring
from kubeflow_mcp.pipelines.api.monitoring import (
    get_pipeline_logs,
    get_pipeline_run,
    list_pipeline_runs,
)
from kubeflow_mcp.pipelines.conftest import RUN_ID, ApiException, make_page, make_run, make_task


@pytest.fixture
def core_api(monkeypatch):
    api = MagicMock(name="CoreV1Api")
    monkeypatch.setattr(monitoring, "get_core_v1_api", lambda: api)
    return api


def make_pod(name: str, node: str | None = None, role: str | None = None) -> SimpleNamespace:
    annotations: dict[str, str] = {}
    if node is not None:
        annotations[monitoring.ARGO_NODE_ID_ANNOTATION] = node
    if role is not None:
        annotations[monitoring._ROLE_ANNOTATION] = role
    return SimpleNamespace(metadata=SimpleNamespace(name=name, annotations=annotations))


class TestListPipelineRuns:
    def test_passes_filters_and_lowercases_status(self, kfp_client):
        kfp_client.list_runs.return_value = make_page("runs", [make_run("FAILED")], token="n")

        result = list_pipeline_runs(pipeline="churn", experiment="exp", status="Failed", limit=5)

        data = result["data"]
        assert data["runs"][0]["state"] == "failed"
        assert data["runs"][0]["run_id"] == RUN_ID
        assert "parameters" not in data["runs"][0]
        assert data["has_more"] is True
        kfp_client.list_runs.assert_called_once_with(
            pipeline="churn", experiment="exp", status="failed", page_token="", page_size=5
        )

    def test_rejects_unknown_status(self, kfp_client):
        result = list_pipeline_runs(status="done")

        assert result["error_code"] == "VALIDATION_ERROR"
        assert "succeeded" in result["details"]["supported_statuses"]
        kfp_client.list_runs.assert_not_called()

    def test_rejects_file_path_as_pipeline(self, kfp_client):
        result = list_pipeline_runs(pipeline="p.yaml")

        assert result["error_code"] == "VALIDATION_ERROR"
        kfp_client.list_runs.assert_not_called()

    def test_missing_experiment_is_not_found(self, kfp_client):
        kfp_client.list_runs.side_effect = ValueError("Experiment not found: 'x'.")

        assert list_pipeline_runs(experiment="x")["error_code"] == "RESOURCE_NOT_FOUND"


class TestListPipelineRunsNamespaceFailClosed:
    """Listing applies the same fail-closed namespace check as get_pipeline_run.

    On standalone KFP the SDK lists runs across every experiment, including ones the
    direct read denies (empty experiment namespace with an allowlist configured).
    Unverifiable runs are dropped, never reported.
    """

    @staticmethod
    def _experiment_lookup(namespaces: dict[str, str | None]):
        return lambda experiment_id, **_: SimpleNamespace(
            experiment_id=experiment_id, namespace=namespaces.get(experiment_id)
        )

    def test_allowlist_hides_runs_that_get_pipeline_run_would_deny(self, kfp_client, monkeypatch):
        # Reviewer's live case: allowlist configured, experiment namespace empty.
        monkeypatch.setattr(
            "kubeflow_mcp.pipelines.api._common.get_allowed_namespaces", lambda: {"team-a"}
        )
        monkeypatch.setattr("kubeflow_mcp.core.policy.get_allowed_namespaces", lambda: {"team-a"})
        kfp_client._backend.experiment_api.experiment_service_get_experiment.side_effect = (
            self._experiment_lookup({"e-1": None})
        )
        kfp_client.list_runs.return_value = make_page("runs", [make_run("PENDING")])

        result = list_pipeline_runs(limit=5)

        data = result["data"]
        assert data["runs"] == []
        assert data["runs_filtered"] == 1
        assert data["has_more"] is False

    def test_runs_from_other_experiment_namespaces_are_dropped(self, kfp_client):
        leak = make_run("PENDING")
        leak.run_id = "aaaaaaaa-1111-2222-3333-444444444444"
        leak.experiment_id = "e-2"
        kfp_client.list_runs.return_value = make_page("runs", [make_run("PENDING"), leak])
        kfp_client._backend.experiment_api.experiment_service_get_experiment.side_effect = (
            self._experiment_lookup({"e-1": "team-a", "e-2": "team-b"})
        )

        result = list_pipeline_runs(limit=10)

        data = result["data"]
        assert [run["run_id"] for run in data["runs"]] == [RUN_ID]
        assert data["runs_filtered"] == 1
        assert data["namespace"] == "team-a"

    def test_standalone_without_allowlist_keeps_unnamespaced_runs(self, kfp_client):
        # No allowlist configured (conftest default): empty experiment namespace is
        # the standalone single-user deployment, not an exposure.
        kfp_client._backend.experiment_api.experiment_service_get_experiment.side_effect = (
            self._experiment_lookup({"e-1": None})
        )
        kfp_client.list_runs.return_value = make_page("runs", [make_run("PENDING")])

        result = list_pipeline_runs()

        data = result["data"]
        assert [run["run_id"] for run in data["runs"]] == [RUN_ID]
        assert "runs_filtered" not in data

    def test_experiment_lookup_failure_fails_closed(self, kfp_client):
        kfp_client._backend.experiment_api.experiment_service_get_experiment.side_effect = (
            ApiException(403, "Forbidden")
        )
        kfp_client.list_runs.return_value = make_page("runs", [make_run("PENDING")])

        result = list_pipeline_runs()

        data = result["data"]
        assert data["runs"] == []
        assert data["runs_filtered"] == 1


class TestGetPipelineRun:
    def test_returns_details_and_failed_next_steps(self, kfp_client):
        kfp_client.get_run.return_value = make_run(
            "FAILED",
            error="task train failed",
            tasks=[make_task("prep", pod="prep-pod"), make_task("train", "FAILED", "train-pod")],
        )

        data = get_pipeline_run(RUN_ID)["data"]

        assert data["state"] == "failed"
        assert data["error"] == "task train failed"
        assert data["parameters"] == {"lr": 0.01}
        assert [t["name"] for t in data["tasks"]] == ["prep", "train"]
        assert data["tasks"][1]["state"] == "failed"
        assert "pipeline_spec" not in data
        assert any("get_pipeline_logs" in s for s in data["next_steps"])
        # Hints never tell the agent to skip the preview.
        assert not any("confirmed=True" in s for s in data["next_steps"])

    def test_running_run_suggests_polling(self, kfp_client):
        kfp_client.get_run.return_value = make_run("RUNNING")

        data = get_pipeline_run(RUN_ID)["data"]

        assert data["next_steps"] == [f"get_pipeline_run(run_id='{RUN_ID}') again to follow it"]

    def test_missing_run_is_not_found(self, kfp_client):
        kfp_client.get_run.side_effect = ApiException(404, "Not Found")

        assert get_pipeline_run(RUN_ID)["error_code"] == "RESOURCE_NOT_FOUND"

    @pytest.mark.parametrize("run_id", ["", "abc", "../etc", RUN_ID + "x", RUN_ID + "\n", None])
    def test_rejects_malformed_run_id(self, kfp_client, run_id):
        result = get_pipeline_run(run_id)

        assert result["error_code"] == "VALIDATION_ERROR"
        kfp_client.get_run.assert_not_called()


class TestGetPipelineLogs:
    def _run(self):
        return make_run(
            "FAILED",
            tasks=[
                make_task("prep", pod="prep-pod"),
                make_task("train", "FAILED", "train-pod", children=["train-child"]),
            ],
        )

    def test_reads_main_container_of_each_task_pod(self, kfp_client, core_api):
        kfp_client.get_run.return_value = self._run()
        core_api.read_namespaced_pod_log.side_effect = lambda name, **kw: f"log of {name}"

        data = get_pipeline_logs(RUN_ID, tail_lines=50)["data"]

        assert [(p["task"], p["pod"]) for p in data["pods"]] == [
            ("prep", "prep-pod"),
            ("train", "train-pod"),
            ("train", "train-child"),
        ]
        assert data["pods"][1]["logs"] == "log of train-pod"
        assert data["tasks"] == ["prep", "train"]
        assert data["pods_truncated"] is False
        call = core_api.read_namespaced_pod_log.call_args_list[0].kwargs
        assert call["namespace"] == "team-a"
        assert call["container"] == "main"
        assert call["tail_lines"] == 50

    def test_resolves_argo_node_ids_and_prefers_launcher_pods(self, kfp_client, core_api):
        run = make_run(
            tasks=[
                make_task("prep", pod="node-prep"),
                make_task("train", pod="node-train"),
            ]
        )
        run.run_id = None
        kfp_client.get_run.return_value = run
        core_api.list_namespaced_pod.return_value = SimpleNamespace(
            items=[
                SimpleNamespace(
                    metadata=SimpleNamespace(
                        name="prep-system-container-impl-hash",
                        annotations={
                            monitoring.ARGO_NODE_ID_ANNOTATION: "node-prep",
                            monitoring._ROLE_ANNOTATION: "launcher",
                        },
                    )
                ),
                SimpleNamespace(
                    metadata=SimpleNamespace(
                        name="train-system-container-driver-hash",
                        annotations={
                            monitoring.ARGO_NODE_ID_ANNOTATION: "node-train",
                            monitoring._ROLE_ANNOTATION: "driver",
                        },
                    )
                ),
                SimpleNamespace(
                    metadata=SimpleNamespace(
                        name="train-system-container-impl-hash",
                        annotations={
                            monitoring.ARGO_NODE_ID_ANNOTATION: "node-train",
                            monitoring._ROLE_ANNOTATION: "launcher",
                        },
                    )
                ),
            ]
        )
        core_api.read_namespaced_pod_log.side_effect = lambda name, **kwargs: name

        data = get_pipeline_logs(RUN_ID)["data"]

        assert [pod["pod"] for pod in data["pods"]] == [
            "prep-system-container-impl-hash",
            "train-system-container-impl-hash",
        ]
        core_api.list_namespaced_pod.assert_called_once_with(
            namespace="team-a",
            label_selector=f"{monitoring.RUN_ID_LABEL}={RUN_ID}",
            _request_timeout=monitoring.K8S_TIMEOUT,
        )

    def test_pod_listing_failure_is_reported(self, kfp_client, core_api):
        kfp_client.get_run.return_value = self._run()
        core_api.list_namespaced_pod.side_effect = ApiException(403, "Forbidden")

        result = get_pipeline_logs(RUN_ID)

        assert result["error_code"] == "PERMISSION_DENIED"
        assert "resolve pods" in result["error"]
        core_api.read_namespaced_pod_log.assert_not_called()

    def test_task_filter(self, kfp_client, core_api):
        kfp_client.get_run.return_value = self._run()
        core_api.read_namespaced_pod_log.return_value = "x"

        data = get_pipeline_logs(RUN_ID, task="prep")["data"]

        assert [p["pod"] for p in data["pods"]] == ["prep-pod"]

    def test_unknown_task_lists_known_tasks(self, kfp_client, core_api):
        kfp_client.get_run.return_value = self._run()

        result = get_pipeline_logs(RUN_ID, task="deploy")

        assert result["error_code"] == "RESOURCE_NOT_FOUND"
        assert result["details"]["tasks"] == ["prep", "train"]
        core_api.read_namespaced_pod_log.assert_not_called()

    def test_falls_back_to_default_container(self, kfp_client, core_api):
        kfp_client.get_run.return_value = make_run(tasks=[make_task("drv", pod="driver-pod")])
        core_api.read_namespaced_pod_log.side_effect = [ApiException(400), "driver log"]

        data = get_pipeline_logs(RUN_ID)["data"]

        assert data["pods"][0]["logs"] == "driver log"
        assert "container" not in core_api.read_namespaced_pod_log.call_args_list[1].kwargs

    def test_cleaned_up_pod_is_reported_not_raised(self, kfp_client, core_api):
        kfp_client.get_run.return_value = make_run(tasks=[make_task("t", pod="gone-pod")])
        core_api.read_namespaced_pod_log.side_effect = ApiException(404)

        data = get_pipeline_logs(RUN_ID)["data"]

        assert data["pods"][0]["error"] == "pod no longer exists"

    def test_bounds_pods_and_characters(self, kfp_client, core_api):
        tasks = [make_task(f"t{i}", pod=f"pod-{i}") for i in range(monitoring.MAX_LOG_PODS + 3)]
        kfp_client.get_run.return_value = make_run(tasks=tasks)
        core_api.read_namespaced_pod_log.return_value = "y" * 50_000

        data = get_pipeline_logs(RUN_ID)["data"]

        assert len(data["pods"]) == monitoring.MAX_LOG_PODS
        assert data["pods_truncated"] is True
        total = sum(len(p["logs"]) for p in data["pods"])
        assert total <= monitoring.MAX_LOG_CHARS + 100 * monitoring.MAX_LOG_PODS

    def test_skips_pod_names_that_are_not_valid(self, kfp_client, core_api):
        kfp_client.get_run.return_value = make_run(tasks=[make_task("t", pod="../../etc")])

        data = get_pipeline_logs(RUN_ID)["data"]

        assert data["pods"][0]["error"] == "invalid pod name"
        core_api.read_namespaced_pod_log.assert_not_called()

    @pytest.mark.parametrize("tail_lines", [0, 1001, -5, True])
    def test_rejects_tail_lines_out_of_range(self, kfp_client, tail_lines):
        assert get_pipeline_logs(RUN_ID, tail_lines=tail_lines)["error_code"] == "VALIDATION_ERROR"
        kfp_client.get_run.assert_not_called()

    def test_dag_structure_launcher_pod_is_found_when_driver_pod_is_gone(
        self, kfp_client, core_api
    ):
        # Live-tested failure: task="work" answered 'pod no longer exists' (its
        # recorded DAG driver pod was cleaned up) while the task's launcher pod still
        # held the logs.
        run = make_run(
            tasks=[
                make_task("work", "FAILED", pod="work-driver-node", children=["work-launch-node"]),
            ]
        )
        kfp_client.get_run.return_value = run
        core_api.list_namespaced_pod.return_value = SimpleNamespace(
            items=[make_pod("work-executor-pod", node="work-launch-node", role="launcher")]
        )

        def read_log(name, **kwargs):
            if name == "work-driver-node":
                raise ApiException(404, "Not Found")
            return name

        core_api.read_namespaced_pod_log.side_effect = read_log

        data = get_pipeline_logs(RUN_ID, task="work")["data"]

        # The launcher pod is read first; the stale recorded ID is reported after.
        assert [(p["task"], p["pod"]) for p in data["pods"]] == [
            ("work", "work-executor-pod"),
            ("work", "work-driver-node"),
        ]
        assert data["pods"][0]["logs"] == "work-executor-pod"
        assert data["pods"][1]["error"] == "pod no longer exists"

    def test_task_filter_matches_executor_child_when_parent_task_id_and_pod_name_are_empty(
        self, kfp_client, core_api
    ):
        work = make_task("work", "FAILED", pod="work-driver-node", task_id="task-104")
        work.child_tasks = [SimpleNamespace(task_id="task-209", pod_name=None)]
        train = make_task("train", "FAILED", pod="train-driver-node", task_id="task-105")
        train.child_tasks = [SimpleNamespace(task_id="task-210", pod_name=None)]
        run = make_run(
            tasks=[
                work,
                train,
                make_task(
                    "executor",
                    pod=None,
                    children=["work-launch-node"],
                    task_id="task-209",
                    parent_task_id="",
                ),
                make_task(
                    "executor",
                    pod=None,
                    children=["train-launch-node"],
                    task_id="task-210",
                    parent_task_id="",
                ),
            ]
        )
        kfp_client.get_run.return_value = run
        core_api.list_namespaced_pod.return_value = SimpleNamespace(
            items=[
                make_pod("work-launcher-pod", node="work-launch-node", role="launcher"),
                make_pod("train-launcher-pod", node="train-launch-node", role="launcher"),
            ]
        )
        core_api.read_namespaced_pod_log.side_effect = lambda name, **kwargs: f"logs:{name}"

        unfiltered = get_pipeline_logs(RUN_ID)["data"]
        data = get_pipeline_logs(RUN_ID, task="work")["data"]

        assert [pod["pod"] for pod in data["pods"]] == [
            "work-launcher-pod",
            "work-driver-node",
        ]
        assert data["pods"][0]["logs"] == next(
            pod["logs"] for pod in unfiltered["pods"] if pod["pod"] == "work-launcher-pod"
        )
        assert "train-launcher-pod" not in [pod["pod"] for pod in data["pods"]]
        assert data["tasks"] == ["train", "work"]

    def test_dag_driver_entries_cannot_exhaust_the_pod_cap(self, kfp_client, core_api):
        tasks = [make_task(f"driver-step-{i}", "FAILED", pod=f"driver-node-{i}") for i in range(12)]
        tasks.append(make_task("work", "FAILED", pod="work-own-node"))
        kfp_client.get_run.return_value = make_run(tasks=tasks)
        pods = [make_pod(f"driver-pod-{i}", node=f"driver-node-{i}") for i in range(12)]
        pods.append(make_pod("work-launcher-pod", node="work-own-node", role="launcher"))
        core_api.list_namespaced_pod.return_value = SimpleNamespace(items=pods)
        core_api.read_namespaced_pod_log.side_effect = lambda name, **kw: name

        data = get_pipeline_logs(RUN_ID)["data"]

        # The launcher pod wins a read slot even though the run recorded it last.
        assert data["pods"][0]["task"] == "work"
        assert data["pods"][0]["pod"] == "work-launcher-pod"
        assert data["pods"][0]["logs"] == "work-launcher-pod"
        assert "work" in data["tasks"]
        assert len(data["pods"]) == monitoring.MAX_LOG_PODS
        assert data["pods_truncated"] is True

    def test_same_pod_recorded_by_task_and_child_is_read_once(self, kfp_client, core_api):
        run = make_run(
            tasks=[
                make_task("parent", pod=None, children=["shared-node"]),
                make_task("work", "FAILED", pod="shared-node"),
            ]
        )
        kfp_client.get_run.return_value = run
        core_api.list_namespaced_pod.return_value = SimpleNamespace(
            items=[make_pod("shared-executor-pod", node="shared-node", role="launcher")]
        )
        core_api.read_namespaced_pod_log.side_effect = lambda name, **kw: name

        data = get_pipeline_logs(RUN_ID)["data"]

        assert [(p["task"], p["pod"]) for p in data["pods"]] == [("parent", "shared-executor-pod")]
        assert core_api.read_namespaced_pod_log.call_count == 1
