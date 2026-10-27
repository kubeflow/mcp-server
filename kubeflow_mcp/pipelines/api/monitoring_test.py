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
