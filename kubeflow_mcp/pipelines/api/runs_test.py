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

"""Tests for create_pipeline_run, stop_pipeline_run and retry_pipeline_run."""

import pytest

from kubeflow_mcp.pipelines.api.runs import (
    MAX_PARAMS,
    create_pipeline_run,
    retry_pipeline_run,
    stop_pipeline_run,
)
from kubeflow_mcp.pipelines.conftest import RUN_ID, ApiException, make_pipeline, make_run


class TestCreatePipelineRun:
    def test_preview_resolves_pipeline_and_does_not_run(self, kfp_client):
        kfp_client.get_pipeline.return_value = make_pipeline()

        result = create_pipeline_run("churn-prediction", params={"lr": 0.1})

        assert result["status"] == "preview"
        config = result["config"]
        assert config["pipeline"]["pipeline_id"] == "p-1"
        assert config["params"] == {"lr": 0.1}
        assert config["version"] == "latest"
        assert config["namespace"] == "team-a"
        kfp_client.run.assert_not_called()

    def test_confirmed_runs_the_pipeline_object_not_the_name(self, kfp_client):
        pipeline = make_pipeline()
        kfp_client.get_pipeline.return_value = pipeline
        kfp_client.run.return_value = make_run("PENDING")

        result = create_pipeline_run(
            "churn-prediction",
            version="v1",
            params={"lr": 0.1},
            experiment="exp",
            run_name="nightly",
            confirmed=True,
        )

        assert result["success"] is True
        assert result["data"]["run_id"] == RUN_ID
        assert result["data"]["state"] == "pending"
        args, kwargs = kfp_client.run.call_args
        # A Pipeline object, never a string the SDK could read as a file path.
        assert args == (pipeline,)
        assert kwargs == {
            "params": {"lr": 0.1},
            "name": "nightly",
            "experiment": "exp",
            "version": "v1",
        }
        kfp_client.get_pipeline_version.assert_called_once_with("churn-prediction", "v1")
        kfp_client.get_experiment.assert_called_once_with("exp")

    def test_missing_version_is_caught_at_preview(self, kfp_client):
        kfp_client.get_pipeline.return_value = make_pipeline()
        kfp_client.get_pipeline_version.side_effect = ValueError(
            "Pipeline version not found: 'v9' for pipeline 'churn-prediction'."
        )

        result = create_pipeline_run("churn-prediction", version="v9")

        assert result["error_code"] == "RESOURCE_NOT_FOUND"
        kfp_client.run.assert_not_called()

    @pytest.mark.parametrize(
        "pipeline", ["pipeline.yaml", "/srv/pipelines/p.yml", "bundle.zip", "x.tgz", "x.tar.gz"]
    )
    def test_refuses_file_paths(self, kfp_client, pipeline):
        result = create_pipeline_run(pipeline, confirmed=True)

        assert result["error_code"] == "VALIDATION_ERROR"
        kfp_client.get_pipeline.assert_not_called()
        kfp_client.run.assert_not_called()

    @pytest.mark.parametrize(
        "params",
        [
            ["lr", 0.1],
            {"bad name": 1},
            {"1lr": 1},
            {"lr": object()},
            {f"p{i}": i for i in range(MAX_PARAMS + 1)},
            {"blob": "x" * 40_000},
        ],
    )
    def test_rejects_bad_params(self, kfp_client, params):
        result = create_pipeline_run("churn-prediction", params=params)

        assert result["error_code"] == "VALIDATION_ERROR"
        kfp_client.get_pipeline.assert_not_called()

    def test_run_failure_is_reported(self, kfp_client):
        kfp_client.get_pipeline.return_value = make_pipeline()
        kfp_client.run.side_effect = ApiException(500, "boom")

        result = create_pipeline_run("churn-prediction", confirmed=True)

        assert result["error_code"] == "SDK_ERROR"


class TestStopPipelineRun:
    def test_preview_shows_run_and_ownership_note(self, kfp_client):
        kfp_client.get_run.return_value = make_run("RUNNING")

        result = stop_pipeline_run(RUN_ID)

        assert result["status"] == "preview"
        assert result["config"]["run"]["state"] == "running"
        assert "no MCP ownership label" in result["message"]
        kfp_client.kfp_client.assert_not_called()

    def test_confirmed_terminates(self, kfp_client):
        kfp_client.get_run.return_value = make_run("RUNNING")

        result = stop_pipeline_run(RUN_ID, confirmed=True)

        assert result["data"]["stopped"] is True
        kfp_client.kfp_client.return_value.terminate_run.assert_called_once_with(RUN_ID)

    @pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED", "CANCELED", "SKIPPED"])
    def test_finished_run_is_refused(self, kfp_client, state):
        kfp_client.get_run.return_value = make_run(state)

        result = stop_pipeline_run(RUN_ID, confirmed=True)

        assert result["error_code"] == "VALIDATION_ERROR"
        kfp_client.kfp_client.assert_not_called()

    def test_missing_run_is_not_found(self, kfp_client):
        kfp_client.get_run.side_effect = ApiException(404)

        assert stop_pipeline_run(RUN_ID)["error_code"] == "RESOURCE_NOT_FOUND"

    def test_rejects_malformed_run_id(self, kfp_client):
        assert stop_pipeline_run("not-a-uuid", confirmed=True)["error_code"] == "VALIDATION_ERROR"
        kfp_client.get_run.assert_not_called()


class TestRetryPipelineRun:
    @pytest.mark.parametrize("state", ["FAILED", "CANCELED"])
    def test_preview_for_retryable_run(self, kfp_client, state):
        kfp_client.get_run.return_value = make_run(state)

        result = retry_pipeline_run(RUN_ID)

        assert result["status"] == "preview"
        assert result["config"]["action"] == "retry"
        kfp_client.kfp_client.assert_not_called()

    def test_confirmed_calls_the_run_api(self, kfp_client):
        kfp_client.get_run.return_value = make_run("FAILED")

        result = retry_pipeline_run(RUN_ID, confirmed=True)

        assert result["data"]["retried"] is True
        run_api = kfp_client.kfp_client.return_value._run_api
        run_api.run_service_retry_run.assert_called_once_with(run_id=RUN_ID)

    @pytest.mark.parametrize("state", ["RUNNING", "PENDING", "SUCCEEDED"])
    def test_run_that_did_not_fail_is_refused(self, kfp_client, state):
        kfp_client.get_run.return_value = make_run(state)

        result = retry_pipeline_run(RUN_ID, confirmed=True)

        assert result["error_code"] == "VALIDATION_ERROR"
        kfp_client.kfp_client.assert_not_called()

    def test_permission_denied_on_retry(self, kfp_client):
        kfp_client.get_run.return_value = make_run("FAILED")
        kfp_client.kfp_client.return_value._run_api.run_service_retry_run.side_effect = (
            ApiException(403)
        )

        assert retry_pipeline_run(RUN_ID, confirmed=True)["error_code"] == "PERMISSION_DENIED"
