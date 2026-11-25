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

"""Tests for the experiment tools."""

import pytest

from kubeflow_mcp.pipelines.api.experiments import (
    create_experiment,
    delete_experiment,
    get_experiment,
    list_experiments,
)
from kubeflow_mcp.pipelines.conftest import ApiException, make_experiment, make_page

NOT_FOUND = ValueError("Experiment not found: 'churn'. Use create_experiment() to create one.")


class TestListExperiments:
    def test_returns_experiments_and_paging(self, kfp_client):
        kfp_client.list_experiments.return_value = make_page(
            "experiments", [make_experiment()], token="next", total=3
        )

        result = list_experiments(limit=1)

        assert result["success"] is True
        data = result["data"]
        assert data["experiments"][0]["name"] == "churn"
        assert data["experiments"][0]["storage_state"] == "available"
        assert data["experiments"][0]["created_at"].startswith("2026-10-20")
        assert data["has_more"] is True and data["next_page_token"] == "next"
        assert data["total"] == 3
        kfp_client.list_experiments.assert_called_once_with(page_token="", page_size=1)

    def test_experiments_from_other_namespaces_are_dropped(self, kfp_client):
        kfp_client.list_experiments.return_value = make_page(
            "experiments",
            [make_experiment(), make_experiment("other", "e-2", namespace="team-b")],
        )

        data = list_experiments()["data"]

        assert [e["name"] for e in data["experiments"]] == ["churn"]
        assert data["experiments_filtered"] == 1

    @pytest.mark.parametrize("limit", [0, 201, True])
    def test_rejects_limit_out_of_range(self, kfp_client, limit):
        assert list_experiments(limit=limit)["error_code"] == "VALIDATION_ERROR"
        kfp_client.list_experiments.assert_not_called()

    def test_permission_denied_maps_to_error_code(self, kfp_client):
        kfp_client.list_experiments.side_effect = ApiException(403, "Forbidden")
        assert list_experiments()["error_code"] == "PERMISSION_DENIED"


class TestGetExperiment:
    def test_returns_experiment_with_next_steps(self, kfp_client):
        kfp_client.get_experiment.return_value = make_experiment()

        data = get_experiment("churn")["data"]

        assert data["experiment_id"] == "e-1"
        assert any("list_pipeline_runs(experiment='churn')" in s for s in data["next_steps"])

    def test_missing_is_not_found(self, kfp_client):
        kfp_client.get_experiment.side_effect = NOT_FOUND
        assert get_experiment("churn")["error_code"] == "RESOURCE_NOT_FOUND"

    def test_other_namespace_is_refused(self, kfp_client):
        kfp_client.get_experiment.return_value = make_experiment(namespace="team-b")
        result = get_experiment("churn")
        assert result["error_code"] == "PERMISSION_DENIED"
        assert "not in namespace 'team-a'" in result["error"]

    def test_empty_namespace_is_refused_with_an_allowlist(self, kfp_client, monkeypatch):
        monkeypatch.setattr(
            "kubeflow_mcp.pipelines.api._common.get_allowed_namespaces", lambda: ["team-a"]
        )
        kfp_client.get_experiment.return_value = make_experiment(namespace="")
        assert get_experiment("churn")["error_code"] == "PERMISSION_DENIED"

    def test_empty_namespace_is_allowed_without_an_allowlist(self, kfp_client):
        kfp_client.get_experiment.return_value = make_experiment(namespace="")
        assert get_experiment("churn")["success"] is True

    @pytest.mark.parametrize("name", ["", "pipeline.yaml", "bad\nname"])
    def test_rejects_bad_names_before_any_call(self, kfp_client, name):
        assert get_experiment(name)["error_code"] == "VALIDATION_ERROR"
        assert kfp_client.namespaces == []


class TestCreateExperiment:
    def test_preview_does_not_create(self, kfp_client):
        kfp_client.get_experiment.side_effect = NOT_FOUND

        result = create_experiment("churn", description="churn runs")

        assert result["status"] == "preview"
        assert result["config"] == {
            "name": "churn",
            "description": "churn runs",
            "namespace": "team-a",
        }
        kfp_client.create_experiment.assert_not_called()

    def test_confirm_creates(self, kfp_client):
        kfp_client.get_experiment.side_effect = NOT_FOUND
        kfp_client.create_experiment.return_value = make_experiment()

        result = create_experiment("churn", description="churn runs", confirmed=True)

        assert result["success"] is True
        assert result["data"]["created"] is True
        kfp_client.create_experiment.assert_called_once_with("churn", description="churn runs")

    @pytest.mark.parametrize("confirmed", [False, True])
    def test_existing_experiment_is_returned_unchanged(self, kfp_client, confirmed):
        kfp_client.get_experiment.return_value = make_experiment()

        result = create_experiment("churn", confirmed=confirmed)

        assert result["success"] is True
        assert result["data"]["created"] is False
        assert "already exists" in result["data"]["message"]
        kfp_client.create_experiment.assert_not_called()

    def test_existing_experiment_in_other_namespace_is_refused(self, kfp_client):
        kfp_client.get_experiment.return_value = make_experiment(namespace="team-b")
        assert create_experiment("churn")["error_code"] == "PERMISSION_DENIED"

    def test_ambiguous_name_is_reported(self, kfp_client):
        kfp_client.get_experiment.side_effect = ValueError(
            "Multiple experiments found with name 'churn': ['e-1', 'e-2']."
        )
        assert create_experiment("churn")["error_code"] == "VALIDATION_ERROR"

    @pytest.mark.parametrize("confirmed", ["true", 1, None])
    def test_rejects_non_bool_confirmed(self, kfp_client, confirmed):
        result = create_experiment("churn", confirmed=confirmed)
        assert result["error_code"] == "VALIDATION_ERROR"
        kfp_client.create_experiment.assert_not_called()

    def test_rejects_long_description(self, kfp_client):
        result = create_experiment("churn", description="x" * 1001)
        assert result["error_code"] == "VALIDATION_ERROR"
        assert kfp_client.namespaces == []

    def test_create_failure_is_mapped(self, kfp_client):
        kfp_client.get_experiment.side_effect = NOT_FOUND
        kfp_client.create_experiment.side_effect = ApiException(403, "Forbidden")
        result = create_experiment("churn", confirmed=True)
        assert result["error_code"] == "PERMISSION_DENIED"


class TestDeleteExperiment:
    def test_preview_does_not_delete_and_says_runs_are_kept(self, kfp_client):
        kfp_client.get_experiment.return_value = make_experiment()

        result = delete_experiment("churn")

        assert result["status"] == "preview"
        assert "not its runs" in result["message"]
        assert result["config"]["experiment"]["experiment_id"] == "e-1"
        kfp_client.delete_experiment.assert_not_called()

    def test_confirm_deletes(self, kfp_client):
        kfp_client.get_experiment.return_value = make_experiment()

        result = delete_experiment("churn", confirmed=True)

        assert result["data"]["deleted"] is True
        assert result["data"]["experiment_id"] == "e-1"
        kfp_client.delete_experiment.assert_called_once_with("churn")

    def test_missing_is_not_found(self, kfp_client):
        kfp_client.get_experiment.side_effect = NOT_FOUND
        assert delete_experiment("churn", confirmed=True)["error_code"] == "RESOURCE_NOT_FOUND"
        kfp_client.delete_experiment.assert_not_called()

    def test_other_namespace_is_refused(self, kfp_client):
        kfp_client.get_experiment.return_value = make_experiment(namespace="team-b")
        assert delete_experiment("churn", confirmed=True)["error_code"] == "PERMISSION_DENIED"
        kfp_client.delete_experiment.assert_not_called()

    def test_rejects_string_confirmed(self, kfp_client):
        assert delete_experiment("churn", confirmed="false")["error_code"] == "VALIDATION_ERROR"
        kfp_client.delete_experiment.assert_not_called()
