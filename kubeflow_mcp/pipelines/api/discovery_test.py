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

"""Tests for list_pipelines and get_pipeline."""

import pytest

from kubeflow_mcp.pipelines.api.discovery import get_pipeline, list_pipelines
from kubeflow_mcp.pipelines.conftest import (
    ApiException,
    make_page,
    make_pipeline,
    make_version,
)


class TestListPipelines:
    def test_returns_pipelines_and_paging(self, kfp_client):
        kfp_client.list_pipelines.return_value = make_page(
            "pipelines", [make_pipeline(), make_pipeline("iris", "p-2")], token="next", total=7
        )

        result = list_pipelines(limit=2)

        assert result["success"] is True
        data = result["data"]
        assert [p["name"] for p in data["pipelines"]] == ["churn-prediction", "iris"]
        assert data["pipelines"][0]["created_at"] == "2026-10-20T09:30:00+00:00"
        assert "pipeline_spec" not in data["pipelines"][0]
        assert data["next_page_token"] == "next"
        assert data["has_more"] is True
        assert data["total"] == 7
        assert data["namespace"] == "team-a"
        kfp_client.list_pipelines.assert_called_once_with(page_token="", page_size=2)

    def test_last_page_has_no_more(self, kfp_client):
        kfp_client.list_pipelines.return_value = make_page("pipelines", [])

        data = list_pipelines(page_token="abc")["data"]

        assert data == {
            "pipelines": [],
            "namespace": "team-a",
            "next_page_token": "",
            "has_more": False,
            "total": None,
        }
        kfp_client.list_pipelines.assert_called_once_with(page_token="abc", page_size=50)

    @pytest.mark.parametrize("limit", [0, 201, -1, True])
    def test_rejects_limit_out_of_range(self, kfp_client, limit):
        result = list_pipelines(limit=limit)

        assert result["error_code"] == "VALIDATION_ERROR"
        kfp_client.list_pipelines.assert_not_called()

    def test_permission_denied_maps_to_error_code(self, kfp_client):
        kfp_client.list_pipelines.side_effect = ApiException(403, "Forbidden")

        result = list_pipelines()

        assert result["error_code"] == "PERMISSION_DENIED"

    def test_subclassed_api_exception_is_still_mapped(self, kfp_client):
        class ForbiddenException(ApiException):
            pass

        kfp_client.list_pipelines.side_effect = ForbiddenException(403, "Forbidden")

        assert list_pipelines()["error_code"] == "PERMISSION_DENIED"

    def test_missing_kfp_explains_how_to_install(self, kfp_client):
        kfp_client.list_pipelines.side_effect = ImportError("kfp is required")

        result = list_pipelines()

        assert result["error_code"] == "SDK_ERROR"
        assert "kubeflow[pipelines]" in result["hint"]


class TestGetPipeline:
    def test_returns_pipeline_with_versions(self, kfp_client):
        kfp_client.get_pipeline.return_value = make_pipeline()
        kfp_client.list_pipeline_versions.return_value = make_page(
            "pipeline_versions", [make_version("v2", "pv-2"), make_version()], token="t"
        )

        result = get_pipeline("churn-prediction")

        data = result["data"]
        assert data["pipeline_id"] == "p-1"
        assert [v["name"] for v in data["versions"]] == ["v2", "v1"]
        assert data["versions_has_more"] is True
        assert "pipeline_spec" not in data["versions"][0]
        assert any("create_pipeline_run" in step for step in data["next_steps"])
        kfp_client.list_pipeline_versions.assert_called_once_with("churn-prediction", page_size=20)

    def test_without_versions_skips_the_lookup(self, kfp_client):
        kfp_client.get_pipeline.return_value = make_pipeline()

        data = get_pipeline("churn-prediction", include_versions=False)["data"]

        assert "versions" not in data
        kfp_client.list_pipeline_versions.assert_not_called()

    def test_missing_pipeline_is_not_found(self, kfp_client):
        kfp_client.get_pipeline.side_effect = ValueError(
            "Pipeline not found: 'nope'. Use list_pipelines() to see available pipelines."
        )

        result = get_pipeline("nope")

        assert result["error_code"] == "RESOURCE_NOT_FOUND"

    def test_ambiguous_name_is_a_validation_error(self, kfp_client):
        kfp_client.get_pipeline.side_effect = ValueError(
            "Multiple pipelines found with name 'dup': ['a', 'b']."
        )

        assert get_pipeline("dup")["error_code"] == "VALIDATION_ERROR"

    def test_other_value_error_is_not_reported_as_not_found(self, kfp_client):
        kfp_client.get_pipeline.side_effect = ValueError("something else was not found here")

        assert get_pipeline("x")["error_code"] == "SDK_ERROR"

    def test_timeout_error_is_reported_as_timeout(self, kfp_client):
        kfp_client.get_pipeline.side_effect = TimeoutError("KFP request timed out")

        result = get_pipeline("slow-pipeline")
        assert result["error_code"] == "TIMEOUT"

    @pytest.mark.parametrize(
        "name", ["", "   ", "pipeline.yaml", "/tmp/p.YML", "p.tar.gz", "a\nb", "x" * 257]
    )
    def test_rejects_bad_names_before_any_call(self, kfp_client, name):
        result = get_pipeline(name)

        assert result["error_code"] == "VALIDATION_ERROR"
        kfp_client.get_pipeline.assert_not_called()
