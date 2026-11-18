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

"""Tests for create_pipeline."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kubeflow_mcp.common.types import ToolError
from kubeflow_mcp.pipelines.api import authoring
from kubeflow_mcp.pipelines.api._compile import CompiledPipeline
from kubeflow_mcp.pipelines.api.authoring import create_pipeline, delete_pipeline
from kubeflow_mcp.pipelines.conftest import (
    PIPELINE_CODE,
    ApiException,
    make_page,
    make_pipeline,
    make_version,
)

NOT_FOUND = ValueError("Pipeline not found: 'churn-prediction'.")
UPLOADED = SimpleNamespace(pipeline_id="p-1", pipeline_version_id="pv-9", display_name="v2")
SPEC = [
    {
        "pipelineInfo": {"name": "churn-prediction"},
        "root": {
            "inputDefinitions": {
                "parameters": {
                    "rows": {"parameterType": "NUMBER_INTEGER", "defaultValue": 100.0},
                    "epochs": {"parameterType": "NUMBER_INTEGER", "defaultValue": 2.0},
                }
            }
        },
        "deploymentSpec": {
            "executors": {
                "exec-load": {"container": {"image": "python:3.11-slim"}},
                "exec-train": {"container": {"image": "python:3.11-slim"}},
            }
        },
    }
]
needs_kfp = pytest.mark.skipif(
    importlib.util.find_spec("kfp") is None, reason="needs the pipelines extra (kfp)"
)


@pytest.fixture
def new_pipeline(kfp_client):
    kfp_client.get_pipeline.side_effect = NOT_FOUND
    kfp_client.upload_pipeline.return_value = UPLOADED
    return kfp_client


@pytest.fixture
def fake_compile(monkeypatch):
    """Replace the child-process compile with one that writes SPEC."""
    calls = []

    def compile_code(code, workdir, timeout=None):
        calls.append(workdir)
        path = os.path.join(workdir, "pipeline.yaml")
        Path(path).write_text("# PIPELINE DEFINITION\n", encoding="utf-8")
        return CompiledPipeline(path=path, documents=SPEC, size_bytes=22), None

    monkeypatch.setattr(authoring, "compile_pipeline_code", compile_code)
    return calls


@pytest.fixture
def no_compile(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("compile must not run")

    monkeypatch.setattr(authoring, "compile_pipeline_code", fail)


class TestPreview:
    def test_new_pipeline_preview(self, new_pipeline, no_compile):
        result = create_pipeline("churn-prediction", PIPELINE_CODE, description="d")

        assert result["status"] == "preview"
        config = result["config"]
        assert config["action"] == "create"
        assert config["existing_pipeline"] is None
        assert config["namespace"] == "team-a"
        assert [c["name"] for c in config["components"]] == ["load", "train"]
        assert config["parameters"][0]["name"] == "rows"
        assert config["safety_warnings"] == []
        new_pipeline.upload_pipeline.assert_not_called()

    def test_existing_pipeline_gets_a_new_version(self, kfp_client, no_compile):
        kfp_client.get_pipeline.return_value = make_pipeline()

        result = create_pipeline("churn-prediction", PIPELINE_CODE, version="v2")

        assert result["config"]["action"] == "new_version"
        assert result["config"]["existing_pipeline"]["pipeline_id"] == "p-1"
        assert "new version" in result["message"]

    def test_preview_shows_safety_warnings(self, new_pipeline, no_compile):
        code = PIPELINE_CODE.replace(
            "import pandas as pd",
            "import subprocess\n    subprocess.run(['ls'])\n    import pandas as pd",
        )

        result = create_pipeline("churn-prediction", code)

        assert result["config"]["safety_warnings"]
        assert "KUBEFLOW_MCP_UNSAFE_SCRIPTS" in result["message"]

    def test_preview_checks_dataset_format_without_the_hub(self, new_pipeline, no_compile):
        with patch("huggingface_hub.dataset_info") as hub:
            result = create_pipeline("churn-prediction", PIPELINE_CODE, dataset="hf://org/ds")

        assert result["config"]["dataset"] == "hf://org/ds"
        hub.assert_not_called()


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"name": ""},
            {"name": "pipeline.yaml"},
            {"version": "v1.zip"},
            {"description": "x" * 1001},
            {"dataset": "hf://not-org-slash-name"},
            {"dataset": " hf://org/ds"},
            {"dataset": ""},
            {"dataset": "s3://" + "b" * 3000},
            {"confirmed": "false"},
            {"pipeline_code": "import os\n"},
            {"pipeline_code": ""},
        ],
    )
    def test_rejected_before_any_client_call(self, kfp_client, no_compile, kwargs):
        args = {"name": "churn-prediction", "pipeline_code": PIPELINE_CODE, **kwargs}

        result = create_pipeline(**args)

        assert result["error_code"] == "VALIDATION_ERROR", result
        assert kfp_client.namespaces == []

    def test_ambiguous_pipeline_name(self, kfp_client, no_compile):
        kfp_client.get_pipeline.side_effect = ValueError(
            "Multiple pipelines found with name 'churn-prediction': ['p-1', 'p-2']."
        )
        assert (
            create_pipeline("churn-prediction", PIPELINE_CODE)["error_code"] == "VALIDATION_ERROR"
        )

    def test_missing_kfp_explains_how_to_install(self, kfp_client, no_compile):
        kfp_client.get_pipeline.side_effect = ImportError("kfp is required")
        result = create_pipeline("churn-prediction", PIPELINE_CODE)
        assert result["error_code"] == "SDK_ERROR"
        assert "pipelines" in result["hint"]


class TestConfirm:
    def test_compiles_uploads_and_cleans_up(self, new_pipeline, fake_compile):
        seen = {}

        def upload(path, **kwargs):
            seen["exists"] = os.path.isfile(path)
            seen["path"] = path
            seen["kwargs"] = kwargs
            return UPLOADED

        new_pipeline.upload_pipeline.side_effect = upload

        result = create_pipeline(
            "churn-prediction", PIPELINE_CODE, description="d", version="v2", confirmed=True
        )

        assert result["success"] is True, result
        data = result["data"]
        assert data["created_new_pipeline"] is True
        assert data["pipeline_version_id"] == "pv-9"
        assert data["images"] == ["python:3.11-slim"]
        assert data["parameters"]["rows"]["type"] == "NUMBER_INTEGER"
        # Defaults as written in the source, not the spec's floats.
        assert "params={'rows': 100, 'epochs': 2}" in data["next_steps"][0]
        assert seen["exists"] is True
        assert seen["kwargs"] == {"name": "churn-prediction", "version": "v2", "description": "d"}
        assert not os.path.exists(seen["path"])
        assert not os.path.exists(fake_compile[0])

    def test_compile_error_stops_before_upload(self, new_pipeline, monkeypatch):
        error = ToolError(error="The pipeline failed to compile: x", error_code="VALIDATION_ERROR")
        monkeypatch.setattr(authoring, "compile_pipeline_code", lambda *a, **k: (None, error))

        result = create_pipeline("churn-prediction", PIPELINE_CODE, confirmed=True)

        assert result["error"].startswith("The pipeline failed to compile")
        new_pipeline.upload_pipeline.assert_not_called()

    def test_bad_image_stops_before_upload(self, new_pipeline, monkeypatch):
        bad = [
            dict(
                SPEC[0], deploymentSpec={"executors": {"e": {"container": {"image": "Bad Image"}}}}
            )
        ]

        def compile_code(code, workdir, timeout=None):
            return CompiledPipeline(
                path=os.path.join(workdir, "p.yaml"), documents=bad, size_bytes=1
            ), None

        monkeypatch.setattr(authoring, "compile_pipeline_code", compile_code)

        result = create_pipeline("churn-prediction", PIPELINE_CODE, confirmed=True)

        assert result["error_code"] == "VALIDATION_ERROR"
        new_pipeline.upload_pipeline.assert_not_called()

    def test_upload_error_is_mapped(self, new_pipeline, fake_compile):
        new_pipeline.upload_pipeline.side_effect = ApiException(403, "Forbidden")
        result = create_pipeline("churn-prediction", PIPELINE_CODE, confirmed=True)
        assert result["error_code"] == "PERMISSION_DENIED"

    def test_unsafe_code_is_blocked(self, new_pipeline, fake_compile, monkeypatch):
        monkeypatch.delenv("KUBEFLOW_MCP_UNSAFE_SCRIPTS", raising=False)
        code = PIPELINE_CODE.replace(
            "import pandas as pd", "import os\n    os.system('ls')\n    import pandas as pd"
        )

        result = create_pipeline("churn-prediction", code, confirmed=True)

        assert result["error_code"] == "VALIDATION_ERROR"
        assert result["details"]["safety_warnings"]
        assert fake_compile == []

    def test_unsafe_code_with_override(self, new_pipeline, fake_compile, monkeypatch):
        monkeypatch.setenv("KUBEFLOW_MCP_UNSAFE_SCRIPTS", "true")
        code = PIPELINE_CODE.replace(
            "import pandas as pd", "import os\n    os.system('ls')\n    import pandas as pd"
        )

        assert create_pipeline("churn-prediction", code, confirmed=True)["success"] is True

    def test_missing_hf_dataset_is_rejected_before_compile(self, new_pipeline, fake_compile):
        with patch("kubeflow_mcp.pipelines.api.authoring.check_hf_dataset_exists") as check:
            check.return_value = ToolError(
                error="Hugging Face dataset 'org/x' does not exist", error_code="VALIDATION_ERROR"
            )
            result = create_pipeline(
                "churn-prediction", PIPELINE_CODE, dataset="hf://org/x", confirmed=True
            )

        assert "does not exist" in result["error"]
        assert fake_compile == []

    def test_non_hf_dataset_is_not_looked_up(self, new_pipeline, fake_compile):
        with patch("huggingface_hub.dataset_info") as hub:
            result = create_pipeline(
                "churn-prediction", PIPELINE_CODE, dataset="s3://b/data", confirmed=True
            )

        assert result["success"] is True
        hub.assert_not_called()


@needs_kfp
def test_end_to_end_with_real_compile(new_pipeline):
    """The real child-process compile; the uploaded file is the compiler's own output."""
    seen = {}

    def upload(path, **kwargs):
        seen["text"] = Path(path).read_text(encoding="utf-8")
        return UPLOADED

    new_pipeline.upload_pipeline.side_effect = upload

    result = create_pipeline("churn-prediction", PIPELINE_CODE, confirmed=True)

    assert result["success"] is True, result
    assert seen["text"].startswith("# PIPELINE DEFINITION\n# Name: churn-prediction\n")
    assert result["data"]["pipeline_name"] == "churn-prediction"
    assert {s["executor"] for s in result["data"]["steps"]} == {"exec-load", "exec-train"}


class TestDeletePipeline:
    @pytest.fixture
    def registered(self, kfp_client):
        kfp_client.get_pipeline.return_value = make_pipeline()
        kfp_client.list_pipeline_versions.return_value = make_page(
            "pipeline_versions", [make_version("v1"), make_version("v2", "pv-2")]
        )
        return kfp_client

    def test_preview_lists_every_version_and_deletes_nothing(self, registered):
        result = delete_pipeline("churn-prediction")

        assert result["status"] == "preview"
        assert result["config"]["versions_to_delete"] == ["v1", "v2"]
        assert result["config"]["action"] == "delete_pipeline"
        assert "all its versions (2)" in result["message"]
        assert "Runs of this pipeline are kept" in result["message"]
        registered.delete_pipeline.assert_not_called()

    def test_confirm_deletes_the_pipeline_with_its_versions(self, registered):
        result = delete_pipeline("churn-prediction", confirmed=True)

        assert result["data"]["deleted"] == "pipeline"
        assert result["data"]["versions_deleted"] == ["v1", "v2"]
        registered.delete_pipeline.assert_called_once_with(
            "churn-prediction", version=None, force=True
        )

    def test_one_version(self, registered):
        registered.get_pipeline_version.return_value = make_version("v1")

        preview = delete_pipeline("churn-prediction", version="v1")
        result = delete_pipeline("churn-prediction", version="v1", confirmed=True)

        assert preview["config"]["action"] == "delete_version"
        assert preview["config"]["versions_to_delete"] == ["v1"]
        assert result["data"]["deleted"] == "version"
        registered.delete_pipeline.assert_called_once_with(
            "churn-prediction", version="v1", force=False
        )
        registered.list_pipeline_versions.assert_not_called()

    def test_missing_version_is_not_found(self, registered):
        registered.get_pipeline_version.side_effect = ValueError(
            "Pipeline version not found: 'v9' for pipeline 'churn-prediction'."
        )
        result = delete_pipeline("churn-prediction", version="v9", confirmed=True)
        assert result["error_code"] == "RESOURCE_NOT_FOUND"
        registered.delete_pipeline.assert_not_called()

    def test_missing_pipeline_is_not_found(self, kfp_client):
        kfp_client.get_pipeline.side_effect = NOT_FOUND
        assert delete_pipeline("churn-prediction")["error_code"] == "RESOURCE_NOT_FOUND"
        kfp_client.delete_pipeline.assert_not_called()

    def test_pipeline_in_another_namespace_is_refused(self, kfp_client):
        other = make_pipeline()
        other.namespace = "team-b"
        kfp_client.get_pipeline.return_value = other

        result = delete_pipeline("churn-prediction", confirmed=True)

        assert result["error_code"] == "PERMISSION_DENIED"
        kfp_client.delete_pipeline.assert_not_called()

    def test_shared_pipeline_is_refused_with_an_allowlist(self, kfp_client, monkeypatch):
        monkeypatch.setattr(
            "kubeflow_mcp.pipelines.api._common.get_allowed_namespaces", lambda: ["team-a"]
        )
        shared = make_pipeline()
        shared.namespace = ""
        kfp_client.get_pipeline.return_value = shared

        assert (
            delete_pipeline("churn-prediction", confirmed=True)["error_code"] == "PERMISSION_DENIED"
        )
        kfp_client.delete_pipeline.assert_not_called()

    def test_more_versions_than_listed_are_flagged(self, registered):
        registered.list_pipeline_versions.return_value = make_page(
            "pipeline_versions", [make_version("v1")], token="more"
        )
        result = delete_pipeline("churn-prediction")
        assert result["config"]["versions_truncated"] is True
        assert "(1+)" in result["message"]

    @pytest.mark.parametrize(
        "kwargs", [{"name": ""}, {"version": "v1.yaml"}, {"confirmed": "true"}]
    )
    def test_rejected_before_any_call(self, kfp_client, kwargs):
        result = delete_pipeline(**{"name": "churn-prediction", **kwargs})
        assert result["error_code"] == "VALIDATION_ERROR"
        assert kfp_client.namespaces == []

    def test_delete_failure_is_mapped(self, registered):
        registered.delete_pipeline.side_effect = ApiException(403, "Forbidden")
        result = delete_pipeline("churn-prediction", confirmed=True)
        assert result["error_code"] == "PERMISSION_DENIED"
