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

"""Namespace policy sweep: every pipelines tool checks the namespace it runs in."""

import pytest

from kubeflow_mcp.pipelines import TOOLS
from kubeflow_mcp.pipelines.conftest import RUN_ID, make_pipeline, make_run

# Minimal valid arguments per tool, so each call reaches the namespace check.
CALLS = {
    "list_pipelines": {},
    "get_pipeline": {"name": "churn-prediction"},
    "list_pipeline_runs": {},
    "get_pipeline_run": {"run_id": RUN_ID},
    "get_pipeline_logs": {"run_id": RUN_ID},
    "create_pipeline_run": {"pipeline": "churn-prediction"},
    "stop_pipeline_run": {"run_id": RUN_ID},
    "retry_pipeline_run": {"run_id": RUN_ID},
}
TOOLS_BY_NAME = {t.__name__: t for t in TOOLS}


def test_sweep_covers_every_tool():
    assert set(CALLS) == set(TOOLS_BY_NAME)


@pytest.fixture
def allowlist(kfp_client, monkeypatch):
    monkeypatch.setattr("kubeflow_mcp.core.policy.get_allowed_namespaces", lambda: ["team-a"])
    kfp_client.get_pipeline.return_value = make_pipeline()
    kfp_client.get_run.return_value = make_run("FAILED")
    return kfp_client


@pytest.mark.parametrize("tool", sorted(CALLS))
def test_disallowed_namespace_is_refused_before_any_call(allowlist, tool):
    result = TOOLS_BY_NAME[tool](namespace="team-b", **CALLS[tool])

    assert result["error_code"] == "PERMISSION_DENIED", result
    assert allowlist.namespaces == []


@pytest.mark.parametrize("tool", sorted(CALLS))
def test_default_namespace_is_checked_and_used(allowlist, monkeypatch, tool):
    # The default resolves to a namespace outside the allowlist: omitting
    # namespace must not slip past the policy.
    monkeypatch.setattr(
        "kubeflow_mcp.pipelines.api._common.get_pipelines_effective_namespace",
        lambda namespace=None: namespace or "team-b",
    )

    result = TOOLS_BY_NAME[tool](**CALLS[tool])

    assert result["error_code"] == "PERMISSION_DENIED", result
    assert allowlist.namespaces == []


@pytest.mark.parametrize("tool", sorted(CALLS))
def test_allowed_default_is_the_namespace_the_client_uses(allowlist, tool):
    TOOLS_BY_NAME[tool](**CALLS[tool])

    assert allowlist.namespaces == ["team-a"]


@pytest.mark.parametrize("tool", sorted(CALLS))
def test_malformed_namespace_is_rejected(kfp_client, tool):
    result = TOOLS_BY_NAME[tool](namespace="Bad_NS!", **CALLS[tool])

    assert result["error_code"] == "VALIDATION_ERROR", result
    assert kfp_client.namespaces == []


def test_unresolvable_default_fails_closed(kfp_client, monkeypatch):
    def boom(namespace=None):
        raise RuntimeError("no kubeconfig")

    monkeypatch.setattr(
        "kubeflow_mcp.pipelines.api._common.get_pipelines_effective_namespace", boom
    )

    result = TOOLS_BY_NAME["list_pipelines"]()

    assert result["error_code"] == "PERMISSION_DENIED"
    assert kfp_client.namespaces == []


RUN_TOOLS = ["get_pipeline_run", "get_pipeline_logs", "stop_pipeline_run", "retry_pipeline_run"]


@pytest.mark.parametrize("tool", RUN_TOOLS)
def test_run_in_different_namespace_is_refused(kfp_client, tool):
    from types import SimpleNamespace

    kfp_client.get_run.return_value = make_run("FAILED")
    kfp_client._backend.experiment_api.experiment_service_get_experiment.return_value = (
        SimpleNamespace(experiment_id="e-1", namespace="team-b")
    )

    result = TOOLS_BY_NAME[tool](namespace="team-a", **CALLS[tool])

    assert result["error_code"] == "PERMISSION_DENIED"
    assert f"Run '{RUN_ID}' is not in namespace 'team-a'" in result["error"]


@pytest.mark.parametrize("tool", RUN_TOOLS)
def test_experiment_lookup_error_fails_closed(kfp_client, tool):
    """When the experiment API raises, the call must be refused (fail-closed)."""
    from kubeflow_mcp.pipelines.conftest import ApiException

    kfp_client.get_run.return_value = make_run("FAILED")
    kfp_client._backend.experiment_api.experiment_service_get_experiment.side_effect = ApiException(
        500, "Internal Server Error"
    )

    result = TOOLS_BY_NAME[tool](namespace="team-a", **CALLS[tool])

    assert result["error_code"] == "PERMISSION_DENIED"
    assert "experiment lookup failed" in result["error"]


@pytest.mark.parametrize("tool", RUN_TOOLS)
def test_missing_experiment_id_fails_closed(kfp_client, tool):
    """Runs without experiment_id must be refused (fail-closed)."""

    run = make_run("FAILED")
    run.experiment_id = None
    kfp_client.get_run.return_value = run

    result = TOOLS_BY_NAME[tool](namespace="team-a", **CALLS[tool])

    assert result["error_code"] == "PERMISSION_DENIED"
    assert "no experiment association" in result["error"]


@pytest.mark.parametrize("tool", RUN_TOOLS)
def test_empty_experiment_ns_with_allowlist_fails_closed(kfp_client, monkeypatch, tool):
    """Empty experiment namespace + namespace allowlist → refuse (fail-closed)."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        "kubeflow_mcp.pipelines.api._common.get_allowed_namespaces", lambda: ["team-a"]
    )
    kfp_client.get_run.return_value = make_run("FAILED")
    kfp_client._backend.experiment_api.experiment_service_get_experiment.return_value = (
        SimpleNamespace(experiment_id="e-1", namespace="")
    )

    result = TOOLS_BY_NAME[tool](namespace="team-a", **CALLS[tool])

    assert result["error_code"] == "PERMISSION_DENIED"
    assert "no experiment namespace" in result["error"]


@pytest.mark.parametrize("tool", RUN_TOOLS)
def test_empty_experiment_ns_without_allowlist_succeeds(kfp_client, tool):
    """Empty experiment namespace without allowlist → allow (single-user KFP)."""
    from types import SimpleNamespace

    kfp_client.get_run.return_value = make_run("FAILED")
    kfp_client._backend.experiment_api.experiment_service_get_experiment.return_value = (
        SimpleNamespace(experiment_id="e-1", namespace="")
    )

    result = TOOLS_BY_NAME[tool](namespace="team-a", **CALLS[tool])

    # Should NOT get PERMISSION_DENIED from namespace check.
    assert result.get("error_code") != "PERMISSION_DENIED" or "not in namespace" not in result.get(
        "error", ""
    )
