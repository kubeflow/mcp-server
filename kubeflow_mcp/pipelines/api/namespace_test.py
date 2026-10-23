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
