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

"""Namespace-allowlist enforcement for optimizer tools.

Regression guard: the policy check must resolve an omitted namespace through the
*optimizer* client. Resolving it through the trainer client validates a
different namespace than the one the operation targets, which both let the
allowlist be bypassed and broke optimizer-only deployments.
"""

from unittest.mock import MagicMock, patch

import pytest

from kubeflow_mcp.core.security import check_namespace_allowed
from kubeflow_mcp.optimizer.api import discovery, lifecycle, monitoring, optimization
from kubeflow_mcp.optimizer.api._common import check_optimizer_namespace

_POLICY = "kubeflow_mcp.core.policy.get_allowed_namespaces"
_TRAINER_NS = "kubeflow_mcp.common.utils.get_trainer_effective_namespace"
_OPTIMIZER_NS = "kubeflow_mcp.common.utils.get_optimizer_effective_namespace"


def test_bypass_is_closed_when_clients_disagree():
    """Trainer default is allowed, optimizer default is not — must deny."""
    with (
        patch(_POLICY, return_value=["team-a"]),
        patch(_TRAINER_NS, return_value="team-a"),
        patch(_OPTIMIZER_NS, return_value="team-b"),
    ):
        err = check_optimizer_namespace(None)
    assert err is not None
    assert err.error_code == "PERMISSION_DENIED"
    assert "team-b" in err.error


def test_allows_when_optimizer_default_is_permitted():
    with (
        patch(_POLICY, return_value=["team-a"]),
        patch(_TRAINER_NS, return_value="team-b"),
        patch(_OPTIMIZER_NS, return_value="team-a"),
    ):
        assert check_optimizer_namespace(None) is None


def test_works_without_a_trainer_client():
    """Optimizer-only servers must not be denied just because TrainerClient
    cannot be constructed."""
    with (
        patch(_POLICY, return_value=["team-a"]),
        patch(_TRAINER_NS, side_effect=RuntimeError("trainer not available")),
        patch(_OPTIMIZER_NS, return_value="team-a"),
    ):
        assert check_optimizer_namespace(None) is None


def test_fails_closed_when_namespace_cannot_be_resolved():
    with (
        patch(_POLICY, return_value=["team-a"]),
        patch(_OPTIMIZER_NS, side_effect=RuntimeError("no cluster")),
    ):
        err = check_optimizer_namespace(None)
    assert err is not None
    assert err.error_code == "PERMISSION_DENIED"


def test_explicit_namespace_is_checked_directly():
    with patch(_POLICY, return_value=["team-a"]):
        assert check_optimizer_namespace("team-a") is None
        assert check_optimizer_namespace("team-b").error_code == "PERMISSION_DENIED"


def test_no_allowlist_permits_everything():
    with patch(_POLICY, return_value=None):
        assert check_optimizer_namespace(None) is None
        assert check_optimizer_namespace("anything") is None


def test_spec_namespace_cannot_bypass_the_allowlist():
    """create_experiment_from_spec must check spec.metadata.namespace, not the
    default, when no namespace argument is given."""
    api = MagicMock()
    spec = {"metadata": {"name": "exp", "namespace": "team-b"}}
    with (
        patch(_POLICY, return_value=["team-a"]),
        patch(_OPTIMIZER_NS, side_effect=lambda ns: ns or "team-a"),
        patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=api),
    ):
        result = optimization.create_experiment_from_spec(spec=spec, confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == "PERMISSION_DENIED"
    assert "team-b" in result["error"]
    api.assert_not_called()


def test_namespace_argument_overrides_the_spec_namespace():
    spec = {"metadata": {"name": "exp", "namespace": "team-b"}}
    with (
        patch(_POLICY, return_value=["team-a"]),
        patch(_OPTIMIZER_NS, side_effect=lambda ns: ns or "team-a"),
    ):
        result = optimization.create_experiment_from_spec(spec=spec, namespace="team-a")

    assert result["status"] == "preview"
    assert result["config"]["metadata"]["namespace"] == "team-a"


def test_non_string_spec_namespace_is_rejected():
    spec = {"metadata": {"name": "exp", "namespace": 7}}
    with patch(_POLICY, return_value=None):
        result = optimization.create_experiment_from_spec(spec=spec)

    assert result["error_code"] == "VALIDATION_ERROR"


def test_generic_helper_still_defaults_to_trainer():
    """Trainer callers must keep their existing behaviour."""
    with (
        patch(_POLICY, return_value=["team-a"]),
        patch(_TRAINER_NS, return_value="team-a"),
    ):
        assert check_namespace_allowed(None) is None


# Minimal valid arguments per tool. Keyed by name so the sweep below can be
# driven from the module's own TOOLS list rather than a parallel copy of it.
_TOOL_KWARGS: dict[str, dict] = {
    "list_experiments": {},
    "get_experiment": {"name": "exp"},
    "get_experiment_status": {"name": "exp"},
    "get_trial": {"name": "exp", "trial": "t1"},
    "get_successful_trials": {"name": "exp"},
    "list_suggestions": {},
    "get_experiment_trials": {"name": "exp"},
    "get_best_trial": {"name": "exp"},
    "get_suggestion": {"name": "exp"},
    "wait_for_experiment": {"name": "exp"},
    "get_experiment_trial_logs": {"name": "exp"},
    "get_experiment_events": {"name": "exp"},
    "delete_experiment": {"name": "exp", "confirmed": True},
    "update_experiment": {"name": "exp", "parallel_trial_count": 1, "confirmed": True},
    "create_hpo_experiment": {
        "name": "exp",
        "objective_metric": "acc",
        "search_space": {"lr": {"min": 0.1, "max": 1}},
        "trial_template": {"kind": "TrainJob"},
        "confirmed": True,
    },
    "create_experiment_from_spec": {
        "spec": {"metadata": {"name": "exp"}},
        "confirmed": True,
    },
}

# katib_pre_flight inspects cluster-scoped resources and takes no namespace,
# so it is the one tool with nothing to enforce.
_NO_NAMESPACE = {"katib_pre_flight"}


def test_every_tool_is_covered_by_the_sweep():
    """Fail when a new tool is added without namespace coverage.

    The sweep below used to be a hand-written list, which silently stayed green
    as tools were added. This binds it to the module's own TOOLS export so a new
    tool must either be exercised or explicitly exempted.
    """
    from kubeflow_mcp.optimizer import TOOLS

    covered = set(_TOOL_KWARGS) | _NO_NAMESPACE
    missing = {t.__name__ for t in TOOLS} - covered
    assert not missing, f"tools missing namespace-policy coverage: {sorted(missing)}"


@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [
        pytest.param(tool, _TOOL_KWARGS[tool.__name__], id=tool.__name__)
        for module in (discovery, monitoring, lifecycle, optimization)
        for tool in vars(module).values()
        if callable(tool) and getattr(tool, "__name__", None) in _TOOL_KWARGS
    ],
)
def test_every_tool_enforces_the_allowlist(tool, kwargs):
    """No optimizer tool may reach the cluster when the resolved namespace is
    outside the allowlist."""
    client = MagicMock()
    api = MagicMock()
    with (
        patch(_POLICY, return_value=["team-a"]),
        patch(_TRAINER_NS, return_value="team-a"),  # would pass the old check
        patch(_OPTIMIZER_NS, return_value="team-b"),  # but this is the real target
        patch("kubeflow_mcp.common.utils.get_optimizer_client_for_namespace", return_value=client),
        patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=api),
        patch(
            "kubeflow_mcp.optimizer.api.discovery.get_optimizer_client_for_namespace",
            return_value=client,
        ),
        patch("kubeflow_mcp.optimizer.api.discovery.get_custom_objects_api", return_value=api),
        patch(
            "kubeflow_mcp.optimizer.api.monitoring.get_optimizer_client_for_namespace",
            return_value=client,
        ),
        patch("kubeflow_mcp.optimizer.api.monitoring.get_custom_objects_api", return_value=api),
    ):
        result = tool(**kwargs)

    assert result["success"] is False, f"{tool.__name__} did not deny"
    assert result["error_code"] == "PERMISSION_DENIED", f"{tool.__name__} wrong error"
    client.assert_not_called()
    api.assert_not_called()


@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [
        pytest.param(tool, _TOOL_KWARGS[tool.__name__], id=tool.__name__)
        for module in (discovery, monitoring, lifecycle, optimization)
        for tool in vars(module).values()
        if callable(tool) and getattr(tool, "__name__", None) in _TOOL_KWARGS
    ],
)
def test_malformed_namespace_is_rejected_before_any_api_request(tool, kwargs):
    """Mirrors the trainer sweep: a malformed namespace never reaches the API."""
    with (
        patch(_POLICY, return_value=None),
        patch("kubernetes.config.load_kube_config"),
        patch("kubernetes.config.load_incluster_config"),
        patch("kubernetes.client.ApiClient.call_api") as call_api,
    ):
        result = tool(**kwargs, namespace="Bad_NS!")

    assert result["error_code"] == "VALIDATION_ERROR", result
    assert "namespace" in result["error"]
    call_api.assert_not_called()
