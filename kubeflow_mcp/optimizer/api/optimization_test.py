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

"""Unit tests for optimizer creation tools (CR construction + two-phase confirm)."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from kubeflow_mcp.common.utils import MCP_MANAGED_LABEL, MCP_MANAGED_VALUE
from kubeflow_mcp.optimizer.api import optimization as opt

_UTILS = "kubeflow_mcp.common.utils"

TRIAL_TEMPLATE = {
    "apiVersion": "trainer.kubeflow.org/v1alpha1",
    "kind": "TrainJob",
    "spec": {
        "runtimeRef": {"name": "torch-distributed"},
        "trainer": {"image": "my/trainer:1", "args": ["--lr=${trialParameters.lr}"]},
    },
}
SEARCH_SPACE = {
    "lr": {"min": 0.001, "max": 0.1, "type": "loguniform"},
    "batch_size": {"choices": [16, 32, 64]},
}


@contextmanager
def cluster(api: MagicMock | None = None):
    """Patch namespace resolution and the CustomObjects API."""
    api = api or MagicMock()
    with (
        patch(f"{_UTILS}.get_optimizer_effective_namespace", return_value="kubeflow"),
        patch(f"{_UTILS}.get_custom_objects_api", return_value=api),
    ):
        yield api


def _create(**overrides):
    args = {
        "name": "demo-hpo",
        "objective_metric": "accuracy",
        "search_space": SEARCH_SPACE,
        "trial_template": TRIAL_TEMPLATE,
    }
    args.update(overrides)
    return opt.create_hpo_experiment(**args)


# ─── create_hpo_experiment: preview / CR construction ──────────────────────


def test_preview_does_not_create():
    with cluster() as api:
        result = _create()
    assert result["status"] == "preview"
    api.create_namespaced_custom_object.assert_not_called()


def test_preview_builds_expected_cr():
    with cluster():
        spec = _create(algorithm="tpe", max_trial_count=6, parallel_trial_count=3)["config"]["spec"]

    assert spec["algorithm"] == {"algorithmName": "tpe"}
    assert spec["objective"] == {"objectiveMetricName": "accuracy", "type": "maximize"}
    assert spec["maxTrialCount"] == 6
    assert spec["parallelTrialCount"] == 3
    # Every search-space key must be referenced as a trial parameter.
    assert [p["name"] for p in spec["trialTemplate"]["trialParameters"]] == ["lr", "batch_size"]
    assert spec["trialTemplate"]["trialSpec"] == TRIAL_TEMPLATE


def test_search_space_encoding_matches_sdk():
    with cluster():
        params = _create()["config"]["spec"]["parameters"]

    by_name = {p["name"]: p for p in params}
    assert by_name["lr"]["parameterType"] == "double"
    assert by_name["lr"]["feasibleSpace"]["distribution"] == "logUniform"
    assert by_name["batch_size"]["parameterType"] == "categorical"
    # Katib requires categorical values as strings.
    assert by_name["batch_size"]["feasibleSpace"]["list"] == ["16", "32", "64"]


def test_uniform_is_the_default_distribution():
    with cluster():
        params = _create(search_space={"momentum": {"min": 0.5, "max": 0.99}})["config"]["spec"][
            "parameters"
        ]
    assert params[0]["feasibleSpace"]["distribution"] == "uniform"


def test_created_experiment_carries_mcp_ownership_label():
    with cluster():
        labels = _create()["config"]["metadata"]["labels"]
    assert labels[MCP_MANAGED_LABEL] == MCP_MANAGED_VALUE


def test_confirmed_creates_experiment():
    with cluster() as api:
        result = _create(confirmed=True)

    assert result["success"] is True
    assert result["data"]["created"] is True
    kwargs = api.create_namespaced_custom_object.call_args.kwargs
    assert kwargs["plural"] == "experiments"
    assert kwargs["namespace"] == "kubeflow"
    assert kwargs["body"]["metadata"]["name"] == "demo-hpo"


def test_caller_supplied_name_is_honoured():
    """optimize() would generate its own name; building the CR must not."""
    with cluster():
        config = _create(name="my-chosen-name")["config"]
    assert config["metadata"]["name"] == "my-chosen-name"


# ─── create_hpo_experiment: validation ─────────────────────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "Invalid_Name!"},
        {"algorithm": "not-an-algorithm"},
        {"objective_type": "max"},
        {"search_space": {}},
        {"trial_template": {}},
        {"max_trial_count": 0},
        {"max_trial_count": opt.MAX_TRIAL_COUNT_LIMIT + 1},
        {"parallel_trial_count": 0},
        {"parallel_trial_count": opt.MAX_PARALLEL_TRIAL_LIMIT + 1},
        {"search_space": {"lr": {"min": 0.1}}},
        {"search_space": {"lr": {"choices": []}}},
        {"search_space": {"lr": {"min": 0, "max": 1, "type": "normal"}}},
        {"search_space": {"lr": "not-a-dict"}},
    ],
)
def test_invalid_input_is_rejected(overrides):
    with cluster() as api:
        result = _create(**overrides)
    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    api.create_namespaced_custom_object.assert_not_called()


def test_all_advertised_algorithms_are_accepted():
    """Descriptions promise these algorithms; each must build a valid CR."""
    for algorithm in opt.ALGORITHMS:
        with cluster():
            result = _create(algorithm=algorithm)
        assert result["status"] == "preview", algorithm
        assert result["config"]["spec"]["algorithm"]["algorithmName"] == algorithm


# ─── integer parameters ────────────────────────────────────────────────────


def test_int_parameter_is_typed_int_not_double():
    """Katib supports "int"; the SDK's Search helpers only emit double."""
    with cluster():
        params = _create(search_space={"layers": {"min": 2, "max": 8, "type": "int"}})["config"][
            "spec"
        ]["parameters"]

    assert params[0]["parameterType"] == "int"
    # Bounds keep the SDK's own string encoding.
    assert params[0]["feasibleSpace"]["min"] == "2"
    assert params[0]["feasibleSpace"]["max"] == "8"


def test_int_parameter_accepts_step():
    with cluster():
        params = _create(search_space={"layers": {"min": 2, "max": 8, "type": "int", "step": 2}})[
            "config"
        ]["spec"]["parameters"]
    assert params[0]["feasibleSpace"]["step"] == "2"


@pytest.mark.parametrize(
    "space",
    [
        {"layers": {"min": 2.5, "max": 8, "type": "int"}},
        {"layers": {"min": 2, "max": "8", "type": "int"}},
        {"layers": {"min": 2, "max": 8, "type": "int", "step": 0}},
        {"layers": {"min": 2, "max": 8, "type": "int", "step": 1.5}},
    ],
)
def test_int_parameter_rejects_non_integers(space):
    with cluster() as api:
        result = _create(search_space=space)
    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    api.create_namespaced_custom_object.assert_not_called()


# ─── primaryContainerName resolution ───────────────────────────────────────

JOB_TEMPLATE = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "spec": {
        "template": {
            "spec": {"containers": [{"name": "training-container", "image": "katib/random:1"}]}
        }
    },
}


def test_primary_container_follows_a_job_template():
    """Katib collects metrics from this container; hardcoding Trainer's "node"
    would silently collect nothing from a plain batch/v1 Job."""
    with cluster():
        spec = _create(trial_template=JOB_TEMPLATE)["config"]["spec"]
    assert spec["trialTemplate"]["primaryContainerName"] == "training-container"


def test_primary_container_defaults_to_node_for_a_trainjob():
    """A TrainJob has no inline pod spec; Trainer names its container "node"."""
    with cluster():
        spec = _create()["config"]["spec"]
    assert spec["trialTemplate"]["primaryContainerName"] == "node"


def test_primary_container_prefers_node_over_a_sidecar():
    template = {
        "spec": {"template": {"spec": {"containers": [{"name": "istio-proxy"}, {"name": "node"}]}}}
    }
    with cluster():
        spec = _create(trial_template=template)["config"]["spec"]
    assert spec["trialTemplate"]["primaryContainerName"] == "node"


def test_explicit_primary_container_wins():
    with cluster():
        spec = _create(trial_template=JOB_TEMPLATE, primary_container_name="sidecar")["config"][
            "spec"
        ]
    assert spec["trialTemplate"]["primaryContainerName"] == "sidecar"


def test_create_failure_maps_to_sdk_error():
    api = MagicMock()
    api.create_namespaced_custom_object.side_effect = RuntimeError("boom")
    with cluster(api):
        result = _create(confirmed=True)
    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"


def test_duplicate_name_is_actionable_and_does_not_trip_breaker():
    """A 409 is caller-fixable: it must not be reported as an infrastructure
    failure, or the circuit breaker opens on a simple name collision."""
    from kubernetes.client.exceptions import ApiException

    from kubeflow_mcp.common.constants import is_infrastructure_error

    api = MagicMock()
    api.create_namespaced_custom_object.side_effect = ApiException(status=409, reason="Conflict")
    with cluster(api):
        result = _create(confirmed=True)

    assert result["error_code"] == "VALIDATION_ERROR"
    assert "already exists" in result["error"]
    assert "delete_experiment" in result["hint"]
    assert is_infrastructure_error(result) is False


def test_create_uses_the_longer_write_timeout():
    """A create is admitted by two webhooks, each with its own 10s budget.

    Sending the 5s read timeout meant the client gave up before the API server
    could report the admission failure, turning a diagnosable error into an
    opaque "read timed out".
    """
    from kubeflow_mcp.common import utils as mcp_utils

    with cluster() as api:
        _create(confirmed=True)
    timeout = api.create_namespaced_custom_object.call_args.kwargs["_request_timeout"]
    assert timeout == mcp_utils.K8S_WRITE_TIMEOUT
    assert timeout > mcp_utils.K8S_TIMEOUT


def test_client_side_timeout_is_reported_as_timeout_not_opaque_sdk_error():
    from urllib3.exceptions import ReadTimeoutError

    from kubeflow_mcp.common.constants import is_infrastructure_error

    api = MagicMock()
    api.create_namespaced_custom_object.side_effect = ReadTimeoutError(
        None, "/apis", "Read timed out."
    )
    with cluster(api):
        result = _create(confirmed=True)

    assert result["error_code"] == "TIMEOUT"
    assert "Timed out waiting for the API server" in result["error"]
    assert "katib_pre_flight()" in result["hint"]
    # A timeout is still an infrastructure failure, so the breaker should trip.
    assert is_infrastructure_error(result) is True


def test_unreachable_katib_webhook_gets_an_actionable_hint():
    """Observed on a cluster whose katib-controller was Running but not Ready:
    the raw 500 is a wall of HTTP headers that never says what to do."""
    from kubernetes.client.exceptions import ApiException

    from kubeflow_mcp.common.constants import is_infrastructure_error

    api = MagicMock()
    api.create_namespaced_custom_object.side_effect = ApiException(
        status=500,
        reason="Internal Server Error",
        http_resp=MagicMock(
            status=500,
            reason="Internal Server Error",
            data=(
                b'{"message":"Internal error occurred: failed calling webhook '
                b'\\"defaulter.experiment.katib.kubeflow.org\\": no endpoints available '
                b'for service \\"katib-controller\\""}'
            ),
            getheaders=lambda: {},
        ),
    )
    with cluster(api):
        result = _create(confirmed=True)

    assert result["error_code"] == "SDK_ERROR"
    assert "katib_pre_flight()" in result["hint"]
    # A downed control plane is a real outage: the breaker should still trip.
    assert is_infrastructure_error(result) is True


# ─── create_experiment_from_spec ───────────────────────────────────────────


def _manifest(**meta):
    return {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {"name": "from-spec", **meta},
        "spec": {
            "objective": {"objectiveMetricName": "loss", "type": "minimize"},
            "algorithm": {"algorithmName": "cmaes"},
            "parameters": [
                {
                    "name": "lr",
                    "parameterType": "double",
                    "feasibleSpace": {"min": "0.01", "max": "0.1"},
                }
            ],
            "trialTemplate": {"trialSpec": TRIAL_TEMPLATE},
        },
    }


def test_from_spec_preview_stamps_label_and_namespace():
    with cluster() as api:
        result = opt.create_experiment_from_spec(spec=_manifest())
    assert result["status"] == "preview"
    metadata = result["config"]["metadata"]
    assert metadata["namespace"] == "kubeflow"
    assert metadata["labels"][MCP_MANAGED_LABEL] == MCP_MANAGED_VALUE
    api.create_namespaced_custom_object.assert_not_called()


def test_from_spec_preserves_advanced_config():
    """The escape hatch must not strip fields create_hpo_experiment cannot model."""
    manifest = _manifest()
    manifest["spec"]["earlyStopping"] = {"algorithmName": "medianstop"}
    with cluster():
        result = opt.create_experiment_from_spec(spec=manifest)
    assert result["config"]["spec"]["earlyStopping"] == {"algorithmName": "medianstop"}


def test_from_spec_confirmed_creates():
    with cluster() as api:
        result = opt.create_experiment_from_spec(spec=_manifest(), confirmed=True)
    assert result["success"] is True
    assert api.create_namespaced_custom_object.call_args.kwargs["plural"] == "experiments"


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"metadata": {}},
        {"metadata": {"name": "Bad_Name!"}},
    ],
)
def test_from_spec_rejects_malformed_input(spec):
    with cluster():
        result = opt.create_experiment_from_spec(spec=spec)
    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"


def test_from_spec_explicit_namespace_wins():
    with cluster():
        with patch(f"{_UTILS}.get_optimizer_effective_namespace", side_effect=lambda ns: ns or "d"):
            result = opt.create_experiment_from_spec(
                spec=_manifest(namespace="in-spec"), namespace="explicit"
            )
    assert result["config"]["metadata"]["namespace"] == "explicit"
