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

# Katib rejects a searched parameter the template never references, so every
# template here references both SEARCH_SPACE keys.
ARGS = ["--lr=${trialParameters.lr}", "--batch-size=${trialParameters.batch_size}"]
TRIAL_TEMPLATE = {
    "apiVersion": "trainer.kubeflow.org/v1alpha1",
    "kind": "TrainJob",
    "spec": {
        "runtimeRef": {"name": "torch-distributed"},
        "trainer": {"image": "my/trainer:1", "args": ARGS},
    },
}


def _template_for(*names):
    """A TrainJob template referencing exactly the given search parameters."""
    args = [f"--{n}=${{trialParameters.{n}}}" for n in names]
    return {
        **TRIAL_TEMPLATE,
        "spec": {**TRIAL_TEMPLATE["spec"], "trainer": {"image": "my/trainer:1", "args": args}},
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
        params = _create(
            search_space={"momentum": {"min": 0.5, "max": 0.99}},
            trial_template=_template_for("momentum"),
        )["config"]["spec"]["parameters"]
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
        params = _create(
            search_space={"layers": {"min": 2, "max": 8, "type": "int"}},
            trial_template=_template_for("layers"),
        )["config"]["spec"]["parameters"]

    assert params[0]["parameterType"] == "int"
    # Bounds keep the SDK's own string encoding.
    assert params[0]["feasibleSpace"]["min"] == "2"
    assert params[0]["feasibleSpace"]["max"] == "8"


def test_int_parameter_accepts_step():
    with cluster():
        params = _create(
            search_space={"layers": {"min": 2, "max": 8, "type": "int", "step": 2}},
            trial_template=_template_for("layers"),
        )["config"]["spec"]["parameters"]
    assert params[0]["feasibleSpace"]["step"] == "2"


@pytest.mark.parametrize(
    ("space", "message"),
    [
        ({"min": 2.5, "max": 8}, "search_space['layers'].min must be an integer"),
        ({"min": 2, "max": "8"}, "search_space['layers'].max must be an integer"),
        ({"min": 8, "max": 2}, "search_space['layers'].min (8) must not exceed max (2)"),
        ({"min": 2, "max": 8, "step": 0}, "search_space['layers'].step must be a positive"),
        ({"min": 2, "max": 8, "step": 1.5}, "search_space['layers'].step must be a positive"),
    ],
)
def test_int_parameter_rejects_bad_bounds_and_steps(space, message):
    with cluster() as api:
        result = _create(
            search_space={"layers": {**space, "type": "int"}},
            trial_template=_template_for("layers"),
        )
    assert result["error_code"] == "VALIDATION_ERROR"
    assert message in result["error"]
    api.create_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize("distribution", ["uniform", "loguniform"])
@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ({"min": "bad", "max": 0.1}, "search_space['lr'].min must be a number"),
        ({"min": 0.01, "max": None}, "search_space['lr'].max must be a number"),
        ({"min": True, "max": 0.1}, "search_space['lr'].min must be a number"),
        ({"min": 0.01, "max": float("inf")}, "search_space['lr'].max must be a number"),
        ({"min": 0.1, "max": 0.01}, "search_space['lr'].min (0.1) must not exceed max (0.01)"),
    ],
)
def test_continuous_range_needs_numeric_bounds_in_order(distribution, bounds, message):
    """The SDK only stringifies these and Katib only checks they are set."""
    with cluster() as api:
        result = _create(
            search_space={"lr": {**bounds, "type": distribution}},
            trial_template=_template_for("lr"),
        )
    assert result["error_code"] == "VALIDATION_ERROR"
    assert message in result["error"]
    api.create_namespaced_custom_object.assert_not_called()


def test_continuous_range_accepts_integer_bounds():
    with cluster():
        result = _create(
            search_space={"lr": {"min": 1, "max": 2}}, trial_template=_template_for("lr")
        )
    assert result["status"] == "preview"


# ─── primaryContainerName resolution ───────────────────────────────────────

JOB_TEMPLATE = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "spec": {
        "template": {
            "spec": {
                "containers": [
                    {"name": "training-container", "image": "katib/random:1", "args": ARGS}
                ]
            }
        }
    },
}


def test_primary_container_follows_a_job_template():
    """Katib rejects every trial pod without this container, so hardcoding
    Trainer's "node" would keep a plain batch/v1 Job from ever starting."""
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
        "kind": "Job",
        "spec": {
            "template": {
                "spec": {"containers": [{"name": "istio-proxy"}, {"name": "node", "args": ARGS}]}
            }
        },
    }
    with cluster():
        spec = _create(trial_template=template)["config"]["spec"]
    assert spec["trialTemplate"]["primaryContainerName"] == "node"


LR_ONLY = {"lr": {"min": 1, "max": 2}}


def _replica(*containers, image="my/trainer:1"):
    return {"template": {"spec": {"containers": [{"name": c, "image": image} for c in containers]}}}


def _training_job(kind, replicas_key, primary, primary_containers, worker_containers):
    """A training-operator job listing its Worker replica before the primary one."""
    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": kind,
        "spec": {
            replicas_key: {
                "Worker": _replica(*worker_containers),
                primary: _replica(*primary_containers, image="my/trainer:${trialParameters.lr}"),
            }
        },
    }


TRAINING_JOBS = [
    ("PyTorchJob", "pytorchReplicaSpecs", "Master"),
    ("TFJob", "tfReplicaSpecs", "Chief"),
    ("XGBoostJob", "xgbReplicaSpecs", "Master"),
    ("MPIJob", "mpiReplicaSpecs", "Launcher"),
]


@pytest.mark.parametrize(("kind", "replicas_key", "primary"), TRAINING_JOBS)
def test_primary_container_comes_from_the_primary_replica(kind, replicas_key, primary):
    """Katib's default primaryPodLabels pick the job-role=master pod: the
    Master, Chief or Launcher replica, whatever order the replicas are listed in.
    Its container order decides between names every replica has."""
    template = _training_job(kind, replicas_key, primary, ("main", "sidecar"), ("sidecar", "main"))
    with cluster():
        spec = _create(search_space=LR_ONLY, trial_template=template)["config"]["spec"]
    assert spec["trialTemplate"]["primaryContainerName"] == "main"


@pytest.mark.parametrize(("kind", "replicas_key", "primary"), TRAINING_JOBS)
def test_a_replica_without_the_primary_container_is_rejected(kind, replicas_key, primary):
    """Katib sets KATIB_TRIAL_NAME on the primary container of every trial pod,
    workers included, and rejects a pod without one, so no trial would start."""
    template = _training_job(kind, replicas_key, primary, ("main",), ("worker-main",))
    with cluster() as api:
        result = _create(search_space=LR_ONLY, trial_template=template, confirmed=True)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"].startswith(
        f"Container 'main' is missing from trial_template.spec.{replicas_key}.Worker"
        ".template.spec.containers (has ['worker-main'])"
    )
    api.create_namespaced_custom_object.assert_not_called()


def test_the_default_skips_a_container_the_workers_lack():
    template = _training_job(
        "PyTorchJob", "pytorchReplicaSpecs", "Master", ("debug", "pytorch"), ("pytorch",)
    )
    with cluster():
        spec = _create(search_space=LR_ONLY, trial_template=template)["config"]["spec"]
    assert spec["trialTemplate"]["primaryContainerName"] == "pytorch"


def test_a_pytorchjob_does_not_default_to_node():
    template = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": _replica("pytorch", image="my/torch:${trialParameters.lr}"),
                "Worker": _replica("pytorch"),
            }
        },
    }
    with cluster():
        spec = _create(search_space={"lr": {"min": 1, "max": 2}}, trial_template=template)[
            "config"
        ]["spec"]
    assert spec["trialTemplate"]["primaryContainerName"] == "pytorch"


def test_explicit_primary_container_wins():
    template = _training_job(
        "PyTorchJob", "pytorchReplicaSpecs", "Master", ("main", "sidecar"), ("sidecar", "main")
    )
    with cluster():
        spec = _create(
            search_space=LR_ONLY, trial_template=template, primary_container_name="sidecar"
        )["config"]["spec"]
    assert spec["trialTemplate"]["primaryContainerName"] == "sidecar"


@pytest.mark.parametrize(
    ("template", "search_space", "path"),
    [
        (JOB_TEMPLATE, SEARCH_SPACE, "trial_template.spec.template.spec"),
        (
            _training_job(
                "PyTorchJob", "pytorchReplicaSpecs", "Master", ("main", "sidecar"), ("main",)
            ),
            LR_ONLY,
            "trial_template.spec.pytorchReplicaSpecs.Worker.template.spec",
        ),
    ],
)
def test_an_explicit_primary_container_is_checked_against_every_replica(
    template, search_space, path
):
    with cluster() as api:
        result = _create(
            search_space=search_space,
            trial_template=template,
            primary_container_name="sidecar",
            confirmed=True,
        )
    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"].startswith(f"Container 'sidecar' is missing from {path}.containers")
    assert "primary_container_name" in result["hint"]
    api.create_namespaced_custom_object.assert_not_called()


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
    assert "delete_hpo_experiment" in result["hint"]
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


# ─── create_hpo_experiment_from_spec ───────────────────────────────────────────


def _manifest(**meta):
    return {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {"name": "from-spec", **meta},
        "spec": {
            "objective": {"objectiveMetricName": "loss", "type": "minimize"},
            "algorithm": {"algorithmName": "cmaes"},
            "maxTrialCount": 6,
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
        result = opt.create_hpo_experiment_from_spec(spec=_manifest())
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
        result = opt.create_hpo_experiment_from_spec(spec=manifest)
    assert result["config"]["spec"]["earlyStopping"] == {"algorithmName": "medianstop"}


def test_from_spec_confirmed_creates():
    with cluster() as api:
        result = opt.create_hpo_experiment_from_spec(spec=_manifest(), confirmed=True)
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
        result = opt.create_hpo_experiment_from_spec(spec=spec)
    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"


def test_from_spec_explicit_namespace_wins():
    with cluster():
        with patch(f"{_UTILS}.get_optimizer_effective_namespace", side_effect=lambda ns: ns or "d"):
            result = opt.create_hpo_experiment_from_spec(
                spec=_manifest(namespace="in-spec"), namespace="explicit"
            )
    assert result["config"]["metadata"]["namespace"] == "explicit"


# ─── Trial image validation ─────────────────────────────────────────────────


def _trainjob(image):
    return {
        **TRIAL_TEMPLATE,
        "spec": {**TRIAL_TEMPLATE["spec"], "trainer": {"image": image, "args": ARGS}},
    }


def _job(image, init_image="busybox:1.36"):
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "spec": {
            "template": {
                "spec": {
                    "initContainers": [{"name": "setup", "image": init_image}],
                    "containers": [{"name": "train", "image": image, "args": ARGS}],
                }
            }
        },
    }


@pytest.mark.parametrize(
    ("template", "field"),
    [
        (_trainjob("My/Trainer:1"), "trial_template.spec.trainer.image"),
        (_trainjob("my trainer:1"), "trial_template.spec.trainer.image"),
        (_job("bad//image"), "trial_template.spec.template.spec.containers[0].image"),
        (_job("ok:1", init_image=""), "trial_template.spec.template.spec.initContainers[0].image"),
        (
            {"kind": "Pod", "spec": {"containers": [{"name": "c", "image": "img@sha256:short"}]}},
            "trial_template.spec.containers[0].image",
        ),
        (
            {
                "kind": "PyTorchJob",
                "spec": {
                    "pytorchReplicaSpecs": {
                        "Master": _replica("pytorch"),
                        "Worker": _replica("pytorch", image="Bad/Image:1"),
                    }
                },
            },
            "trial_template.spec.pytorchReplicaSpecs.Worker.template.spec.containers[0].image",
        ),
        (_trainjob("registry.example.com:99999/trainer:1"), "trial_template.spec.trainer.image"),
    ],
)
def test_malformed_trial_image_is_rejected_before_any_call(template, field):
    with cluster() as api:
        result = _create(trial_template=template, confirmed=True)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"].startswith(field)
    api.create_namespaced_custom_object.assert_not_called()


def test_trial_image_rejected_before_namespace_policy():
    """Input validation runs before the namespace check (mutation-safety order)."""
    with patch("kubeflow_mcp.optimizer.api.optimization.check_optimizer_namespace") as ns_check:
        result = _create(trial_template=_trainjob("Bad Image"))
    assert result["error_code"] == "VALIDATION_ERROR"
    ns_check.assert_not_called()


@pytest.mark.parametrize(
    "image",
    [
        "my/trainer:${trialParameters.tag}",
        "${trialParameters.image}",
        "registry.example.com:${trialParameters.port}/trainer:1",
        "registry.example.com:5000/team/trainer:v2",
        "localhost:5000/a/b@sha256:" + "a" * 64,
    ],
)
def test_valid_or_templated_trial_image_is_accepted(image):
    with cluster():
        result = _create(trial_template=_trainjob(image))
    assert result["status"] == "preview"


def test_rejected_image_reports_the_original_value():
    with cluster():
        result = _create(trial_template=_trainjob("Bad/${trialParameters.tag}"))
    assert result["details"] == {"value": "Bad/${trialParameters.tag}"}


def test_from_spec_rejects_a_malformed_trial_image():
    manifest = _manifest()
    manifest["spec"]["trialTemplate"] = {"trialSpec": _job("UPPER/case:1")}
    with cluster() as api:
        result = opt.create_hpo_experiment_from_spec(spec=manifest, confirmed=True)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"].startswith(
        "spec.spec.trialTemplate.trialSpec.spec.template.spec.containers[0].image"
    )
    api.create_namespaced_custom_object.assert_not_called()


def test_from_spec_configmap_template_skips_image_check():
    manifest = _manifest()
    manifest["spec"]["trialTemplate"] = {
        "configMap": {"configMapName": "tpl", "configMapNamespace": "kubeflow"}
    }
    with cluster():
        result = opt.create_hpo_experiment_from_spec(spec=manifest)
    assert result["status"] == "preview"


# ─── Budget and manifest validation ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_trial_count": 4, "parallel_trial_count": 5}, "must not exceed max_trial_count"),
        ({"max_trial_count": 4, "max_failed_trials": 5}, "max_failed_trials must be between 0"),
        ({"max_failed_trials": -1}, "max_failed_trials must be between 0"),
    ],
)
def test_trial_budget_katib_would_reject_is_refused(overrides, message):
    with cluster() as api:
        result = _create(confirmed=True, **overrides)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert message in result["error"]
    api.create_namespaced_custom_object.assert_not_called()


def test_parallelism_equal_to_the_budget_is_valid():
    with cluster():
        result = _create(max_trial_count=4, parallel_trial_count=4, max_failed_trials=4)
    assert result["status"] == "preview"


def _without(path):
    manifest = _manifest()
    node = manifest
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return manifest


@pytest.mark.parametrize(
    ("manifest", "field"),
    [
        (_without(("apiVersion",)), "spec.apiVersion"),
        (_without(("kind",)), "spec.kind"),
        (_without(("spec",)), "spec.spec must be an object"),
        (_without(("spec", "objective")), "spec.spec.objective.type"),
        (_without(("spec", "objective", "objectiveMetricName")), "objectiveMetricName"),
        (_without(("spec", "algorithm")), "spec.spec.algorithm.algorithmName"),
        (_without(("spec", "trialTemplate")), "spec.spec.trialTemplate"),
        (_without(("spec", "parameters")), "spec.spec.parameters"),
        ({**_manifest(), "metadata": "exp"}, "spec.metadata must be an object"),
    ],
)
def test_from_spec_names_the_missing_field_before_the_namespace_check(manifest, field):
    with (
        cluster() as api,
        patch("kubeflow_mcp.optimizer.api.optimization.check_optimizer_namespace") as ns_check,
    ):
        result = opt.create_hpo_experiment_from_spec(spec=manifest, confirmed=True)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert field in result["error"]
    ns_check.assert_not_called()
    api.create_namespaced_custom_object.assert_not_called()


# ─── Templates Katib can run ───────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["Pod", "RayJob", None])
def test_kinds_katib_has_no_completion_conditions_for_are_rejected(kind):
    """Katib requires successCondition/failureCondition and only fills them in for
    Job and the Kubeflow job kinds; the flat tool cannot pass them."""
    template = {"kind": kind, "spec": {"containers": [{"name": "c", "args": ARGS}]}}
    with cluster() as api:
        result = _create(trial_template=template, confirmed=True)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "create_hpo_experiment_from_spec" in result["error"]
    api.create_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize(
    ("spec", "path"),
    [
        ("not-an-object", "trial_template.spec"),
        ({"template": ["x"]}, "trial_template.spec.template"),
        ({"template": {"spec": "containers"}}, "trial_template.spec.template.spec"),
    ],
)
def test_a_malformed_nested_template_is_a_validation_error(spec, path):
    """Regression: a string spec.template.spec raised an uncaught AttributeError."""
    with cluster() as api:
        result = _create(trial_template={"kind": "Job", "spec": spec}, confirmed=True)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"].startswith(f"{path} must be an object")
    api.create_namespaced_custom_object.assert_not_called()


def test_a_trial_template_that_is_not_an_object_is_rejected():
    with cluster():
        result = _create(trial_template=["TrainJob"])
    assert result["error_code"] == "VALIDATION_ERROR"


def test_a_searched_parameter_the_template_never_uses_is_rejected():
    """Katib rejects the experiment at admission, so the preview must not pass."""
    with cluster() as api:
        result = _create(trial_template=_template_for("lr"), confirmed=True)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "['batch_size']" in result["error"]
    api.create_namespaced_custom_object.assert_not_called()


def test_a_parameterized_image_digest_is_accepted():
    """Each digest choice is a valid hash, so the placeholder must not fail the check."""
    space = {**SEARCH_SPACE, "digest": {"choices": ["a" * 64, "b" * 64]}}
    template = _trainjob("my/trainer@sha256:${trialParameters.digest}")
    with cluster():
        result = _create(search_space=space, trial_template=template)
    assert result["status"] == "preview", result


def test_hyperband_is_not_offered_by_the_flat_tool():
    """Katib's Hyperband needs r_l and resource_name settings this tool cannot pass."""
    with cluster():
        result = _create(algorithm="hyperband")
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "hyperband" not in opt.ALGORITHMS


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_trial_count": 2, "parallel_trial_count": 3},
        {"search_space": {"lr": {"min": 0.1, "max": 0.01}}, "trial_template": _template_for("lr")},
    ],
)
def test_input_is_validated_before_the_namespace_policy(overrides):
    """A bad request gets the same error whatever namespace it names."""
    with patch("kubeflow_mcp.optimizer.api.optimization.check_optimizer_namespace") as ns_check:
        result = _create(**overrides)
    assert result["error_code"] == "VALIDATION_ERROR"
    ns_check.assert_not_called()


@pytest.mark.parametrize("name", ["1-hpo", "a" * 41])
def test_names_katib_refuses_are_rejected_before_the_preview(name):
    """Katib wants a leading letter and at most 40 characters; Kubernetes allows both."""
    with cluster() as api:
        result = _create(name=name)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "name must start with a letter and be at most 40 characters" in result["error"]
    api.create_namespaced_custom_object.assert_not_called()


def test_a_forty_character_name_is_accepted():
    with cluster():
        assert _create(name="a" * 40)["status"] == "preview"


def test_from_spec_applies_katib_name_rules():
    manifest = _manifest()
    manifest["metadata"]["name"] = "1-hpo"
    with cluster():
        result = opt.create_hpo_experiment_from_spec(spec=manifest)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"].startswith("spec.metadata.name must start with a letter")


def _manifest_running(trial_spec, primary_container):
    manifest = _manifest()
    manifest["spec"]["trialTemplate"] = {
        "primaryContainerName": primary_container,
        "trialSpec": trial_spec,
    }
    return manifest


def test_from_spec_checks_the_primary_container_against_every_replica():
    template = _training_job("PyTorchJob", "pytorchReplicaSpecs", "Master", ("main",), ("worker",))
    with cluster() as api:
        result = opt.create_hpo_experiment_from_spec(
            spec=_manifest_running(template, "main"), confirmed=True
        )
    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"].startswith(
        "Container 'main' is missing from spec.spec.trialTemplate.trialSpec.spec"
        ".pytorchReplicaSpecs.Worker.template.spec.containers (has ['worker'])"
    )
    assert "primaryContainerName" in result["hint"]
    api.create_namespaced_custom_object.assert_not_called()


def test_from_spec_accepts_a_primary_container_every_replica_has():
    template = _training_job("PyTorchJob", "pytorchReplicaSpecs", "Master", ("main",), ("main",))
    with cluster():
        result = opt.create_hpo_experiment_from_spec(spec=_manifest_running(template, "main"))
    assert result["status"] == "preview"


@pytest.mark.parametrize(
    "call",
    [
        lambda: _create(confirmed=True),
        lambda: opt.create_hpo_experiment_from_spec(spec=_manifest(), confirmed=True),
    ],
    ids=["flat", "from_spec"],
)
def test_an_unresolvable_namespace_is_a_structured_error(call):
    """Without an allowlist the default is first resolved here, and can fail."""
    api = MagicMock()
    with (
        patch(
            f"{_UTILS}.get_optimizer_effective_namespace",
            side_effect=RuntimeError("Cannot determine default namespace: no kubeconfig found"),
        ),
        patch(f"{_UTILS}.get_custom_objects_api", return_value=api),
    ):
        result = call()
    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"
    assert "Cannot determine default namespace" in result["error"]
    api.create_namespaced_custom_object.assert_not_called()


# ─── create_hpo_experiment_from_spec: names, caps and completion conditions ─


def test_from_spec_rejects_a_non_string_name():
    manifest = _manifest()
    manifest["metadata"]["name"] = 123
    with cluster():
        result = opt.create_hpo_experiment_from_spec(spec=manifest)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "must be a string" in result["error"]


@pytest.mark.parametrize(
    ("field", "value", "limit"),
    [("maxTrialCount", 1_000_000, opt.MAX_TRIAL_COUNT_LIMIT), ("parallelTrialCount", 10_000, 100)],
)
def test_from_spec_applies_the_same_trial_caps(field, value, limit):
    manifest = _manifest()
    manifest["spec"][field] = value
    with cluster() as api:
        result = opt.create_hpo_experiment_from_spec(spec=manifest, confirmed=True)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert f"spec.spec.{field} must be at most {limit}" in result["error"]
    api.create_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize("budget", ["omitted", None])
def test_from_spec_requires_a_trial_budget(budget):
    """Katib runs trials with no limit when maxTrialCount is not set."""
    manifest = _manifest()
    if budget == "omitted":
        del manifest["spec"]["maxTrialCount"]
    else:
        manifest["spec"]["maxTrialCount"] = budget
    with cluster() as api:
        result = opt.create_hpo_experiment_from_spec(spec=manifest)
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "spec.spec.maxTrialCount is required" in result["error"]
    api.create_namespaced_custom_object.assert_not_called()


def test_from_spec_needs_completion_conditions_for_a_pod_trial():
    pod = {"apiVersion": "v1", "kind": "Pod", "spec": {"containers": [{"name": "c"}]}}
    manifest = _manifest()
    manifest["spec"]["trialTemplate"] = {"trialSpec": pod}
    with cluster():
        refused = opt.create_hpo_experiment_from_spec(spec=manifest)
        manifest["spec"]["trialTemplate"].update(
            successCondition="status.phase == 'Succeeded'",
            failureCondition="status.phase == 'Failed'",
        )
        accepted = opt.create_hpo_experiment_from_spec(spec=manifest)
    assert refused["error_code"] == "VALIDATION_ERROR"
    assert "successCondition and failureCondition" in refused["error"]
    assert accepted["status"] == "preview"


def test_from_spec_still_allows_hyperband_with_its_settings():
    manifest = _manifest()
    manifest["spec"]["algorithm"] = {
        "algorithmName": "hyperband",
        "algorithmSettings": [
            {"name": "resource_name", "value": "lr"},
            {"name": "r_l", "value": "9"},
        ],
    }
    with cluster():
        result = opt.create_hpo_experiment_from_spec(spec=manifest)
    assert result["status"] == "preview"
