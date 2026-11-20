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

"""Tests for trainer/api/training.py — fine_tune, run_custom_training, run_container_training."""

from __future__ import annotations

import ast
import inspect
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from kubeflow.trainer.options import Labels, Name
from kubernetes.client.exceptions import ApiException
from tests.common import FAILED, PREVIEW, SDK_ERROR, VALIDATION_ERROR, TestCase, assert_test_case

from kubeflow_mcp.common.utils import MCP_MANAGED_LABEL, MCP_MANAGED_VALUE
from kubeflow_mcp.trainer.api.training import (
    _build_dataset_preprocess_cfg,
    _build_fine_tune_config,
    _build_initializer,
    _build_lora_cfg,
    _build_runtime_patch,
    _check_gpu_available,
    _forwarded_train_args,
    _get_client,
    _has_required_params,
    _has_train_call,
    _inject_ownership_label,
    _inject_trainer_hf_home,
    _make_train_func,
    _sdk_error,
    _should_apply_hf_dataset_workaround,
    _uncalled_train_call,
    _validate_fine_tune_params,
    _validate_resources_per_node,
    fine_tune,
    run_container_training,
    run_custom_training,
)

PATCH_CLIENT = "kubeflow_mcp.trainer.api.training._get_client"
PATCH_NS_CHECK = "kubeflow_mcp.trainer.api.training.check_namespace_allowed"
PATCH_GPU_CHECK = "kubeflow_mcp.trainer.api.training._check_gpu_available"


@pytest.fixture(autouse=True)
def _offline_hub(monkeypatch):
    """fine_tune(confirmed=True) checks hf:// repos on the Hub; keep unit tests offline."""
    monkeypatch.setattr("huggingface_hub.model_info", MagicMock())
    monkeypatch.setattr("huggingface_hub.dataset_info", MagicMock())
    monkeypatch.setattr("huggingface_hub.list_models", MagicMock(return_value=[]))
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = {
        "spec": {
            "template": {
                "spec": {
                    "replicatedJobs": [
                        {
                            "name": "node",
                            "template": {
                                "spec": {
                                    "template": {
                                        "spec": {
                                            "containers": [{"name": "node"}],
                                        }
                                    }
                                }
                            },
                        }
                    ],
                }
            }
        }
    }
    monkeypatch.setattr(
        "kubeflow_mcp.common.utils.get_custom_objects_api", MagicMock(return_value=api)
    )
    monkeypatch.setattr(
        "kubeflow_mcp.common.utils.get_trainer_effective_namespace",
        lambda namespace=None: namespace or "default",
    )
    monkeypatch.setattr(
        "kubeflow_mcp.trainer.api.planning.get_cluster_resources", MagicMock(return_value={})
    )


# ─── fine_tune validation ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="preview accepts any model URI",
            expected_status=PREVIEW,
            config={"model": "not-a-model", "dataset": "hf://org/ds"},
        ),
        TestCase(
            name="preview includes model in config",
            expected_status=PREVIEW,
            config={"model": "hf://org/model", "dataset": "hf://org/ds"},
            expected_output={"model": "hf://org/model"},
        ),
        TestCase(
            name="invalid name format rejected",
            expected_status=FAILED,
            config={"model": "hf://org/model", "dataset": "hf://org/ds", "name": "INVALID_NAME"},
            expected_error_code=VALIDATION_ERROR,
        ),
        TestCase(
            name="batch_size out of bounds rejected",
            expected_status=FAILED,
            config={"model": "hf://org/model", "dataset": "hf://org/ds", "batch_size": 9999},
            expected_error_code=VALIDATION_ERROR,
        ),
    ],
)
def test_fine_tune_validation(test_case):
    with patch(PATCH_GPU_CHECK, return_value=None):
        assert_test_case(test_case, fine_tune)


class TestFineTuneConfirmed:
    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_confirmed_creates_job(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-gemma-abc"
        mock_client_fn.return_value = mock_client
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            runtime="torchtune-llama",
            confirmed=True,
        )
        assert result["success"] is True
        assert result["data"]["job_name"] == "train-gemma-abc"
        assert result["data"]["status"] == "Created"
        mock_client.train.assert_called_once()

    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_hf_token_passed_to_sdk(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            runtime="torchtune-llama",
            hf_token="hf_secret_token",
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        initializer = call_kwargs.kwargs["initializer"]
        assert initializer.model.access_token == "hf_secret_token"

    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_dataset_parameter_handling(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/dataset",
            runtime="torchtune-llama",
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        initializer = call_kwargs.kwargs["initializer"]
        assert "hf://org/dataset" in initializer.dataset.storage_uri

    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_lora_config(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            runtime="torchtune-llama",
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.1,
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        trainer = call_kwargs.kwargs["trainer"]
        assert trainer.config.peft_config.lora_rank == 16
        assert trainer.config.peft_config.lora_alpha == 32
        assert trainer.config.peft_config.lora_dropout == 0.1

    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_s3_model_source(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = fine_tune(
            model="s3://bucket/model",
            dataset="s3://bucket/dataset",
            runtime="torchtune-llama",
            s3_access_key_id="AKIA",
            s3_secret_access_key="secret",
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        initializer = call_kwargs.kwargs["initializer"]
        assert initializer.model.storage_uri == "s3://bucket/model"
        assert initializer.dataset.storage_uri == "s3://bucket/dataset"

    @patch(PATCH_NS_CHECK)
    def test_namespace_policy_enforcement(self, mock_ns_check):
        from kubeflow_mcp.common.types import ToolError as ToolErrorModel

        mock_ns_check.return_value = ToolErrorModel(
            error="namespace blocked", error_code="PERMISSION_DENIED"
        )
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            namespace="forbidden-ns",
            confirmed=True,
        )
        assert result["success"] is False
        assert result["error_code"] == "PERMISSION_DENIED"

    def test_gpu_validation_blocks_on_zero_gpus(self):
        from kubeflow_mcp.common.types import ToolError as ToolErrorModel

        gpu_err = ToolErrorModel(
            error="fine_tune() requires GPUs", error_code="VALIDATION_ERROR"
        ).model_dump()
        with patch(PATCH_GPU_CHECK, return_value=gpu_err):
            result = fine_tune(
                model="hf://org/model",
                dataset="hf://org/ds",
            )
        assert result["success"] is False
        assert "GPU" in result["error"]

    @patch(PATCH_GPU_CHECK, return_value=None)
    def test_non_torchtune_runtime_rejected(self, _gpu):
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            runtime="torch-distributed",
        )
        assert result["success"] is False
        assert "torchtune" in result["error"]


# ─── run_custom_training validation ─────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="empty script rejected",
            expected_status=FAILED,
            config={"script": "   ", "runtime": "torch-distributed"},
        ),
        TestCase(
            name="invalid name rejected",
            expected_status=FAILED,
            config={"script": "print('hello')", "runtime": "torch-distributed", "name": "BadName"},
        ),
        TestCase(
            name="unsafe script blocked on confirmed",
            expected_status=FAILED,
            config={
                "script": "import os\nos.system('rm -rf /')",
                "runtime": "torch-distributed",
                "confirmed": True,
            },
        ),
        TestCase(
            name="unsafe script preview includes warnings",
            expected_status=PREVIEW,
            config={
                "script": "import os\nos.system('rm -rf /')",
                "runtime": "torch-distributed",
                "confirmed": False,
            },
        ),
    ],
)
def test_run_custom_training_validation(test_case):
    if test_case.expected_status == PREVIEW:
        result = run_custom_training(**test_case.config)
        assert result["status"] == "preview"
        assert len(result["config"]["safety_warnings"]) > 0
    else:
        assert_test_case(test_case, run_custom_training)


@pytest.mark.parametrize(
    ("tool", "config"),
    [
        (run_custom_training, {"script": "print('hello')", "runtime": "torch-distributed"}),
        (run_container_training, {"image": "pytorch/pytorch:2.0"}),
        (
            fine_tune,
            {
                "model": "hf://org/model",
                "dataset": "hf://org/dataset",
                "runtime": "torchtune-llama",
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "bad_resources",
    [
        {"cpu": "4cores"},
        {"memory": "256MB"},
        {"gpu": -1},
        {"gpu": "two"},
        {"gpu": 1.5},
        {"cpu": ""},
        {"memory": None},
        {"amd.com/gpu": "lots"},
        {},
        ["gpu", 1],
    ],
)
def test_training_tools_reject_invalid_resources_before_sdk_call(tool, config, bad_resources):
    with patch(PATCH_CLIENT) as mock_client:
        result = tool(**config, resources_per_node=bad_resources, confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == VALIDATION_ERROR
    mock_client.assert_not_called()


@pytest.mark.parametrize(
    "resources",
    [
        {"cpu": 8},
        {"cpu": 0.5},
        {"cpu": "500m"},
        {"memory": "10G"},
        {"memory": "1.5Gi"},
        {"gpu": 2},
        {"amd.com/gpu": 2},
    ],
)
def test_validate_resources_per_node_accepts_kubernetes_quantities(resources):
    assert _validate_resources_per_node(resources) is None


def test_fine_tune_validates_resources_before_gpu_preflight():
    with patch(PATCH_GPU_CHECK) as gpu_check:
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/dataset",
            runtime="torchtune-llama",
            resources_per_node={"cpu": "4cores"},
        )

    assert result["error_code"] == VALIDATION_ERROR
    gpu_check.assert_not_called()


# ─── fine_tune Hugging Face model/dataset checks ────────────────────────────


@pytest.mark.parametrize(
    ("model", "dataset"),
    [
        ("hf://org/ model", "hf://org/ds"),
        ("hf://org/model", "hf://only-one-segment"),
        ("hf://org/../model", "hf://org/ds"),
    ],
)
@pytest.mark.parametrize("confirmed", [False, True])
def test_fine_tune_rejects_bad_hf_reference_without_resolving_namespace(model, dataset, confirmed):
    """Malformed hf:// input must be rejected before the namespace is resolved."""
    with (
        patch("kubeflow_mcp.core.policy.get_allowed_namespaces", return_value=["team-a"]),
        patch("kubeflow_mcp.common.utils.get_trainer_effective_namespace") as mock_resolve_ns,
        patch(PATCH_CLIENT) as mock_client,
        patch(PATCH_GPU_CHECK) as gpu_check,
    ):
        result = fine_tune(
            model=model,
            dataset=dataset,
            runtime="torchtune-llama",
            namespace=None,
            confirmed=confirmed,
        )

    assert result["success"] is False
    assert result["error_code"] == VALIDATION_ERROR
    mock_resolve_ns.assert_not_called()
    mock_client.assert_not_called()
    gpu_check.assert_not_called()


def _hub_error(error_cls: type[Exception], status_code: int) -> Exception:
    request = httpx.Request("GET", "https://huggingface.co/api/models/x")
    return error_cls("Hub error", response=httpx.Response(status_code, request=request))


@pytest.mark.parametrize(
    ("model", "dataset"),
    [
        ("hf://", "hf://org/ds"),
        ("hf://bad!name", "hf://org/ds"),
        ("hf://org/model/extra", "hf://org/ds"),
        ("hf://org//model", "hf://org/ds"),
        ("hf://org/bad name", "hf://org/ds"),
        (" hf://org/model", "hf://org/ds"),  # leading whitespace skips the hf:// check
        ("hf://org/model", "hf://squad"),  # datasets need org/name, unlike models
        ("hf://org/model", "hf://org//ds"),
        ("hf://org/model", "hf://org/ds/my path"),  # whitespace in the dataset subpath
        ("hf://org/model", "hf://org/ds/../other"),  # path traversal in the subpath
    ],
)
def test_fine_tune_preview_rejects_malformed_hf_references(model, dataset):
    with patch(PATCH_GPU_CHECK) as gpu_check, patch("huggingface_hub.model_info") as hub:
        result = fine_tune(
            model=model,
            dataset=dataset,
            runtime="torchtune-llama",
        )

    assert result["error_code"] == VALIDATION_ERROR
    gpu_check.assert_not_called()
    hub.assert_not_called()


@pytest.mark.parametrize(
    ("model", "dataset"),
    [
        ("hf://org/model", "hf://org/ds/subdir"),
        ("hf://gpt2", "hf://org/ds"),  # the Hub still serves unscoped model names
        (
            "hf://org/model",
            "s3://my-bucket/path with spaces",
        ),  # non-HF refs with spaces are allowed
    ],
)
def test_fine_tune_preview_accepts_valid_hf_references_without_hub_call(model, dataset):
    with (
        patch(PATCH_NS_CHECK, return_value=None),
        patch(PATCH_GPU_CHECK, return_value=None),
        patch("huggingface_hub.model_info") as model_hub,
        patch("huggingface_hub.dataset_info") as dataset_hub,
    ):
        result = fine_tune(model=model, dataset=dataset, runtime="torchtune-llama")

    assert result["status"] == "preview"
    model_hub.assert_not_called()
    dataset_hub.assert_not_called()


def test_fine_tune_reports_bad_model_before_runtime_error():
    with patch(PATCH_GPU_CHECK, return_value=None):
        result = fine_tune(model="hf://a/b/c", dataset="hf://org/ds", runtime="")

    assert result["error_code"] == VALIDATION_ERROR
    assert "model reference" in result["error"]


@pytest.mark.parametrize(
    ("hub_function", "model", "missing"),
    [
        ("huggingface_hub.model_info", "hf://org/model", "model"),
        ("huggingface_hub.model_info", "hf://gpt3-typo", "model"),  # unscoped model
        ("huggingface_hub.dataset_info", "hf://org/model", "dataset"),
    ],
)
def test_fine_tune_submission_rejects_confirmed_missing_repo(hub_function, model, missing):
    with (
        patch(PATCH_NS_CHECK, return_value=None),
        patch(PATCH_CLIENT) as client,
        patch(hub_function, side_effect=_hub_error(RepositoryNotFoundError, 404)),
        patch("huggingface_hub.list_models", return_value=[]),
    ):
        result = fine_tune(
            model=model,
            dataset="hf://org/ds",
            runtime="torchtune-llama",
            confirmed=True,
        )

    assert result["error_code"] == VALIDATION_ERROR
    assert f"Hugging Face {missing}" in result["error"]
    assert "does not exist" in result["error"]  # the Hub check, not the format check
    client.assert_not_called()


def test_fine_tune_submission_missing_unscoped_model_returns_suggestions():
    with (
        patch(PATCH_NS_CHECK, return_value=None),
        patch(PATCH_CLIENT) as client,
        patch(
            "huggingface_hub.model_info",
            side_effect=_hub_error(RepositoryNotFoundError, 404),
        ),
        patch(
            "huggingface_hub.list_models",
            return_value=[MagicMock(id="gpt2"), MagicMock(id="gpt2-medium")],
        ),
    ):
        result = fine_tune(
            model="hf://gpt2-typo",
            dataset="hf://org/ds",
            runtime="torchtune-llama",
            confirmed=True,
        )

    assert result["error_code"] == VALIDATION_ERROR
    assert result["details"]["suggestions"] == ["gpt2", "gpt2-medium"]
    client.assert_not_called()


@pytest.mark.parametrize(
    "hub_error",
    [
        _hub_error(GatedRepoError, 403),
        _hub_error(RepositoryNotFoundError, 401),
        ConnectionError("Hub unreachable"),
    ],
    ids=["gated", "private-or-anonymous", "hub-unavailable"],
)
def test_fine_tune_submission_not_blocked_by_inaccessible_or_unavailable_hub(hub_error):
    with (
        patch(PATCH_NS_CHECK, return_value=None),
        patch(PATCH_CLIENT) as client_fn,
        patch("huggingface_hub.model_info", side_effect=hub_error),
        patch("huggingface_hub.dataset_info", side_effect=hub_error),
        patch("huggingface_hub.list_models", return_value=[]),
    ):
        client_fn.return_value.train.return_value = "train-job"
        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            runtime="torchtune-llama",
            confirmed=True,
        )

    assert result["success"] is True
    client_fn.return_value.train.assert_called_once()


_TRAINING_TOOLS = [
    (run_custom_training, {"script": "print('hello')", "runtime": "torch-distributed"}),
    (run_container_training, {"image": "pytorch/pytorch:2.0"}),
    (
        fine_tune,
        {"model": "hf://org/model", "dataset": "hf://org/dataset", "runtime": "torchtune-llama"},
    ),
]


@pytest.mark.parametrize(("tool", "config"), _TRAINING_TOOLS)
@pytest.mark.parametrize(
    "bad_name", ["INVALID_NAME", "bad service account", "sa/name", "", "a" * 254]
)
def test_training_tools_reject_invalid_service_account_before_any_call(tool, config, bad_name):
    with patch(PATCH_CLIENT) as mock_client, patch(PATCH_GPU_CHECK) as gpu_check:
        result = tool(**config, service_account_name=bad_name, confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == VALIDATION_ERROR
    mock_client.assert_not_called()
    gpu_check.assert_not_called()


@pytest.mark.parametrize(("tool", "config"), _TRAINING_TOOLS)
def test_training_tools_reject_invalid_service_account_without_resolving_namespace(tool, config):
    """An invalid ServiceAccount name must be rejected before resolving default namespace."""
    with (
        patch("kubeflow_mcp.core.policy.get_allowed_namespaces", return_value=["team-a"]),
        patch("kubeflow_mcp.common.utils.get_trainer_effective_namespace") as mock_resolve_ns,
        patch(PATCH_CLIENT) as mock_client,
        patch(PATCH_GPU_CHECK) as gpu_check,
    ):
        result = tool(**config, namespace=None, service_account_name="INVALID_NAME", confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == VALIDATION_ERROR
    mock_resolve_ns.assert_not_called()
    mock_client.assert_not_called()
    gpu_check.assert_not_called()


@pytest.mark.parametrize(("tool", "config"), _TRAINING_TOOLS)
def test_training_tools_accept_dotted_service_account(tool, config):
    with patch(PATCH_NS_CHECK, return_value=None), patch(PATCH_GPU_CHECK, return_value=None):
        result = tool(**config, service_account_name="trainer.sa-1")

    assert result["status"] == "preview"


def _forbidden(**_kwargs):
    """Mimic the SDK, which wraps the Kubernetes ApiException in a RuntimeError."""
    try:
        raise ApiException(status=403, reason="Forbidden")
    except ApiException as e:
        raise RuntimeError("Failed to create TrainJob") from e


def _forbidden_without_from(**_kwargs):
    """Wrap the 403 without ``from``, so it is only on ``__context__``."""
    try:
        raise ApiException(status=403, reason="Forbidden")
    except ApiException:
        raise RuntimeError("Failed to create TrainJob")  # noqa: B904


@pytest.mark.parametrize("raise_forbidden", [_forbidden, _forbidden_without_from])
@patch(PATCH_NS_CHECK, return_value=None)
@patch(PATCH_CLIENT)
def test_forbidden_submission_returns_permission_denied(mock_client_fn, _ns, raise_forbidden):
    """A 403 from RBAC or an admission policy is a permission problem, not SDK_ERROR."""
    mock_client_fn.return_value.train.side_effect = raise_forbidden

    result = run_custom_training(
        script="print('hello')",
        runtime="torch-distributed",
        service_account_name="privileged-sa",
        confirmed=True,
    )

    assert result["success"] is False
    assert result["error_code"] == "PERMISSION_DENIED"
    assert "service_account_name" in result["hint"]


@pytest.mark.parametrize(
    ("tool", "config"),
    [
        (run_custom_training, {"script": "print('hello')", "runtime": "torch-distributed"}),
        (run_container_training, {}),
    ],
)
@pytest.mark.parametrize("bad_image", ["   ", "img::double", "-leading/x"])
def test_training_tools_reject_invalid_image_before_sdk_call(tool, config, bad_image):
    with patch(PATCH_CLIENT) as mock_client:
        result = tool(**config, image=bad_image, confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == VALIDATION_ERROR
    mock_client.assert_not_called()


@pytest.mark.parametrize(
    ("tool", "config"),
    [
        (run_custom_training, {"script": "print('hello')", "runtime": "torch-distributed"}),
        (run_container_training, {}),
    ],
)
def test_training_tools_reject_invalid_image_without_resolving_namespace(tool, config):
    """A malformed image must be rejected before resolving the default namespace."""
    with (
        patch("kubeflow_mcp.core.policy.get_allowed_namespaces", return_value=["team-a"]),
        patch("kubeflow_mcp.common.utils.get_trainer_effective_namespace") as mock_resolve_ns,
        patch(PATCH_CLIENT) as mock_client,
    ):
        result = tool(**config, namespace=None, image="img::double", confirmed=True)

    assert result["success"] is False
    assert result["error_code"] == VALIDATION_ERROR
    mock_resolve_ns.assert_not_called()
    mock_client.assert_not_called()


def test_run_container_training_rejects_empty_image():
    """The image is required here, so "" is an error, unlike run_custom_training."""
    with patch(PATCH_CLIENT) as mock_client:
        result = run_container_training(image="", confirmed=True)

    assert result["error_code"] == VALIDATION_ERROR
    mock_client.assert_not_called()


# "" counts as unset: agents often send it for unused optional fields, and the SDK
# then falls back to the runtime image.
@pytest.mark.parametrize("image", [None, "", "localhost:5000/team/trainer:v1"])
def test_run_custom_training_accepts_valid_or_missing_image(image):
    with patch(PATCH_NS_CHECK, return_value=None):
        result = run_custom_training(
            script="print('hello')", runtime="torch-distributed", image=image
        )

    assert result["status"] == "preview"


class TestRunCustomTrainingConfirmed:
    @pytest.mark.parametrize(
        ("volumes", "mounts"),
        [
            (None, None),
            ([], []),
            ([{"name": "data", "emptyDir": {}}], [{"name": "data", "mountPath": "/data"}]),
            (
                [{"name": "workspace", "persistentVolumeClaim": {"claimName": "data"}}],
                [{"name": "workspace", "mountPath": "/data"}],
            ),
            (
                [{"name": "data", "persistentVolumeClaim": {"claimName": "data"}}],
                [{"name": "data", "mountPath": "/workspace"}],
            ),
        ],
        ids=["omitted", "empty", "other-volume", "name-collision", "existing-workspace"],
    )
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_workspace_matches_preview_and_preserves_inputs(
        self,
        mock_client_fn: MagicMock,
        _ns: MagicMock,
        volumes: list[dict] | None,
        mounts: list[dict] | None,
    ) -> None:
        original_volumes = [dict(v) for v in volumes] if volumes is not None else None
        original_mounts = [dict(m) for m in mounts] if mounts is not None else None
        kwargs = {"script": "print('hello')", "volumes": volumes, "volume_mounts": mounts}
        preview = run_custom_training(**kwargs)
        mock_client_fn.assert_not_called()
        mock_client_fn.return_value.train.return_value = "workspace-test"
        result = run_custom_training(**kwargs, confirmed=True)

        assert result["success"] is True
        node = _pod_specs(mock_client_fn.return_value.train.call_args.kwargs["options"])["node"]
        actual_mounts = node.containers[0].volume_mounts
        assert node.volumes == preview["config"]["volumes"]
        assert actual_mounts == preview["config"]["volume_mounts"]
        assert len({v["name"] for v in node.volumes}) == len(node.volumes)
        workspace_mounts = [m for m in actual_mounts if m["mountPath"] == "/workspace"]
        assert len(workspace_mounts) == 1
        workspace = next(v for v in node.volumes if v["name"] == workspace_mounts[0]["name"])
        if any(m["mountPath"] == "/workspace" for m in (mounts or [])):
            assert node.volumes == volumes
            assert actual_mounts == mounts
        else:
            assert workspace["emptyDir"] == {}
        assert node.volumes[: len(volumes or [])] == (volumes or [])
        assert actual_mounts[: len(mounts or [])] == (mounts or [])
        assert volumes == original_volumes
        assert mounts == original_mounts

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_confirmed_creates_job(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "custom-train-abc"
        mock_client_fn.return_value = mock_client
        result = run_custom_training(
            script="print('hello')",
            runtime="torch-distributed",
            confirmed=True,
        )
        assert result["success"] is True
        assert result["data"]["job_name"] == "custom-train-abc"
        assert result["data"]["status"] == "Created"
        mock_client.train.assert_called_once()

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_env_parameter_passed(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = run_custom_training(
            script="print('hello')",
            runtime="torch-distributed",
            env={"MY_VAR": "value"},
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        trainer = call_kwargs.kwargs["trainer"]
        assert trainer.env == {"MY_VAR": "value"}

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_packages_parameter(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = run_custom_training(
            script="import torch; print(torch.__version__)",
            runtime="torch-distributed",
            packages=["torch", "transformers"],
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        trainer = call_kwargs.kwargs["trainer"]
        assert trainer.packages_to_install == ["torch", "transformers"]

    def test_preview_returns_config(self):
        result = run_custom_training(
            script="print('hello')",
            runtime="torch-distributed",
            confirmed=False,
        )
        assert result["status"] == "preview"
        assert result["config"]["runtime"] == "torch-distributed"
        assert "print('hello')" in result["config"]["script"]

    def test_auto_generated_name_format(self):
        result = run_custom_training(
            script="print('hello')",
            runtime="torch-distributed",
            confirmed=False,
        )
        assert result["status"] == "preview"
        assert result["config"]["name"] is None

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_unsafe_scripts_override(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        with patch.dict("os.environ", {"KUBEFLOW_MCP_UNSAFE_SCRIPTS": "true"}):
            result = run_custom_training(
                script="import os\nos.system('ls')",
                runtime="torch-distributed",
                confirmed=True,
            )
        assert result["success"] is True

    @patch(PATCH_NS_CHECK)
    def test_namespace_policy_enforcement(self, mock_ns_check):
        from kubeflow_mcp.common.types import ToolError as ToolErrorModel

        mock_ns_check.return_value = ToolErrorModel(
            error="namespace blocked", error_code="PERMISSION_DENIED"
        )
        result = run_custom_training(
            script="print('hello')",
            namespace="forbidden-ns",
            confirmed=True,
        )
        assert result["success"] is False
        assert result["error_code"] == "PERMISSION_DENIED"


# ─── run_container_training validation ──────────────────────────────────────


def test_run_container_training_invalid_name():
    result = run_container_training(
        image="ghcr.io/kubeflow/trainer/torch:latest",
        name="BadName",
    )
    assert result["success"] is False


class TestRunContainerTrainingConfirmed:
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_confirmed_creates_job(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "container-train-abc"
        mock_client_fn.return_value = mock_client
        result = run_container_training(
            image="pytorch/pytorch:2.0",
            confirmed=True,
        )
        assert result["success"] is True
        assert result["data"]["job_name"] == "container-train-abc"
        assert result["data"]["status"] == "Created"
        mock_client.train.assert_called_once()

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_command_and_args_override(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "train-job"
        mock_client_fn.return_value = mock_client
        result = run_container_training(
            image="pytorch/pytorch:2.0",
            command=["python", "train.py"],
            args=["--epochs", "5"],
            confirmed=True,
        )
        assert result["success"] is True
        call_kwargs = mock_client.train.call_args
        options = call_kwargs.kwargs["options"]
        from kubeflow.trainer.options import TrainerArgs, TrainerCommand

        cmd_opts = [o for o in options if isinstance(o, TrainerCommand)]
        args_opts = [o for o in options if isinstance(o, TrainerArgs)]
        assert len(cmd_opts) == 1
        assert cmd_opts[0].command == ["python", "train.py"]
        assert len(args_opts) == 1
        assert args_opts[0].args == ["--epochs", "5"]

    def test_preview_returns_config(self):
        result = run_container_training(
            image="pytorch/pytorch:2.0",
            confirmed=False,
        )
        assert result["status"] == "preview"
        assert result["config"]["image"] == "pytorch/pytorch:2.0"

    @patch(PATCH_NS_CHECK)
    def test_namespace_policy_enforcement(self, mock_ns_check):
        from kubeflow_mcp.common.types import ToolError as ToolErrorModel

        mock_ns_check.return_value = ToolErrorModel(
            error="namespace blocked", error_code="PERMISSION_DENIED"
        )
        result = run_container_training(
            image="pytorch/pytorch:2.0",
            namespace="forbidden-ns",
            confirmed=True,
        )
        assert result["success"] is False
        assert result["error_code"] == "PERMISSION_DENIED"


# ─── _make_train_func ────────────────────────────────────────────────────────


def test_make_train_func_wraps_script():
    func = _make_train_func("x = 1 + 1\nprint(x)")
    assert callable(func)
    assert func.__name__ == "train"


def test_make_train_func_accepts_args():
    func = _make_train_func("print('hi')", func_args={"lr": 0.001})
    assert callable(func)


def test_make_train_func_inspect_getsource():
    func = _make_train_func("x = 42\nprint(x)")
    source = inspect.getsource(func)
    assert "x = 42" in source
    assert "def train" in source


def test_make_train_func_syntax_error():
    with pytest.raises(SyntaxError):
        _make_train_func("def broken(:\n    pass")


def test_make_train_func_args_appear_as_params():
    func = _make_train_func("print(lr)", func_args={"lr": 0.001, "epochs": 5})
    source = inspect.getsource(func)
    assert "lr=None" in source
    assert "epochs=None" in source


# ─── Pure helpers (AST / builders) ────────────────────────────────────────────


@pytest.mark.parametrize(
    "script,expected",
    [
        ("def train():\n    pass\ntrain()", True),
        ("def train():\n    pass", False),
    ],
)
def test_has_train_call(script, expected):
    tree = ast.parse(script)
    assert _has_train_call(tree.body) is expected


def test_has_required_params_detects_missing_defaults():
    tree = ast.parse("def train(lr, batch_size=32):\n    pass")
    train_def = tree.body[0]
    assert _has_required_params(train_def) is True


def test_has_required_params_all_optional():
    tree = ast.parse("def train(lr=0.01):\n    pass")
    train_def = tree.body[0]
    assert _has_required_params(train_def) is False


def test_forwarded_train_args_builds_kwargs():
    tree = ast.parse("def train(lr, batch_size):\n    pass")
    train_def = tree.body[0]
    assert _forwarded_train_args(train_def, {"lr": 0.01, "batch_size": 4}) == (
        "lr=lr, batch_size=batch_size"
    )


def test_forwarded_train_args_rejects_unknown_keys():
    tree = ast.parse("def train(lr):\n    pass")
    train_def = tree.body[0]
    with pytest.raises(ValueError, match="func_args"):
        _forwarded_train_args(train_def, {"lr": 0.01, "extra": 1})


def test_uncalled_train_call_appends_invocation():
    lines = _uncalled_train_call("def train():\n    print('hi')")
    assert lines == ["train()"]


def test_uncalled_train_call_skips_when_already_called():
    script = "def train():\n    pass\ntrain()"
    assert _uncalled_train_call(script) == []


@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("hf://org/dataset", True),
        ("hf://org/dataset/subpath", False),
        ("s3://bucket/ds", False),
    ],
)
def test_should_apply_hf_dataset_workaround(dataset, expected):
    assert _should_apply_hf_dataset_workaround(dataset) is expected


def test_build_fine_tune_config_masks_secrets():
    config = _build_fine_tune_config(
        model="hf://org/model",
        dataset="hf://org/ds",
        runtime="torchtune-llama",
        name="train-gemma",
        hf_token="hf_secret",
        batch_size=4,
        epochs=1,
        num_nodes=1,
        dtype=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=None,
        use_dora=None,
        quantize_base=None,
        s3_access_key_id="AKIA",
        s3_secret_access_key="secret",
        optional_fields=[],
    )
    assert config["hf_token"] == "***"
    assert config["s3_access_key_id"] == "***"
    assert config["s3_secret_access_key"] == "***"


def test_inject_trainer_hf_home_concurrent_isolation():
    """Verify concurrent _inject_trainer_hf_home calls maintain context isolation and restore original function."""
    import kubeflow.trainer.backends.kubernetes.utils as k8s_utils
    from kubeflow.trainer.types import types as sdk_types

    original_func = k8s_utils.get_trainer_cr_from_builtin_trainer
    runtime_trainer = sdk_types.RuntimeTrainer(
        trainer_type=sdk_types.TrainerType.BUILTIN_TRAINER,
        framework="torchtune",
        image="test:latest",
    )
    runtime_trainer.set_command(("torchrun",))
    runtime = sdk_types.Runtime(
        name="test-rt",
        trainer=runtime_trainer,
        kind=sdk_types.RuntimeKind.CLUSTER_TRAINING_RUNTIME,
    )
    builtin = sdk_types.BuiltinTrainer(config=sdk_types.TorchTuneConfig(batch_size=4, epochs=1))

    barrier = threading.Barrier(4)
    errors: list[str] = []
    completed: list[int | str] = []

    def worker(worker_id: int) -> None:
        try:
            target_path = f"/workspace/.hf_{worker_id}"
            with _inject_trainer_hf_home(target_path):
                barrier.wait(timeout=5.0)
                cr = k8s_utils.get_trainer_cr_from_builtin_trainer(runtime, builtin)
                if cr.env is None:
                    errors.append(f"Worker {worker_id}: cr.env is None")
                    return
                hf_env = next((e for e in cr.env if e.name == "HF_HOME"), None)
                if not hf_env or hf_env.value != target_path:
                    errors.append(f"Worker {worker_id}: expected {target_path}, got {hf_env}")
                    return
                completed.append(worker_id)
        except Exception as exc:
            errors.append(f"Worker {worker_id} raised: {exc!r}")

    def uninjected_worker() -> None:
        try:
            barrier.wait(timeout=5.0)
            cr = k8s_utils.get_trainer_cr_from_builtin_trainer(runtime, builtin)
            if cr.env is not None and any(e.name == "HF_HOME" for e in cr.env):
                errors.append("Uninjected worker unexpectedly got HF_HOME")
                return
            completed.append("uninjected")
        except Exception as exc:
            errors.append(f"Uninjected worker raised: {exc!r}")

    threads = [
        threading.Thread(target=worker, args=(1,)),
        threading.Thread(target=worker, args=(2,)),
        threading.Thread(target=worker, args=(3,)),
        threading.Thread(target=uninjected_worker),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive(), "Worker thread timed out"

    assert not errors, f"Concurrency errors encountered: {errors}"
    assert len(completed) == 4, f"Expected 4 completed workers, got {len(completed)}"
    assert k8s_utils.get_trainer_cr_from_builtin_trainer is original_func, (
        "Monkey-patch function identity was not restored to original"
    )


def test_inject_trainer_hf_home_exception_handling():
    """Verify refcount and monkey-patch cleanly revert if an exception occurs within yield."""
    import kubeflow.trainer.backends.kubernetes.utils as k8s_utils

    original_func = k8s_utils.get_trainer_cr_from_builtin_trainer
    with pytest.raises(RuntimeError, match="simulated failure"):
        with _inject_trainer_hf_home("/workspace/.hf_err"):
            assert k8s_utils.get_trainer_cr_from_builtin_trainer is not original_func
            raise RuntimeError("simulated failure")

    assert k8s_utils.get_trainer_cr_from_builtin_trainer is original_func, (
        "Monkey-patch function identity was not restored after exception"
    )


# ─── _build_runtime_patch ───────────────────────────────────────────────────

_ENV = [{"name": "HF_HOME", "value": "/cache"}]
_VOLUMES = [{"name": "data", "persistentVolumeClaim": {"claimName": "data-pvc"}}]
_MOUNTS = [{"name": "data", "mountPath": "/data"}]


def _pod_specs(options: list) -> dict:
    """Map each replicated job name in the RuntimePatch to its pod spec patch."""
    runtime_patch = options[0]
    jobs = runtime_patch.training_runtime_spec.template.spec.replicated_jobs
    return {job.name: job.template.spec.template.spec for job in jobs}


class TestBuildRuntimePatch:
    def test_no_options_returns_empty_list(self):
        assert _build_runtime_patch() == []

    def test_pod_fields_go_to_node_pod_spec(self):
        affinity = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {"matchExpressions": [{"key": "zone", "operator": "In", "values": ["a"]}]}
                    ]
                }
            }
        }
        pull_secrets = [{"name": "registry-creds"}]
        options = _build_runtime_patch(
            node_selector={"gpu": "a100"},
            tolerations=[{"key": "gpu", "operator": "Exists"}],
            affinity=affinity,
            service_account_name="trainer-sa",
            image_pull_secrets=pull_secrets,
            env=_ENV,
            volume_mounts=_MOUNTS,
        )

        specs = _pod_specs(options)
        assert list(specs) == ["node"]
        node = specs["node"]
        assert node.node_selector == {"gpu": "a100"}
        assert node.tolerations == [{"key": "gpu", "operator": "Exists"}]
        assert node.affinity == affinity
        assert node.service_account_name == "trainer-sa"
        assert node.image_pull_secrets == pull_secrets
        assert [(c.name, c.env, c.volume_mounts) for c in node.containers] == [
            ("node", _ENV, _MOUNTS)
        ]

    def test_initializers_get_volumes_when_enabled(self):
        options = _build_runtime_patch(
            volumes=_VOLUMES, volume_mounts=_MOUNTS, has_initializers=True
        )

        specs = _pod_specs(options)
        assert list(specs) == ["node", "dataset-initializer", "model-initializer"]
        for name in ("dataset-initializer", "model-initializer"):
            assert specs[name].volumes == _VOLUMES
            assert [(c.name, c.volume_mounts) for c in specs[name].containers] == [(name, _MOUNTS)]

    def test_initializers_not_patched_without_has_initializers(self):
        options = _build_runtime_patch(volumes=_VOLUMES, volume_mounts=_MOUNTS)

        assert list(_pod_specs(options)) == ["node"]

    def test_labels_and_annotations_are_separate_options(self):
        options = _build_runtime_patch(labels={"team": "ml"}, annotations={"owner": "alice"})

        assert [type(o).__name__ for o in options] == ["Labels", "Annotations"]
        assert options[0].labels == {"team": "ml"}
        assert options[1].annotations == {"owner": "alice"}


# ─── Top-level Hugging Face Dataset Workaround (#32, #494) ───────────────────


class TestFineTuneHuggingFaceDatasetWorkaround:
    """Verify fine_tune handles top-level Hugging Face dataset URIs with SDK 0.5.0."""

    @pytest.fixture
    def mock_backend(self):
        from unittest.mock import MagicMock

        from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend
        from kubeflow.trainer.types.types import TrainerType

        backend = KubernetesBackend.__new__(KubernetesBackend)
        backend.namespace = "default"
        backend.custom_api = MagicMock()

        fake_runtime = MagicMock()
        fake_runtime.name = "torchtune-llama3.2-1b"
        fake_runtime.kind.value = "ClusterTrainingRuntime"
        fake_runtime.trainer.trainer_type = TrainerType.BUILTIN_TRAINER
        fake_runtime.trainer.command = ["tune", "run"]
        backend.get_runtime = MagicMock(return_value=fake_runtime)
        return backend

    @patch("kubeflow_mcp.trainer.api.training._check_hf_repos_exist", return_value=None)
    @patch("kubeflow_mcp.trainer.api.training.check_namespace_allowed", return_value=None)
    def test_top_level_hf_dataset_reaches_creation_without_trainer_args_error(
        self, _ns, _hf, mock_backend
    ):
        with patch("kubeflow_mcp.trainer.api.training._get_client", return_value=mock_backend):
            result = fine_tune(
                model="hf://meta-llama/Llama-3.2-1B",
                dataset="hf://tatsu-lab/alpaca",
                runtime="torchtune-llama3.2-1b",
                namespace="default",
                confirmed=True,
            )

        assert result["success"] is True
        mock_backend.custom_api.create_namespaced_custom_object.assert_called_once()
        train_job = mock_backend.custom_api.create_namespaced_custom_object.call_args[0][4]
        args = train_job["spec"]["trainer"]["args"]

        assert "dataset.source=/workspace/dataset" in args
        assert "dataset.data_dir=null" in args
        assert not any(
            a.startswith("dataset.data_dir=/workspace") and a.endswith("/.") for a in args
        )
        assert "batch_size=4" in args
        assert "epochs=1" in args
        assert "model.lora_rank=8" in args
        assert "model.lora_alpha=16" in args

    @patch("kubeflow_mcp.trainer.api.training._check_hf_repos_exist", return_value=None)
    @patch("kubeflow_mcp.trainer.api.training.check_namespace_allowed", return_value=None)
    def test_top_level_hf_dataset_preserves_explicit_dataset_source(self, _ns, _hf, mock_backend):
        with patch("kubeflow_mcp.trainer.api.training._get_client", return_value=mock_backend):
            result = fine_tune(
                model="hf://meta-llama/Llama-3.2-1B",
                dataset="hf://tatsu-lab/alpaca",
                runtime="torchtune-llama3.2-1b",
                dataset_source="json",
                namespace="default",
                confirmed=True,
            )

        assert result["success"] is True
        train_job = mock_backend.custom_api.create_namespaced_custom_object.call_args[0][4]
        args = train_job["spec"]["trainer"]["args"]

        assert "dataset.source=json" in args
        assert "dataset.data_dir=/workspace/dataset" in args
        assert "dataset.data_dir=null" not in args
        assert "dataset.source=/workspace/dataset" not in args

    @patch("kubeflow_mcp.trainer.api.training._check_hf_repos_exist", return_value=None)
    @patch("kubeflow_mcp.trainer.api.training.check_namespace_allowed", return_value=None)
    def test_s3_dataset_does_not_apply_hf_workaround(self, _ns, _hf, mock_backend):
        with patch("kubeflow_mcp.trainer.api.training._get_client", return_value=mock_backend):
            result = fine_tune(
                model="hf://meta-llama/Llama-3.2-1B",
                dataset="s3://my-bucket/dataset",
                runtime="torchtune-llama3.2-1b",
                namespace="default",
                confirmed=True,
            )

        assert result["success"] is True
        train_job = mock_backend.custom_api.create_namespaced_custom_object.call_args[0][4]
        args = train_job["spec"]["trainer"]["args"]

        assert "dataset.source=/workspace/dataset" not in args
        assert "dataset.data_dir=null" not in args


# ─── Script parsing & uncalled train edge cases ─────────────────────────────


class TestUncalledTrainCallEdgeCases:
    def test_uncalled_train_with_required_params_and_no_func_args_raises(self):
        script = "def train(lr, batch_size):\n    pass\n"
        with pytest.raises(
            ValueError,
            match="User-defined train\\(\\) requires parameters but no func_args were provided.",
        ):
            _uncalled_train_call(script, func_args=None)

    def test_has_required_params_kwonly_arg_without_default(self):
        code_without_default = "def train(*, lr):\n    pass\n"
        def_node_without_default = ast.parse(code_without_default).body[0]
        assert _has_required_params(def_node_without_default) is True

        code_with_default = "def train(*, lr=0.01):\n    pass\n"
        def_node_with_default = ast.parse(code_with_default).body[0]
        assert _has_required_params(def_node_with_default) is False

    def test_uncalled_train_async_function_wraps_in_asyncio_run(self):
        script = "async def train():\n    pass\n"
        lines = _uncalled_train_call(script)
        assert lines == ["import asyncio", "asyncio.run(train())"]

        func = _make_train_func(script)
        assert callable(func)

    def test_forwarded_train_args_with_var_kwargs(self):
        code = "def train(**kwargs):\n    pass\n"
        def_node = ast.parse(code).body[0]
        args_str = _forwarded_train_args(def_node, {"lr": 0.01, "epochs": 5})
        assert "lr=lr" in args_str
        assert "epochs=epochs" in args_str


# ─── Client resolution & Ownership labels ───────────────────────────────────


class TestClientAndOwnershipLabel:
    def test_get_client_singleton_vs_namespace(self):
        with (
            patch("kubeflow_mcp.trainer.api.training.get_trainer_client") as mock_singleton,
            patch(
                "kubeflow_mcp.trainer.api.training.get_trainer_client_for_namespace"
            ) as mock_scoped,
        ):
            mock_singleton.return_value = "singleton-client"
            mock_scoped.return_value = "scoped-client"

            assert _get_client(None) == "singleton-client"
            mock_singleton.assert_called_once()
            mock_scoped.assert_not_called()

            assert _get_client("custom-ns") == "scoped-client"
            mock_scoped.assert_called_once_with("custom-ns")

    def test_inject_ownership_label_merges_into_existing_labels(self):
        existing_label = Labels(labels={"app": "training-job"})
        options = [existing_label]
        _inject_ownership_label(options)
        assert len(options) == 1
        assert options[0].labels["app"] == "training-job"
        assert options[0].labels[MCP_MANAGED_LABEL] == MCP_MANAGED_VALUE

    def test_inject_ownership_label_appends_when_no_labels_present(self):
        options = []
        _inject_ownership_label(options)
        assert len(options) == 1
        assert isinstance(options[0], Labels)
        assert options[0].labels == {MCP_MANAGED_LABEL: MCP_MANAGED_VALUE}


# ─── SDK Error wrapper details ──────────────────────────────────────────────


class TestSdkErrorHandling:
    def test_sdk_error_with_cause(self):
        cause = ValueError("original connection error")
        err = RuntimeError("wrapper error")
        err.__cause__ = cause
        result = _sdk_error(err, hint="check connection")
        assert result["success"] is False
        assert result["error_code"] == SDK_ERROR
        assert result["details"] == {"cause": "original connection error"}
        assert result["hint"] == "check connection"

    def test_sdk_error_with_response_text(self):
        err = Exception("k8s api call failed")
        err.response = SimpleNamespace(text='{"message": "resource not found"}')
        result = _sdk_error(err)
        assert result["details"] == {"response": '{"message": "resource not found"}'}

    def test_sdk_error_with_failing_response_text(self):
        class BrokenResponse:
            @property
            def text(self):
                raise RuntimeError("unreadable stream")

        err = Exception("read failed")
        err.response = BrokenResponse()
        result = _sdk_error(err)
        assert result["details"] is None


# ─── GPU availability & Resource validation ─────────────────────────────────


class TestGpuAvailabilityAndResourceValidation:
    def test_check_gpu_available_zero_gpus_returns_error(self):
        err = _check_gpu_available({"gpu_total": 0})
        assert err is not None
        assert err["error_code"] == VALIDATION_ERROR
        assert "fine_tune() requires GPUs" in err["error"]
        assert err["details"]["tool_selection"]["recommended"] == "run_custom_training"

    def test_check_gpu_available_positive_gpus_returns_none(self):
        assert _check_gpu_available({"gpu_total": 8}) is None

    def test_check_gpu_available_handles_missing_or_negative_gracefully(self):
        assert _check_gpu_available({}) is None
        assert _check_gpu_available({"gpu_total": -1}) is None

    def test_validate_resources_per_node_rejects_bool(self):
        err = _validate_resources_per_node({"cpu": True})
        assert err is not None
        assert "must be a string or a number" in err.error


# ─── fine_tune parameter validation and helpers ─────────────────────────────


class TestFineTuneValidationAndConfigExtended:
    def _validate(self, **overrides):
        defaults = {
            "model": "hf://meta-llama/Llama-3-8B",
            "dataset": "hf://tatsu-lab/alpaca",
            "namespace": None,
            "name": None,
            "dtype": "bf16",
            "loss": None,
            "dataset_source": None,
            "batch_size": 4,
            "epochs": 1,
            "num_nodes": 1,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": None,
            "confirmed": True,
        }
        defaults.update(overrides)
        return _validate_fine_tune_params(**defaults)

    def test_validate_fine_tune_params_invalid_dtype(self):
        err = self._validate(dtype="fp16")
        assert err is not None
        assert err["error_code"] == VALIDATION_ERROR
        assert "Must be 'bf16' or 'fp32'" in err["error"]

    def test_validate_fine_tune_params_invalid_loss(self):
        err = self._validate(loss="INVALID_LOSS_NAME")
        assert err is not None
        assert err["error_code"] == VALIDATION_ERROR
        assert "Invalid loss 'INVALID_LOSS_NAME'" in err["error"]

    def test_validate_fine_tune_params_invalid_dataset_source(self):
        err = self._validate(dataset_source="unknown_format")
        assert err is not None
        assert err["error_code"] == VALIDATION_ERROR
        assert "Invalid dataset_source 'unknown_format'" in err["error"]

    def test_validate_fine_tune_params_when_sdk_not_available(self):
        with patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", False):
            err = self._validate(dtype=None)
            assert err is not None
            assert err["error_code"] == SDK_ERROR
            assert "Required training libraries not available" in err["error"]

    def test_build_lora_cfg_extended_options(self):
        cfg = _build_lora_cfg(
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.05,
            use_dora=True,
            quantize_base=True,
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            lora_attn_modules=["q_proj", "v_proj"],
        )
        assert cfg.apply_lora_to_mlp is True
        assert cfg.apply_lora_to_output is True
        assert cfg.lora_attn_modules == ["q_proj", "v_proj"]

    def test_build_dataset_preprocess_cfg_all_none_returns_none(self):
        assert _build_dataset_preprocess_cfg(None, None, None, None, None) is None

    def test_build_dataset_preprocess_cfg_with_arguments(self):
        ds = _build_dataset_preprocess_cfg(
            dataset_source="json",
            dataset_split="train",
            dataset_train_on_input=True,
            dataset_system_prompt="system prompt",
            dataset_column_map={"prompt": "instruction"},
        )
        assert ds is not None
        assert ds.split == "train"
        assert ds.train_on_input is True
        assert ds.new_system_prompt == "system prompt"
        assert ds.column_map == {"prompt": "instruction"}

    def test_build_initializer_s3_with_credentials_and_region(self):
        init = _build_initializer(
            model="s3://my-bucket/model",
            dataset="s3://my-bucket/dataset",
            s3_endpoint="https://s3.custom.io",
            s3_access_key_id="KEY",
            s3_secret_access_key="SECRET",
            s3_region="us-west-2",
            s3_role_arn="arn:aws:iam::123:role/trainer",
        )
        assert init.model.storage_uri == "s3://my-bucket/model"
        assert init.dataset.storage_uri == "s3://my-bucket/dataset"
        assert init.model.endpoint == "https://s3.custom.io"
        assert init.model.region == "us-west-2"
        assert init.model.role_arn == "arn:aws:iam::123:role/trainer"

    def test_fine_tune_preview_includes_optional_fields(self):
        with patch(PATCH_GPU_CHECK, return_value=None):
            result = fine_tune(
                model="hf://org/model",
                dataset="hf://org/ds",
                name="fine-tune-preview",
                dtype="bf16",
                loss="CEWithChunkedOutputLoss",
                dataset_source="json",
                dataset_split="train",
                apply_lora_to_mlp=True,
                apply_lora_to_output=True,
                lora_attn_modules=["q_proj"],
                s3_endpoint="https://s3.local",
                confirmed=False,
            )
            assert result["status"] == "preview"
            cfg = result["config"]
            assert cfg["name"] == "fine-tune-preview"
            assert cfg["loss"] == "CEWithChunkedOutputLoss"
            assert cfg["dataset_source"] == "json"
            assert cfg["apply_lora_to_mlp"] is True
            assert cfg["s3_endpoint"] == "https://s3.local"


# ─── fine_tune submission & error handling ──────────────────────────────────


class TestFineTuneConfirmedExtended:
    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_fine_tune_confirmed_with_name_and_all_options(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.return_value = "custom-named-job"
        mock_client_fn.return_value = mock_client

        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            runtime="torchtune",
            name="custom-named-job",
            dtype="bf16",
            loss="CEWithChunkedOutputLoss",
            resources_per_node={"cpu": 4},
            dataset_source="json",
            dataset_split="train",
            apply_lora_to_mlp=True,
            confirmed=True,
        )
        assert result["success"] is True
        assert result["data"]["job_name"] == "custom-named-job"

        call_kwargs = mock_client.train.call_args.kwargs
        options = call_kwargs["options"]
        name_opts = [o for o in options if isinstance(o, Name)]
        assert len(name_opts) == 1
        assert name_opts[0].name == "custom-named-job"

        trainer = call_kwargs["trainer"]
        assert trainer.config.loss is not None
        assert trainer.config.dtype is not None
        assert trainer.config.dataset_preprocess_config is not None

    @patch(PATCH_GPU_CHECK, return_value=None)
    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_fine_tune_confirmed_sdk_exception_returns_sdk_error(self, mock_client_fn, _ns, _gpu):
        mock_client = MagicMock()
        mock_client.train.side_effect = RuntimeError("SDK client backend connection failure")
        mock_client_fn.return_value = mock_client

        result = fine_tune(
            model="hf://org/model",
            dataset="hf://org/ds",
            runtime="torchtune",
            confirmed=True,
        )
        assert result["success"] is False
        assert result["error_code"] == SDK_ERROR
        assert "SDK client backend connection failure" in result["error"]
        assert "Read trainer://guides/troubleshooting" in result["hint"]


# ─── run_custom_training extended tests ─────────────────────────────────────


def _extract_runtime_node_spec(runtime_patch):
    return runtime_patch.training_runtime_spec.template.spec.replicated_jobs[
        0
    ].template.spec.template.spec


class TestRunCustomTrainingExtended:
    def test_run_custom_training_sdk_not_available(self):
        with patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", False):
            res = run_custom_training(script="print(1)", confirmed=True)
            assert res["success"] is False
            assert res["error_code"] == SDK_ERROR

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_run_custom_training_workspace_volume_injection(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "custom-job"
        mock_client_fn.return_value = mock_client

        res = run_custom_training(
            script="print('hello')",
            volumes=[{"name": "custom-data", "persistentVolumeClaim": {"claimName": "pvc"}}],
            volume_mounts=[{"name": "custom-data", "mountPath": "/data"}],
            confirmed=True,
        )
        assert res["success"] is True

        call_kwargs = mock_client.train.call_args.kwargs
        options = call_kwargs["options"]
        node_spec = _extract_runtime_node_spec(options[0])
        volume_names = [v["name"] for v in node_spec.volumes]
        assert "workspace" in volume_names
        assert "custom-data" in volume_names

        mount_paths = [m["mountPath"] for m in node_spec.containers[0].volume_mounts]
        assert "/workspace" in mount_paths
        assert "/data" in mount_paths

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_run_custom_training_workspace_volume_already_present_no_duplicate(
        self, mock_client_fn, _ns
    ):
        mock_client = MagicMock()
        mock_client.train.return_value = "custom-job"
        mock_client_fn.return_value = mock_client

        res = run_custom_training(
            script="print('hello')",
            volumes=[{"name": "my-workspace", "emptyDir": {}}],
            volume_mounts=[{"name": "my-workspace", "mountPath": "/workspace"}],
            confirmed=True,
        )
        assert res["success"] is True

        call_kwargs = mock_client.train.call_args.kwargs
        options = call_kwargs["options"]
        node_spec = _extract_runtime_node_spec(options[0])
        assert len(node_spec.volumes) == 1
        assert node_spec.volumes[0]["name"] == "my-workspace"

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_run_custom_training_with_explicit_name(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "explicit-train-job"
        mock_client_fn.return_value = mock_client

        res = run_custom_training(
            script="print('hello')",
            name="explicit-train-job",
            confirmed=True,
        )
        assert res["success"] is True
        call_kwargs = mock_client.train.call_args.kwargs
        options = call_kwargs["options"]
        name_opts = [o for o in options if isinstance(o, Name)]
        assert len(name_opts) == 1
        assert name_opts[0].name == "explicit-train-job"

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_run_custom_training_sdk_exception_returns_sdk_error(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.side_effect = RuntimeError("Kubernetes rejected custom job")
        mock_client_fn.return_value = mock_client

        res = run_custom_training(script="print('hello')", confirmed=True)
        assert res["success"] is False
        assert res["error_code"] == SDK_ERROR
        assert "Kubernetes rejected custom job" in res["error"]
        assert "Read trainer://guides/troubleshooting" in res["hint"]


# ─── run_container_training extended tests ──────────────────────────────────


class TestRunContainerTrainingExtended:
    def test_run_container_training_sdk_not_available(self):
        with patch("kubeflow_mcp.trainer.api.training._SDK_AVAILABLE", False):
            res = run_container_training(image="torch:latest", confirmed=True)
            assert res["success"] is False
            assert res["error_code"] == SDK_ERROR

    def test_run_container_training_bounds_error(self):
        res = run_container_training(image="torch:latest", num_nodes=9999)
        assert res["success"] is False
        assert res["error_code"] == VALIDATION_ERROR

    def test_run_container_training_resource_error(self):
        res = run_container_training(image="torch:latest", resources_per_node={"cpu": True})
        assert res["success"] is False
        assert res["error_code"] == VALIDATION_ERROR

    def test_run_container_training_preview_with_extra_configs(self):
        res = run_container_training(
            image="torch:latest",
            name="container-preview-test",
            labels={"team": "ml"},
            annotations={"env": "prod"},
            volumes=[{"name": "vol", "emptyDir": {}}],
            volume_mounts=[{"name": "vol", "mountPath": "/data"}],
            node_selector={"gpu": "true"},
            tolerations=[{"key": "gpu", "operator": "Exists"}],
            affinity={"nodeAffinity": {}},
            service_account_name="trainer-sa",
            image_pull_secrets=[{"name": "regcred"}],
            confirmed=False,
        )
        assert res["status"] == "preview"
        cfg = res["config"]
        assert cfg["name"] == "container-preview-test"
        assert cfg["labels"] == {"team": "ml"}
        assert cfg["annotations"] == {"env": "prod"}
        assert cfg["node_selector"] == {"gpu": "true"}
        assert cfg["service_account_name"] == "trainer-sa"

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_run_container_training_confirmed_with_name_and_options(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.return_value = "named-container-job"
        mock_client_fn.return_value = mock_client

        res = run_container_training(
            image="torch:latest",
            name="named-container-job",
            labels={"team": "ml"},
            confirmed=True,
        )
        assert res["success"] is True
        assert res["data"]["job_name"] == "named-container-job"

        call_kwargs = mock_client.train.call_args.kwargs
        options = call_kwargs["options"]
        name_opts = [o for o in options if isinstance(o, Name)]
        assert len(name_opts) == 1
        assert name_opts[0].name == "named-container-job"

    @patch(PATCH_NS_CHECK, return_value=None)
    @patch(PATCH_CLIENT)
    def test_run_container_training_sdk_exception_returns_sdk_error(self, mock_client_fn, _ns):
        mock_client = MagicMock()
        mock_client.train.side_effect = RuntimeError("Container runtime unavailable")
        mock_client_fn.return_value = mock_client

        res = run_container_training(image="torch:latest", confirmed=True)
        assert res["success"] is False
        assert res["error_code"] == SDK_ERROR
        assert "Container runtime unavailable" in res["error"]
        assert "Read trainer://guides/troubleshooting" in res["hint"]


@pytest.mark.parametrize("scope", ["namespace", "cluster", "denied"])
def test_runtime_node_lookup_respects_scope_and_denials(scope: str) -> None:
    from kubeflow_mcp.trainer.api.training import _get_runtime_node_spec

    pod = {"containers": [{"name": "node"}]}
    raw = {
        "spec": {
            "template": {
                "spec": {
                    "replicatedJobs": [
                        {"name": "node", "template": {"spec": {"template": {"spec": pod}}}}
                    ]
                }
            }
        }
    }
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = raw
    api.get_cluster_custom_object.return_value = raw
    if scope != "namespace":
        api.get_namespaced_custom_object.side_effect = ApiException(
            status=404 if scope == "cluster" else 403
        )
    with patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=api):
        if scope == "denied":
            with pytest.raises(ApiException):
                _get_runtime_node_spec("custom-runtime", "workloads")
        else:
            assert _get_runtime_node_spec("custom-runtime", "workloads") == pod
    assert api.get_namespaced_custom_object.call_args.kwargs["namespace"] == "workloads"
    assert api.get_namespaced_custom_object.call_args.kwargs["name"] == "custom-runtime"
    assert api.get_cluster_custom_object.call_count == int(scope == "cluster")


@pytest.mark.parametrize("confirmed", [False, True])
@pytest.mark.parametrize("cluster_scoped", [False, True])
@pytest.mark.parametrize("mount_path", ["/workspace", "/workspace/"])
def test_custom_training_preserves_inherited_workspace(
    confirmed: bool, cluster_scoped: bool, mount_path: str
) -> None:
    pod_spec = {
        "volumes": [{"name": "saved", "persistentVolumeClaim": {"claimName": "checkpoints"}}],
        "containers": [
            {"name": "node", "volumeMounts": [{"name": "saved", "mountPath": mount_path}]}
        ],
    }
    raw = {
        "spec": {
            "template": {
                "spec": {
                    "replicatedJobs": [
                        {"name": "node", "template": {"spec": {"template": {"spec": pod_spec}}}},
                    ]
                }
            }
        }
    }
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = raw
    api.get_cluster_custom_object.return_value = raw
    if cluster_scoped:
        api.get_namespaced_custom_object.side_effect = ApiException(status=404)
    with (
        patch(PATCH_NS_CHECK, return_value=None),
        patch(PATCH_CLIENT) as client,
        patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=api),
    ):
        client.return_value.train.return_value = "workspace-test"
        result = run_custom_training(
            script="print('hello')",
            runtime="pvc-runtime",
            namespace="workloads",
            confirmed=confirmed,
        )
    if confirmed:
        assert result["success"] is True
        assert not any(
            type(option).__name__ == "RuntimePatch"
            for option in client.return_value.train.call_args.kwargs["options"]
        )
    else:
        assert result["status"] == "preview"
        assert "volumes" not in result["config"]
        assert "volume_mounts" not in result["config"]
        client.assert_not_called()
    assert api.get_namespaced_custom_object.call_args.kwargs["namespace"] == "workloads"
    assert api.get_namespaced_custom_object.call_args.kwargs["name"] == "pvc-runtime"
    assert api.get_cluster_custom_object.call_count == int(cluster_scoped)
    assert pod_spec["volumes"][0]["persistentVolumeClaim"]["claimName"] == "checkpoints"


def test_workspace_fallback_avoids_runtime_volume_name_collision() -> None:
    from kubeflow_mcp.trainer.api.training import _ensure_workspace

    node = {
        "containers": [{"name": "node"}],
        "volumes": [
            {"name": "workspace", "persistentVolumeClaim": {"claimName": "data"}},
        ],
    }
    with patch("kubeflow_mcp.trainer.api.training._get_runtime_node_spec", return_value=node):
        volumes, mounts = _ensure_workspace(None, None, runtime="custom", namespace="workloads")
    assert volumes == [{"name": "workspace-1", "emptyDir": {}}]
    assert mounts == [{"name": "workspace-1", "mountPath": "/workspace"}]


@pytest.mark.parametrize("confirmed", [False, True])
def test_custom_training_does_not_replace_unknown_runtime_storage(confirmed: bool) -> None:
    api = MagicMock()
    api.get_namespaced_custom_object.side_effect = ApiException(
        status=403, reason="Cannot read runtime"
    )
    with (
        patch(PATCH_NS_CHECK, return_value=None),
        patch(PATCH_CLIENT) as client,
        patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=api),
    ):
        result = run_custom_training(
            script="print('hello')",
            runtime="private-runtime",
            namespace="workloads",
            confirmed=confirmed,
        )
    assert result["error_code"] == "PERMISSION_DENIED"
    client.assert_not_called()
    api.get_cluster_custom_object.assert_not_called()


@pytest.mark.parametrize("mount_path", ["/workspace", "/workspace/"])
def test_explicit_workspace_does_not_require_runtime_lookup(mount_path: str) -> None:
    from kubeflow_mcp.trainer.api.training import _ensure_workspace

    volumes = [{"name": "data", "persistentVolumeClaim": {"claimName": "checkpoints"}}]
    mounts = [{"name": "data", "mountPath": mount_path}]
    with patch("kubeflow_mcp.trainer.api.training._get_runtime_node_spec") as read:
        assert _ensure_workspace(volumes, mounts, runtime="custom", namespace="workloads") == (
            volumes,
            mounts,
        )
    read.assert_not_called()


@pytest.mark.parametrize(("tool", "config"), _TRAINING_TOOLS)
@pytest.mark.parametrize("confirmed", [False, True])
@pytest.mark.parametrize(
    "storage",
    [
        {"volumes": [{"name": "data", "persistent_volume_claim": {"claim_name": "data"}}]},
        {"volumes": [{"name": "data", "persistentVolumeClaim": {"claim_name": "data"}}]},
        {"volumes": [{"name": "data", "persistentVolumeClaim": {}}]},
        {"volumes": [{"name": "data"}]},
        {
            "volumes": [
                {"name": "data", "emptyDir": {}, "persistentVolumeClaim": {"claimName": "data"}}
            ]
        },
        {"volume_mounts": [{"name": "data", "mount_path": "/workspace"}]},
        {"volume_mounts": [{"name": "data", "mountPath": "relative"}]},
    ],
)
def test_storage_rejected_before_preview_or_submission(tool, config, confirmed, storage):
    with patch(PATCH_CLIENT) as sdk, patch(PATCH_GPU_CHECK) as gpu:
        result = tool(**config, **storage, confirmed=confirmed)
    assert result["error_code"] == "VALIDATION_ERROR"
    sdk.assert_not_called()
    gpu.assert_not_called()


def test_storage_accepts_kubernetes_fields_and_runtime_mounts():
    from kubeflow_mcp.trainer.api.training import _validate_storage

    assert (
        _validate_storage(
            [
                {"name": "checkpoint", "persistentVolumeClaim": {"claimName": "saved.model"}},
                {"name": "tmp", "emptyDir": {}},
            ],
            [{"name": "checkpoint", "mountPath": "/workspace"}],
        )
        is None
    )
    assert _validate_storage(None, [{"name": "runtime-volume", "mountPath": "/runtime"}]) is None


@pytest.mark.parametrize("confirmed", [False, True])
@pytest.mark.parametrize("missing", ["node job", "node container"])
def test_custom_training_reports_missing_runtime_node(missing: str, confirmed: bool) -> None:
    pod = {"containers": [{"name": "sidecar"}]}
    jobs = (
        []
        if missing == "node job"
        else [{"name": "node", "template": {"spec": {"template": {"spec": pod}}}}]
    )
    raw = {"spec": {"template": {"spec": {"replicatedJobs": jobs}}}}
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = raw
    with (
        patch(PATCH_NS_CHECK, return_value=None),
        patch(PATCH_CLIENT) as client,
        patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=api),
    ):
        result = run_custom_training(
            script="print('hello')",
            runtime="invalid-runtime",
            namespace="workloads",
            confirmed=confirmed,
        )
    assert result["success"] is False
    assert "invalid-runtime" in result["error"]
    assert missing in result["error"]
    client.assert_not_called()


@pytest.mark.parametrize("confirmed", [False, True])
def test_custom_training_reports_runtime_not_found(confirmed: bool) -> None:
    api = MagicMock()
    api.get_namespaced_custom_object.side_effect = ApiException(status=404)
    api.get_cluster_custom_object.side_effect = ApiException(status=404)
    with (
        patch(PATCH_NS_CHECK, return_value=None),
        patch(PATCH_CLIENT) as client,
        patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=api),
    ):
        result = run_custom_training(
            script="print('hello')",
            runtime="missing-runtime",
            namespace="workloads",
            confirmed=confirmed,
        )
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "missing-runtime" in result["error"]
    assert "workloads" in result["error"]
    client.assert_not_called()


def test_custom_training_namespace_404_keeps_the_submission_context() -> None:
    with patch(PATCH_NS_CHECK, return_value=None), patch(PATCH_CLIENT) as client:
        client.return_value.train.side_effect = ApiException(status=404, reason="Namespace missing")
        result = run_custom_training(
            script="print('hello')",
            runtime="torch-distributed",
            namespace="missing-namespace",
            volumes=[{"name": "workspace", "emptyDir": {}}],
            volume_mounts=[{"name": "workspace", "mountPath": "/workspace"}],
            confirmed=True,
        )
    assert result["error_code"] == "RESOURCE_NOT_FOUND"
    assert "missing-namespace" in result["error"]
    assert "torch-distributed" in result["error"]
    assert "namespace" in result["hint"]
    client.return_value.train.assert_called_once()


@pytest.mark.parametrize(
    "message,expected",
    [
        (
            'admission webhook "validator.trainer" denied the request: spec.runtimeRef.name: Invalid value: "qwen": QLoRA is not supported',
            "VALIDATION_ERROR",
        ),
        (
            'admission webhook "validator.trainer" denied the request: spec.runtimeRef: Required value',
            "VALIDATION_ERROR",
        ),
        (
            'admission webhook "validator.trainer" denied the request: spec.trainer: Unsupported value',
            "VALIDATION_ERROR",
        ),
        (
            'admission webhook "validator.trainer" denied the request: spec.initializer: Forbidden',
            "VALIDATION_ERROR",
        ),
        (
            'admission webhook "security-policy" denied the request: spec.trainerPolicy: Invalid value',
            "PERMISSION_DENIED",
        ),
        (
            'admission webhook "security-policy" denied the request: service account forbidden',
            "PERMISSION_DENIED",
        ),
        (
            'admission webhook "security-policy" denied the request: spec.serviceAccountName: Invalid value: "admin"',
            "PERMISSION_DENIED",
        ),
        ("User cannot create trainjobs in namespace workloads", "PERMISSION_DENIED"),
    ],
)
@pytest.mark.parametrize("wrapping", ["direct", "cause", "context"])
def test_admission_validation_distinguished_from_permissions(
    message: str, expected: str, wrapping: str
) -> None:
    import json

    from kubeflow_mcp.trainer.api.training import _sdk_error

    api_error = ApiException(status=403)
    api_error.body = json.dumps({"message": message})
    error: Exception = api_error
    if wrapping != "direct":
        error = RuntimeError("Failed to create TrainJob")
        if wrapping == "cause":
            error.__cause__ = api_error
        else:
            error.__context__ = api_error
    result = _sdk_error(error)
    assert result["error_code"] == expected
    if expected == "VALIDATION_ERROR":
        assert result["error"] == message
        assert "service_account_name" not in result["hint"]


@pytest.mark.parametrize(
    "free,expected",
    [
        (1, "no node has that capacity"),
        (None, "Free GPUs are unknown"),
        (0, "cannot fit all requested workers"),
    ],
)
def test_fine_tune_preview_inherits_resources_and_warns_per_pod_fit(free, expected):
    from kubeflow_mcp.trainer.api.training import _fine_tune_resource_preview

    api = MagicMock()
    api.get_namespaced_custom_object.side_effect = ApiException(status=404)
    api.get_cluster_custom_object.return_value = {
        "spec": {
            "template": {
                "spec": {
                    "replicatedJobs": [
                        {
                            "name": "node",
                            "template": {
                                "spec": {
                                    "template": {
                                        "spec": {
                                            "containers": [
                                                {
                                                    "name": "node",
                                                    "resources": {
                                                        "requests": {"cpu": "4", "memory": "16Gi"},
                                                        "limits": {"nvidia.com/gpu": "2"},
                                                    },
                                                }
                                            ]
                                        }
                                    }
                                }
                            },
                        }
                    ]
                }
            }
        }
    }
    cluster = {
        "data": {
            "gpu_total": 2,
            "gpu_available": free,
            "nodes": [{"gpus": 1, "gpu_available": free}, {"gpus": 1, "gpu_available": free}],
        }
    }
    with (
        patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=api),
    ):
        # An explicit 1-GPU override fits capacity but availability can still be unknown.
        result = _fine_tune_resource_preview(
            "torchtune-qwen", "workloads", {"gpu": 1} if free != 1 else None, 1, cluster["data"]
        )
    assert result["effective_resources_per_node"]["cpu"] == "4"
    assert result["effective_resources_per_node"]["nvidia.com/gpu"] == (1 if free != 1 else "2")
    assert any(expected in warning for warning in result["resource_warnings"])
    assert api.get_namespaced_custom_object.call_args.kwargs["namespace"] == "workloads"


def test_resource_preview_keeps_explicit_requests_when_runtime_unreadable():
    from kubeflow_mcp.trainer.api.training import _fine_tune_resource_preview

    api = MagicMock()
    api.get_namespaced_custom_object.side_effect = ApiException(status=403)
    with patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=api):
        result = _fine_tune_resource_preview("torchtune-qwen", "workloads", {"cpu": "2"}, 1, {})
    assert result["effective_resources_per_node"] == {"cpu": "2"}
    assert "incomplete" in result["resource_warnings"][0]
    api.get_cluster_custom_object.assert_not_called()


@pytest.mark.parametrize(
    "resources,expected",
    [
        ({"GPU": 1}, {"nvidia.com/gpu": 1}),
        ({"nvidia.com/gpu": 1}, {"nvidia.com/gpu": 1}),
        (
            {"gpu": 1, "CPU": "2", "Memory": "8Gi", "storage": "10Gi"},
            {"nvidia.com/gpu": 1, "cpu": "2", "memory": "8Gi", "ephemeral-storage": "10Gi"},
        ),
    ],
)
def test_fine_tune_preview_matches_sdk_resources_and_inspects_cluster_once(resources, expected):
    pod_spec = {
        "containers": [
            {
                "name": "node",
                "resources": {
                    "requests": {"cpu": "4"},
                    "limits": {"nvidia.com/gpu": "2"},
                },
            }
        ]
    }
    job = {"name": "node", "template": {"spec": {"template": {"spec": pod_spec}}}}
    api = MagicMock()
    api.get_namespaced_custom_object.return_value = {
        "spec": {
            "template": {
                "spec": {"replicatedJobs": [job]},
            }
        }
    }
    snapshot = {
        "gpu_total": 2,
        "gpu_available": 2,
        "nodes": [
            {"gpus": 1, "gpu_available": 1},
            {"gpus": 1, "gpu_available": 1},
        ],
    }
    with (
        patch(PATCH_NS_CHECK, return_value=None),
        patch(PATCH_CLIENT) as client,
        patch("kubeflow_mcp.common.utils.get_custom_objects_api", return_value=api),
        patch(
            "kubeflow_mcp.trainer.api.planning.get_cluster_resources",
            return_value={"data": snapshot},
        ) as inspect,
    ):
        result = fine_tune(
            "hf://org/model",
            "hf://org/dataset",
            runtime="torchtune-test",
            namespace="workloads",
            resources_per_node=resources,
        )
    assert result["status"] == "preview"
    effective = result["config"]["effective_resources_per_node"]
    assert expected.items() <= effective.items()
    assert "GPU" not in effective
    assert "CPU" not in effective
    assert not any("no node has that capacity" in w for w in result["config"]["resource_warnings"])
    assert result["config"]["resources_per_node"] == resources
    inspect.assert_called_once_with()
    client.assert_not_called()


def test_fine_tune_zero_gpu_snapshot_blocks_before_runtime_inspection():
    with (
        patch(PATCH_NS_CHECK, return_value=None),
        patch("kubeflow_mcp.common.utils.get_custom_objects_api") as api,
        patch(
            "kubeflow_mcp.trainer.api.planning.get_cluster_resources",
            return_value={"data": {"gpu_total": 0}},
        ) as inspect,
    ):
        result = fine_tune("hf://org/model", "hf://org/dataset")
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "No GPUs detected" in result["error"]
    inspect.assert_called_once_with()
    api.assert_not_called()


@pytest.mark.parametrize(
    "failure", [{"success": False, "error": "Unavailable"}, RuntimeError("Unavailable")]
)
def test_fine_tune_preview_preserves_unknown_availability_on_cluster_failure(failure):
    with (
        patch(PATCH_NS_CHECK, return_value=None),
        patch("kubeflow_mcp.trainer.api.planning.get_cluster_resources") as inspect,
    ):
        if isinstance(failure, Exception):
            inspect.side_effect = failure
        else:
            inspect.return_value = failure
        result = fine_tune("hf://org/model", "hf://org/dataset", resources_per_node={"GPU": 1})
    assert result["status"] == "preview"
    assert any("Free GPUs are unknown" in w for w in result["config"]["resource_warnings"])
    inspect.assert_called_once_with()


def test_fine_tune_healthy_resource_preview_has_no_warnings() -> None:
    from kubeflow_mcp.trainer.api.training import _fine_tune_resource_preview

    pod = {"containers": [{"name": "node", "resources": {"requests": {"nvidia.com/gpu": "1"}}}]}
    cluster = {"gpu_available": 2, "nodes": [{"gpus": 2, "gpu_available": 2}]}
    with patch("kubeflow_mcp.trainer.api.training._get_runtime_node_spec", return_value=pod):
        result = _fine_tune_resource_preview("torchtune", "workloads", None, 1, cluster)
    assert result["effective_resources_per_node"] == {"nvidia.com/gpu": "1"}
    assert result["resource_warnings"] == []


@pytest.mark.parametrize(
    "resources,runtime_gpus,warned",
    [
        ({"cpu": "8", "memory": "32Gi"}, 2, True),
        ({"memory": "32Gi"}, 2, True),
        ({"CPU": "8"}, 2, True),
        ({"gpu": 2, "cpu": "8"}, 2, False),
        ({"GPU": 2}, 2, False),
        ({"nvidia.com/gpu": 2}, 2, False),
        ({"mig-1g.5gb": 1}, 2, False),
        ({"memory": "32Gi"}, 0, False),
        (None, 2, False),
    ],
)
def test_preview_warns_when_overrides_omit_inherited_gpus(
    resources: dict | None, runtime_gpus: int, warned: bool
) -> None:
    pod = {
        "containers": [
            {
                "name": "node",
                "resources": {"limits": {"nvidia.com/gpu": str(runtime_gpus)}},
            }
        ]
    }
    cluster = {
        "data": {
            "gpu_total": 2,
            "gpu_available": 2,
            "nodes": [{"gpus": 2, "gpu_available": 2}],
        }
    }
    original = dict(resources) if resources is not None else None
    with (
        patch(PATCH_NS_CHECK, return_value=None),
        patch(PATCH_CLIENT) as sdk,
        patch("kubeflow_mcp.trainer.api.training._get_runtime_node_spec", return_value=pod),
        patch(
            "kubeflow_mcp.trainer.api.planning.get_cluster_resources", return_value=cluster
        ) as inspect,
    ):
        result = fine_tune(
            "hf://org/model",
            "hf://org/dataset",
            runtime="torchtune-test",
            resources_per_node=resources,
        )
    assert result["status"] == "preview"
    config = result["config"]
    assert any("Set gpu explicitly" in w for w in config["resource_warnings"]) == warned
    assert int(config["effective_resources_per_node"]["nvidia.com/gpu"]) == runtime_gpus
    assert config.get("resources_per_node") == original
    assert resources == original
    inspect.assert_called_once_with()
    sdk.assert_not_called()
