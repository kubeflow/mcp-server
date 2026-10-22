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
from unittest.mock import MagicMock, patch

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from kubernetes.client.exceptions import ApiException
from tests.common import FAILED, PREVIEW, VALIDATION_ERROR, TestCase, assert_test_case

from kubeflow_mcp.trainer.api.training import (
    _build_fine_tune_config,
    _build_runtime_patch,
    _forwarded_train_args,
    _has_required_params,
    _has_train_call,
    _inject_trainer_hf_home,
    _make_train_func,
    _should_apply_hf_dataset_workaround,
    _uncalled_train_call,
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
