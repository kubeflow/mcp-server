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

"""Tests for planning helpers: HuggingFace model ID validation and suggestions."""

import importlib.metadata
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from huggingface_hub.utils import hf_raise_for_status
from packaging.version import Version
from tests.common import TestCase

from kubeflow_mcp.common.constants import (
    KUBEFLOW_SDK_VERSION_MIN,
    KUBEFLOW_SDK_VERSION_SPEC,
    MIN_TRAINER_CRD_VERSION,
    TRAINER_CRD_NAME,
)
from kubeflow_mcp.trainer.api.planning import (
    _check_k8s_version,
    _check_sdk_version,
    _check_trainer_crd,
    _detect_platform,
    _estimate_from_params,
    _get_model_info_from_hf,
    _parse_k8s_version,
    _suggest_hf_model_ids,
    check_compatibility,
    estimate_resources,
    get_cluster_resources,
    pre_flight,
)


def test_check_sdk_version_rejects_sdk_outside_supported_range():
    checks = {}
    blockers = []
    minimum = list(Version(KUBEFLOW_SDK_VERSION_MIN).release)
    for index in reversed(range(len(minimum))):
        if minimum[index] > 0:
            minimum[index] -= 1
            break
    unsupported_version = ".".join(str(part) for part in minimum)

    with patch("importlib.metadata.version", return_value=unsupported_version):
        _check_sdk_version(checks, blockers)

    assert checks["kubeflow_sdk"] == {
        "status": "fail",
        "version": unsupported_version,
        "required": KUBEFLOW_SDK_VERSION_SPEC,
    }
    assert blockers == [
        f"kubeflow {unsupported_version} is installed; "
        f"kubeflow-mcp requires kubeflow{KUBEFLOW_SDK_VERSION_SPEC}"
    ]


def _fake_models(*ids):
    return [SimpleNamespace(id=model_id) for model_id in ids]


def _repo_not_found():
    # The concrete exception model_info raises for a nonexistent repo.
    request = httpx.Request("GET", "https://huggingface.co/api/models/x")
    return RepositoryNotFoundError(
        "Repository Not Found", response=httpx.Response(404, request=request)
    )


def test_suggest_normalizes_ollama_tag():
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct")
        suggestions = _suggest_hf_model_ids("qwen3:8b")

    assert suggestions == ["Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"]
    # The Ollama-style ":8b" tag is dropped before searching the Hub.
    assert mock_list.call_args.kwargs["search"] == "qwen3"


def test_suggest_drops_hf_prefix():
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models("google/gemma-2b")
        _suggest_hf_model_ids("hf://google/gemma:2b")

    assert mock_list.call_args.kwargs["search"] == "google/gemma"


def test_suggest_returns_empty_when_hub_errors():
    with patch("huggingface_hub.list_models", side_effect=RuntimeError("offline")):
        assert _suggest_hf_model_ids("qwen3:8b") == []


def test_suggest_returns_empty_for_blank_input():
    # No Hub call is needed when normalization leaves nothing to search for.
    with patch("huggingface_hub.list_models") as mock_list:
        assert _suggest_hf_model_ids("hf://") == []
    mock_list.assert_not_called()


def test_invalid_format_attaches_suggestions():
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct")
        result = _get_model_info_from_hf("qwen3:8b")

    assert result["error"] == "Invalid HuggingFace model ID format: 'qwen3:8b'"
    assert result["suggestions"] == ["Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"]


def test_invalid_format_omits_suggestions_when_none_found():
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models()
        result = _get_model_info_from_hf("qwen3:8b")

    assert result["error"] == "Invalid HuggingFace model ID format: 'qwen3:8b'"
    assert "suggestions" not in result


def test_invalid_format_omits_suggestions_when_hub_errors():
    with patch("huggingface_hub.list_models", side_effect=RuntimeError("offline")):
        result = _get_model_info_from_hf("not a model id")

    assert "Invalid HuggingFace model ID format" in result["error"]
    assert "suggestions" not in result


# Tool-boundary tests: estimate_resources re-wraps the helper's error, so verify
# the suggestions actually survive into the user-facing response. pre_flight()
# delegates model handling to estimate_resources(), so this covers both tools.
def test_estimate_resources_surfaces_suggestions_at_tool_boundary():
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct")
        result = estimate_resources("qwen3:8b")

    assert result["success"] is False
    assert "Invalid HuggingFace model ID format" in result["error"]
    assert result["details"]["suggestions"] == ["Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"]


def test_estimate_resources_omits_suggestions_when_none_found():
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models()
        result = estimate_resources("qwen3:8b")

    assert result["success"] is False
    assert "suggestions" not in result["details"]


def test_suggest_ranks_by_downloads():
    # Ranking by downloads surfaces canonical repos ahead of community fine-tunes.
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models("Qwen/Qwen3-8B")
        _suggest_hf_model_ids("qwen3:8b")

    assert mock_list.call_args.kwargs.get("sort") == "downloads"


def test_suggest_falls_back_to_model_name_when_org_typo_finds_nothing():
    # A typo'd org (meta-lama vs meta-llama) makes the full "org/name" search
    # empty; retry on just the model name so the user still gets a suggestion.
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.side_effect = [_fake_models(), _fake_models("meta-llama/Llama-3.2-1B")]
        suggestions = _suggest_hf_model_ids("meta-lama/Llama-3")

    assert suggestions == ["meta-llama/Llama-3.2-1B"]
    assert [c.kwargs["search"] for c in mock_list.call_args_list] == [
        "meta-lama/Llama-3",
        "Llama-3",
    ]


def test_suggest_skips_name_fallback_when_name_too_short():
    # A very short model name ("ab") makes a bare search too broad to be a useful
    # "did you mean", so the name-only retry is skipped: only the full term runs.
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models()
        suggestions = _suggest_hf_model_ids("wrongorg/ab")

    assert suggestions == []
    assert mock_list.call_count == 1
    assert mock_list.call_args.kwargs["search"] == "wrongorg/ab"


def test_not_found_id_attaches_suggestions():
    # A valid-format but nonexistent id fails at model_info (not the regex check),
    # and should still get "did you mean" suggestions.
    with (
        patch("huggingface_hub.model_info", side_effect=_repo_not_found()),
        patch("huggingface_hub.list_models") as mock_list,
    ):
        mock_list.return_value = _fake_models("meta-llama/Llama-3.2-1B")
        result = _get_model_info_from_hf("meta-lama/Llama-3")

    assert "error" in result
    assert result["suggestions"] == ["meta-llama/Llama-3.2-1B"]


def test_not_found_id_omits_suggestions_when_hub_errors():
    with (
        patch("huggingface_hub.model_info", side_effect=_repo_not_found()),
        patch("huggingface_hub.list_models", side_effect=RuntimeError("offline")),
    ):
        result = _get_model_info_from_hf("meta-lama/Llama-3")

    assert "error" in result
    assert "suggestions" not in result


def test_estimate_resources_surfaces_suggestions_on_not_found():
    with (
        patch("huggingface_hub.model_info", side_effect=_repo_not_found()),
        patch("huggingface_hub.list_models") as mock_list,
    ):
        mock_list.return_value = _fake_models("meta-llama/Llama-3.2-1B")
        result = estimate_resources("meta-lama/Llama-3")

    assert result["success"] is False
    assert result["details"]["suggestions"] == ["meta-llama/Llama-3.2-1B"]


def test_non_not_found_error_skips_suggestions():
    # Only a missing repo should trigger suggestions. Other failures (timeout,
    # auth/rate-limit, metadata errors) surface unchanged and make no extra Hub
    # request, so an outage or a gated model does not get misleading matches.
    with (
        patch("huggingface_hub.model_info", side_effect=TimeoutError("timed out")),
        patch("huggingface_hub.list_models") as mock_list,
    ):
        result = _get_model_info_from_hf("meta-llama/Llama-3.2-1B")

    assert "error" in result
    assert "suggestions" not in result
    mock_list.assert_not_called()


# ─── Pure helpers (version parse / resource estimate) ───────────────────────


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="parses v1.29.3",
            config={"git_version": "v1.29.3"},
            expected_output=(1, 29),
        ),
        TestCase(
            name="parses without v prefix",
            config={"git_version": "1.30.0"},
            expected_output=(1, 30),
        ),
        TestCase(
            name="invalid version returns None",
            config={"git_version": "garbage"},
            expected_output=None,
        ),
    ],
)
def test_parse_k8s_version(test_case):
    assert _parse_k8s_version(**test_case.config) == test_case.expected_output


def _version_api(git_version: str) -> MagicMock:
    api = MagicMock()
    api.get_code.return_value = SimpleNamespace(git_version=git_version)
    return api


def test_check_k8s_version_passes_for_supported_version():
    checks, blockers = {}, []

    _check_k8s_version(checks, blockers, lambda: _version_api("v1.29.3"), timeout=5)

    assert checks["kubernetes_version"]["status"] == "pass"
    assert checks["kubernetes_version"]["version"] == "v1.29.3"
    assert blockers == []


def test_check_k8s_version_blocks_version_below_minimum():
    checks, blockers = {}, []

    _check_k8s_version(checks, blockers, lambda: _version_api("v1.25.0"), timeout=5)

    assert checks["kubernetes_version"]["status"] == "fail"
    assert len(blockers) == 1
    assert "v1.25.0 is below minimum v1.27" in blockers[0]


def test_check_k8s_version_warns_when_version_cannot_be_parsed():
    checks, blockers = {}, []

    _check_k8s_version(checks, blockers, lambda: _version_api("garbage"), timeout=5)

    assert checks["kubernetes_version"]["status"] == "warn"
    assert blockers == []


def test_check_k8s_version_blocks_when_api_is_unreachable():
    checks, blockers = {}, []
    api = MagicMock()
    api.get_code.side_effect = ConnectionError("Connection refused")

    _check_k8s_version(checks, blockers, lambda: api, timeout=5)

    assert checks["kubernetes_version"]["status"] == "fail"
    assert "Connection refused" in checks["kubernetes_version"]["error"]
    assert blockers == ["Cannot reach Kubernetes API: Connection refused"]


def _core_api_with_node_labels(*labels: dict | None) -> MagicMock:
    api = MagicMock()
    api.list_node.return_value = SimpleNamespace(
        items=[SimpleNamespace(metadata=SimpleNamespace(labels=lbl)) for lbl in labels]
    )
    return api


@pytest.mark.parametrize(
    ("label", "platform"),
    [
        ("node.openshift.io/os_id", "openshift"),
        ("eks.amazonaws.com/nodegroup", "eks"),
        ("cloud.google.com/gke-nodepool", "gke"),
    ],
)
def test_detect_platform_from_node_labels(label, platform):
    checks, recommendations = {}, []
    api = _core_api_with_node_labels({label: "x"})

    result = _detect_platform(checks, recommendations, lambda: api, timeout=5)

    assert result == platform
    assert checks["platform"] == {"status": "info", "detected": platform}
    assert len(recommendations) == 1
    assert "trainer://guides/platform-fixes" in recommendations[0]


def test_detect_platform_defaults_to_kubernetes():
    checks, recommendations = {}, []
    api = _core_api_with_node_labels({"kubernetes.io/os": "linux"}, None)

    result = _detect_platform(checks, recommendations, lambda: api, timeout=5)

    assert result == "kubernetes"
    assert checks["platform"] == {"status": "info", "detected": "kubernetes"}
    assert recommendations == []


def test_detect_platform_returns_unknown_when_api_fails():
    checks, recommendations = {}, []
    api = MagicMock()
    api.list_node.side_effect = ConnectionError("Connection refused")

    result = _detect_platform(checks, recommendations, lambda: api, timeout=5)

    assert result == "unknown"
    assert checks["platform"]["status"] == "warn"
    assert "Connection refused" in checks["platform"]["error"]
    assert recommendations == []


def test_estimate_from_params_small_model():
    result = _estimate_from_params(7e9, batch_size=4, quantization="bf16")
    assert result["gpu_count"] >= 1
    assert result["gpu_memory_gb"] >= 1
    assert result["params_billions"] == 7.0
    assert result["quantization"] == "bf16"
    assert "weights_gb" in result["breakdown"]


# ─── Error code classification ──────────────────────────────────────────────


def test_estimate_resources_returns_validation_error_for_invalid_format():
    """Invalid model ID format (not org/name) must return VALIDATION_ERROR, not SDK_ERROR."""
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models()
        result = estimate_resources("random gibberish")

    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    assert "Invalid HuggingFace model ID format" in result["error"]


def test_estimate_resources_returns_validation_error_for_confirmed_404():
    """A model ID the Hub confirms missing with a 404 is a user typo, so VALIDATION_ERROR."""
    with (
        patch("huggingface_hub.model_info", side_effect=_repo_not_found()),
        patch("huggingface_hub.list_models") as mock_list,
    ):
        mock_list.return_value = _fake_models("meta-llama/Llama-3.2-1B")
        result = estimate_resources("meta-lama/Llama-3")

    assert result["success"] is False
    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["details"]["suggestions"] == ["meta-llama/Llama-3.2-1B"]


def _hub_error(error_cls: type[RepositoryNotFoundError], status_code: int):
    request = httpx.Request("GET", "https://huggingface.co/api/models/x")
    return error_cls("Hub error", response=httpx.Response(status_code, request=request))


@pytest.mark.parametrize(
    "hub_error",
    [
        _hub_error(RepositoryNotFoundError, 401),  # private repo or missing auth
        _hub_error(GatedRepoError, 403),  # gated repo
    ],
)
def test_estimate_resources_keeps_sdk_error_for_access_failures(hub_error):
    """An inaccessible repo is not a typo, so it must not become VALIDATION_ERROR."""
    with (
        patch("huggingface_hub.model_info", side_effect=hub_error),
        patch("huggingface_hub.list_models", return_value=[]),
    ):
        result = estimate_resources("meta-llama/Llama-3.2-1B")

    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"


def test_estimate_resources_keeps_sdk_error_for_anonymous_401():
    """The Hub answers an anonymous request for a missing model with 401, the same
    as for a private repo, so it cannot be classified as bad input."""
    request = httpx.Request("GET", "https://huggingface.co/api/models/meta-lama/Llama-3")
    response = httpx.Response(
        401, headers={"X-Error-Message": "Invalid username or password."}, request=request
    )
    with pytest.raises(RepositoryNotFoundError) as hub_error:
        hf_raise_for_status(response)

    with (
        patch("huggingface_hub.model_info", side_effect=hub_error.value),
        patch("huggingface_hub.list_models", return_value=[]),
    ):
        result = estimate_resources("meta-lama/Llama-3")

    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"


def test_estimate_resources_returns_sdk_error_for_api_failure():
    """Actual API/network failures must still return SDK_ERROR."""
    with patch(
        "kubeflow_mcp.trainer.api.planning._get_model_info_from_hf",
        return_value={"error": "Connection timed out"},
    ):
        result = estimate_resources("meta-llama/Llama-3.2-1B")

    assert result["success"] is False
    assert result["error_code"] == "SDK_ERROR"


@pytest.mark.parametrize("deny_pods", [False, True])
def test_cluster_gpu_capacity_separate_from_allocations(deny_pods):
    from kubernetes import client as k8s
    from kubernetes.client.exceptions import ApiException

    from kubeflow_mcp.trainer.api.planning import get_cluster_resources

    api = MagicMock()
    api.list_node.return_value.items = [
        k8s.V1Node(
            metadata=k8s.V1ObjectMeta(name=name),
            status=k8s.V1NodeStatus(
                allocatable={"nvidia.com/gpu": "1", "cpu": "8", "memory": "32Gi"}
            ),
        )
        for name in ("gpu-a", "gpu-b")
    ]

    def pod(phase, node):
        return k8s.V1Pod(
            spec=k8s.V1PodSpec(
                node_name=node,
                containers=[
                    k8s.V1Container(
                        name="node",
                        resources=k8s.V1ResourceRequirements(limits={"nvidia.com/gpu": "1"}),
                    )
                ],
            ),
            status=k8s.V1PodStatus(phase=phase),
        )

    api.list_pod_for_all_namespaces.return_value.metadata = k8s.V1ListMeta()
    api.list_pod_for_all_namespaces.return_value.items = [
        pod("Running", "gpu-a"),
        pod("Pending", None),
        pod("Succeeded", "gpu-b"),
        pod("Failed", "gpu-b"),
    ]
    if deny_pods:
        api.list_pod_for_all_namespaces.side_effect = ApiException(status=403)
    with patch("kubeflow_mcp.common.utils.get_core_v1_api", return_value=api):
        result = get_cluster_resources()["data"]
    assert result["gpu_total"] == 2
    assert result["gpu_available"] == (None if deny_pods else 1)
    assert result["gpu_allocated"] == (None if deny_pods else 1)
    assert result["nodes"][0]["gpu_available"] == (None if deny_pods else 0)
    assert result["nodes"][1]["gpu_available"] == (None if deny_pods else 1)
    assert api.list_pod_for_all_namespaces.call_args.kwargs["_request_timeout"] > 0


def test_gpu_requests_include_init_peak_and_restartable_sidecars():
    from kubernetes import client as k8s

    from kubeflow_mcp.trainer.api.planning import _pod_gpu_request

    def container(name, gpus, restart_policy=None):
        return k8s.V1Container(
            name=name,
            restart_policy=restart_policy,
            resources=k8s.V1ResourceRequirements(requests={"nvidia.com/gpu": str(gpus)}),
        )

    pod = k8s.V1Pod(
        spec=k8s.V1PodSpec(
            containers=[container("trainer", "1e0")],
            init_containers=[container("sidecar", 1, "Always"), container("initializer", 3)],
        )
    )
    assert _pod_gpu_request(pod) == 4
    pod.spec.init_containers = [container("sidecar", 2, "Always")]
    assert _pod_gpu_request(pod) == 3


@pytest.mark.parametrize("outcome", ["complete", "page-error", "page-cap"])
def test_gpu_allocations_require_a_complete_paginated_snapshot(outcome: str) -> None:
    from kubernetes import client as k8s
    from kubernetes.client.exceptions import ApiException

    from kubeflow_mcp.trainer.api.planning import get_cluster_resources

    api = MagicMock()
    api.list_node.return_value.items = [
        k8s.V1Node(
            metadata=k8s.V1ObjectMeta(name="gpu-a"),
            status=k8s.V1NodeStatus(allocatable={"nvidia.com/gpu": "2"}),
        )
    ]
    pod = k8s.V1Pod(
        spec=k8s.V1PodSpec(
            node_name="gpu-a",
            containers=[
                k8s.V1Container(
                    name="trainer",
                    resources=k8s.V1ResourceRequirements(requests={"nvidia.com/gpu": "1"}),
                )
            ],
        ),
        status=k8s.V1PodStatus(phase="Running"),
    )
    first = k8s.V1PodList(items=[pod], metadata=k8s.V1ListMeta(_continue="next-page"))
    last = k8s.V1PodList(items=[pod], metadata=k8s.V1ListMeta())
    if outcome == "page-error":
        last = ApiException(status=410, reason="Expired continuation token")
    api.list_pod_for_all_namespaces.side_effect = [first, last]
    if outcome == "page-cap":
        api.list_pod_for_all_namespaces.side_effect = None
        api.list_pod_for_all_namespaces.return_value = first
    with (
        patch("kubeflow_mcp.common.utils.get_core_v1_api", return_value=api),
        patch("kubeflow_mcp.trainer.constants.MAX_GPU_POD_PAGES", 2),
    ):
        result = get_cluster_resources()["data"]
    assert result["gpu_total"] == 2
    assert result["gpu_allocated"] == (2 if outcome == "complete" else None)
    assert result["gpu_available"] == (0 if outcome == "complete" else None)
    calls = api.list_pod_for_all_namespaces.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["limit"] > 0
    assert calls[0].kwargs["_continue"] is None
    assert calls[1].kwargs["_continue"] == "next-page"
    assert calls[0].kwargs["field_selector"] == "status.phase!=Succeeded,status.phase!=Failed"


# ─── _check_trainer_crd ─────────────────────────────────────────────────────


class TestCheckTrainerCrd:
    def test_check_trainer_crd_passes_when_served_version_matches(self):
        checks = {}
        blockers = []
        crd = SimpleNamespace(
            spec=SimpleNamespace(
                versions=[
                    SimpleNamespace(name=MIN_TRAINER_CRD_VERSION, served=True),
                    SimpleNamespace(name="v1alpha1", served=False),
                ]
            )
        )
        api_ext_v1 = MagicMock()
        api_ext_v1.read_custom_resource_definition.return_value = crd

        with patch("kubernetes.client.ApiextensionsV1Api", return_value=api_ext_v1):
            installed = _check_trainer_crd(
                checks,
                blockers,
                get_custom_objects_api=lambda: MagicMock(api_client=MagicMock()),
                timeout=5,
            )

        assert installed is True
        assert checks["trainer_crd"]["status"] == "pass"
        assert checks["trainer_crd"]["crd_name"] == TRAINER_CRD_NAME
        assert checks["trainer_api_version"]["status"] == "pass"
        assert blockers == []

    def test_check_trainer_crd_blocks_when_served_version_mismatched(self):
        checks = {}
        blockers = []
        crd = SimpleNamespace(
            spec=SimpleNamespace(
                versions=[
                    SimpleNamespace(name="v1beta1", served=True),
                ]
            )
        )
        api_ext_v1 = MagicMock()
        api_ext_v1.read_custom_resource_definition.return_value = crd

        with patch("kubernetes.client.ApiextensionsV1Api", return_value=api_ext_v1):
            installed = _check_trainer_crd(
                checks,
                blockers,
                get_custom_objects_api=lambda: MagicMock(api_client=MagicMock()),
                timeout=5,
            )

        assert installed is True
        assert checks["trainer_crd"]["status"] == "pass"
        assert checks["trainer_api_version"]["status"] == "fail"
        assert len(blockers) == 1
        assert MIN_TRAINER_CRD_VERSION in blockers[0]

    def test_check_trainer_crd_fails_when_not_installed(self):
        checks = {}
        blockers = []
        api_ext_v1 = MagicMock()
        api_ext_v1.read_custom_resource_definition.side_effect = RuntimeError("CRD not found")

        with patch("kubernetes.client.ApiextensionsV1Api", return_value=api_ext_v1):
            installed = _check_trainer_crd(
                checks,
                blockers,
                get_custom_objects_api=lambda: MagicMock(api_client=MagicMock()),
                timeout=5,
            )

        assert installed is False
        assert checks["trainer_crd"]["status"] == "fail"
        assert checks["trainer_api_version"]["status"] == "fail"
        assert len(blockers) == 1
        assert "Kubeflow Trainer CRD not installed" in blockers[0]


# ─── _check_sdk_version edge cases ──────────────────────────────────────────


class TestCheckSdkVersionEdgeCases:
    def test_check_sdk_version_package_not_found(self):
        checks = {}
        blockers = []
        with patch(
            "importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError
        ):
            _check_sdk_version(checks, blockers)

        assert checks["kubeflow_sdk"] == {"status": "fail", "error": "Not installed"}
        assert len(blockers) == 1
        assert "kubeflow not installed" in blockers[0]

    def test_check_sdk_version_import_error_warns(self):
        checks = {}
        blockers = []
        with patch("importlib.metadata.version", side_effect=ImportError("packaging missing")):
            _check_sdk_version(checks, blockers)

        assert checks["kubeflow_sdk"]["status"] == "warn"
        assert "packaging library not available" in checks["kubeflow_sdk"]["detail"]


# ─── check_compatibility ───────────────────────────────────────────────────


class TestCheckCompatibility:
    @patch("kubeflow_mcp.trainer.api.planning._detect_platform", return_value="kubernetes")
    @patch("kubeflow_mcp.trainer.api.planning._check_sdk_version")
    @patch("kubeflow_mcp.trainer.api.planning._check_trainer_crd", return_value=True)
    @patch("kubeflow_mcp.trainer.api.planning._check_k8s_version")
    def test_check_compatibility_success(self, mock_k8s, mock_crd, mock_sdk, mock_platform):
        res = check_compatibility()
        assert res["success"] is True
        data = res["data"]
        assert data["compatible"] is True
        assert data["platform"] == "kubernetes"
        assert "Environment compatible" in data["recommendations"][0]

    @patch("kubeflow_mcp.trainer.api.planning._detect_platform", return_value="kubernetes")
    @patch("kubeflow_mcp.trainer.api.planning._check_sdk_version")
    @patch("kubeflow_mcp.trainer.api.planning._check_k8s_version")
    def test_check_compatibility_blockers_found(self, mock_k8s, mock_sdk, mock_platform):
        def add_blocker(checks, blockers, *args, **kwargs):
            blockers.append("Trainer CRD missing")
            return False

        with patch("kubeflow_mcp.trainer.api.planning._check_trainer_crd", side_effect=add_blocker):
            res = check_compatibility()

        assert res["success"] is True
        data = res["data"]
        assert data["compatible"] is False
        assert "Trainer CRD missing" in data["blockers"]

    @patch(
        "kubeflow_mcp.trainer.api.planning._check_k8s_version",
        side_effect=RuntimeError("Cluster dead"),
    )
    def test_check_compatibility_handles_exception(self, mock_k8s):
        res = check_compatibility()
        assert res["success"] is False
        assert res["error_code"] == "KUBERNETES_ERROR"
        assert "Cluster dead" in res["error"]


# ─── get_cluster_resources ─────────────────────────────────────────────────


class TestGetClusterResources:
    def test_get_cluster_resources_success_with_gpus(self):
        node1 = SimpleNamespace(
            metadata=SimpleNamespace(name="node-gpu"),
            status=SimpleNamespace(
                allocatable={"nvidia.com/gpu": "4", "cpu": "16", "memory": "64Gi"}
            ),
        )
        node2 = SimpleNamespace(
            metadata=SimpleNamespace(name="node-cpu"),
            status=SimpleNamespace(allocatable={"cpu": "8", "memory": "32Gi"}),
        )
        v1_mock = MagicMock()
        v1_mock.list_node.return_value = SimpleNamespace(items=[node1, node2])

        with patch("kubeflow_mcp.common.utils.get_core_v1_api", return_value=v1_mock):
            res = get_cluster_resources()

        assert res["success"] is True
        data = res["data"]
        assert data["gpu_total"] == 4
        assert data["nodes_with_gpu"] == 1
        assert data["node_count"] == 2
        assert len(data["nodes"]) == 2
        assert data["nodes"][0]["name"] == "node-gpu"
        assert data["nodes"][0]["gpus"] == 4
        assert "gpus" not in data["nodes"][1]

    def test_get_cluster_resources_cpu_only(self):
        node = SimpleNamespace(
            metadata=SimpleNamespace(name="node-cpu-only"),
            status=SimpleNamespace(allocatable={"cpu": "4", "memory": "16Gi"}),
        )
        v1_mock = MagicMock()
        v1_mock.list_node.return_value = SimpleNamespace(items=[node])

        with patch("kubeflow_mcp.common.utils.get_core_v1_api", return_value=v1_mock):
            res = get_cluster_resources()

        assert res["success"] is True
        data = res["data"]
        assert data["gpu_total"] == 0
        assert data["nodes_with_gpu"] == 0
        assert data["node_count"] == 1

    def test_get_cluster_resources_exception_handling(self):
        v1_mock = MagicMock()
        v1_mock.list_node.side_effect = ConnectionError("K8s API down")

        with patch("kubeflow_mcp.common.utils.get_core_v1_api", return_value=v1_mock):
            res = get_cluster_resources()

        assert res["success"] is False
        assert res["error_code"] == "KUBERNETES_ERROR"
        assert "K8s API down" in res["error"]


# ─── estimate_resources & model parameter parsing ───────────────────────────


class TestEstimateResourcesExtended:
    def test_get_model_info_from_hf_safetensors_params(self):
        info = SimpleNamespace(
            id="test/safetensors-model",
            safetensors=SimpleNamespace(total=7000000000),
            card_data=None,
            library_name="transformers",
            pipeline_tag="text-generation",
        )
        with patch("huggingface_hub.model_info", return_value=info):
            res = _get_model_info_from_hf("test/safetensors-model")

        assert res["model_id"] == "test/safetensors-model"
        assert res["params"] == 7000000000
        assert res["library"] == "transformers"
        assert res["pipeline"] == "text-generation"

    def test_get_model_info_from_hf_card_data_params(self):
        info = SimpleNamespace(
            id="test/card-model",
            safetensors=None,
            card_data=SimpleNamespace(num_parameters=3000000000),
            library_name=None,
            pipeline_tag=None,
        )
        with patch("huggingface_hub.model_info", return_value=info):
            res = _get_model_info_from_hf("test/card-model")

        assert res["model_id"] == "test/card-model"
        assert res["params"] == 3000000000

    def test_estimate_from_params_all_gpu_memory_tiers(self):
        # Tier 1: <= 8GB
        t1 = _estimate_from_params(1e9, batch_size=4, quantization="bf16")
        assert t1["gpu_type"] == "8GB (RTX 3070/4070)"
        assert t1["gpu_count"] == 1

        # Tier 2: <= 16GB
        t2 = _estimate_from_params(5e9, batch_size=4, quantization="bf16")
        assert t2["gpu_type"] == "16GB (T4/RTX 4080)"
        assert t2["gpu_count"] == 1

        # Tier 3: <= 24GB
        t3 = _estimate_from_params(7.5e9, batch_size=4, quantization="bf16")
        assert t3["gpu_type"] == "24GB (A10/RTX 3090)"
        assert t3["gpu_count"] == 1

        # Tier 4: <= 40GB
        t4 = _estimate_from_params(13e9, batch_size=4, quantization="bf16")
        assert t4["gpu_type"] == "40GB (A100-40GB)"
        assert t4["gpu_count"] == 1

        # Tier 5: <= 80GB
        t5 = _estimate_from_params(25e9, batch_size=4, quantization="bf16")
        assert t5["gpu_type"] == "80GB (A100-80GB/H100)"
        assert t5["gpu_count"] == 1

        # Tier 6: > 80GB (multi-GPU)
        t6 = _estimate_from_params(70e9, batch_size=4, quantization="bf16")
        assert "80GB GPUs" in t6["gpu_type"]
        assert t6["gpu_count"] >= 2

    def test_estimate_resources_input_validation_errors(self):
        # Invalid num_workers
        res_workers = estimate_resources("google/gemma-2b", num_workers=0)
        assert res_workers["success"] is False
        assert res_workers["error_code"] == "VALIDATION_ERROR"
        assert "num_workers must be between 1 and 100" in res_workers["error"]

        # Invalid batch_size
        res_batch = estimate_resources("google/gemma-2b", batch_size=2048)
        assert res_batch["success"] is False
        assert res_batch["error_code"] == "VALIDATION_ERROR"
        assert "batch_size must be between 1 and 1024" in res_batch["error"]

        # Invalid quantization
        res_quant = estimate_resources("google/gemma-2b", quantization="invalid_quant")
        assert res_quant["success"] is False
        assert res_quant["error_code"] == "VALIDATION_ERROR"
        assert "Invalid quantization" in res_quant["error"]

    def test_estimate_resources_model_missing_params(self):
        with patch(
            "kubeflow_mcp.trainer.api.planning._get_model_info_from_hf",
            return_value={"model_id": "google/gemma-2b", "params": None},
        ):
            res = estimate_resources("google/gemma-2b")

        assert res["success"] is False
        assert res["error_code"] == "SDK_ERROR"
        assert "parameter count not available" in res["error"]

    def test_estimate_resources_success(self):
        with patch(
            "kubeflow_mcp.trainer.api.planning._get_model_info_from_hf",
            return_value={"model_id": "google/gemma-2b", "params": 2e9},
        ):
            res = estimate_resources(
                "hf://google/gemma-2b", num_workers=2, batch_size=4, quantization="bf16"
            )

        assert res["success"] is True
        data = res["data"]
        assert data["model"] == "hf://google/gemma-2b"
        assert data["params_billions"] == 2.0
        assert data["total_gpu"] == data["gpu_per_worker"] * 2
        assert "breakdown" in data

    def test_estimate_resources_unexpected_exception(self):
        with patch(
            "kubeflow_mcp.trainer.api.planning._get_model_info_from_hf",
            side_effect=RuntimeError("HF Hub crash"),
        ):
            res = estimate_resources("google/gemma-2b")

        assert res["success"] is False
        assert res["error_code"] == "SDK_ERROR"
        assert "HF Hub crash" in res["error"]


# ─── pre_flight ─────────────────────────────────────────────────────────────


class TestPreFlight:
    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    def test_pre_flight_blocker_short_circuit(self, mock_compat):
        mock_compat.return_value = {
            "success": True,
            "data": {
                "compatible": False,
                "blockers": ["Kubernetes version below minimum v1.27"],
                "platform": "kubernetes",
            },
        }
        res = pre_flight()
        assert res["success"] is True
        data = res["data"]
        assert data["next_steps"] == ["Blocker: Kubernetes version below minimum v1.27"]
        assert "cluster" not in data

    @patch("kubeflow_mcp.trainer.api.discovery.list_runtimes")
    @patch("kubeflow_mcp.trainer.api.planning.get_cluster_resources")
    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    def test_pre_flight_platform_non_kubernetes_recommendation(
        self, mock_compat, mock_cluster, mock_runtimes
    ):
        mock_compat.return_value = {
            "success": True,
            "data": {"compatible": True, "blockers": [], "platform": "openshift"},
        }
        mock_cluster.return_value = {"success": True, "data": {"gpu_total": 4}}
        mock_runtimes.return_value = {
            "success": True,
            "data": {"runtimes": [{"name": "torchtune"}]},
        }

        res = pre_flight()
        data = res["data"]
        assert any("openshift detected" in step for step in data["next_steps"])
        assert any("platform-fixes" in rec for rec in data["recommendations"])

    @patch("kubeflow_mcp.trainer.api.discovery.list_runtimes")
    @patch("kubeflow_mcp.trainer.api.planning.get_cluster_resources")
    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    def test_pre_flight_zero_gpus_warning(self, mock_compat, mock_cluster, mock_runtimes):
        mock_compat.return_value = {
            "success": True,
            "data": {"compatible": True, "blockers": [], "platform": "kubernetes"},
        }
        mock_cluster.return_value = {"success": True, "data": {"gpu_total": 0}}
        mock_runtimes.return_value = {
            "success": True,
            "data": {"runtimes": [{"name": "torch-distributed"}]},
        }

        res = pre_flight()
        data = res["data"]
        assert "No GPUs detected — LLM fine-tuning requires GPUs" in data["next_steps"]

    @patch("kubeflow_mcp.trainer.api.discovery.list_runtimes")
    @patch("kubeflow_mcp.trainer.api.planning.estimate_resources")
    @patch("kubeflow_mcp.trainer.api.planning.get_cluster_resources")
    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    def test_pre_flight_large_model_qlora_recommendation(
        self, mock_compat, mock_cluster, mock_estimate, mock_runtimes
    ):
        mock_compat.return_value = {
            "success": True,
            "data": {"compatible": True, "blockers": [], "platform": "kubernetes"},
        }
        mock_cluster.return_value = {"success": True, "data": {"gpu_total": 8}}
        mock_estimate.return_value = {
            "success": True,
            "data": {"gpu_memory_required": "40GB"},
        }
        mock_runtimes.return_value = {
            "success": True,
            "data": {"runtimes": [{"name": "torchtune"}]},
        }

        res = pre_flight(model="meta-llama/Llama-3.1-70B")
        data = res["data"]
        assert any("consider QLoRA" in rec for rec in data["recommendations"])

    @patch("kubeflow_mcp.trainer.api.discovery.list_runtimes")
    @patch("kubeflow_mcp.trainer.api.planning.get_cluster_resources")
    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    def test_pre_flight_tool_selection_torchtune_with_gpus(
        self, mock_compat, mock_cluster, mock_runtimes
    ):
        mock_compat.return_value = {
            "success": True,
            "data": {"compatible": True, "blockers": [], "platform": "kubernetes"},
        }
        mock_cluster.return_value = {"success": True, "data": {"gpu_total": 4}}
        mock_runtimes.return_value = {
            "success": True,
            "data": {"runtimes": [{"name": "torchtune"}]},
        }

        res = pre_flight()
        data = res["data"]
        assert data["tool_selection"]["recommended"] == "fine_tune"

    @patch("kubeflow_mcp.trainer.api.discovery.list_runtimes")
    @patch("kubeflow_mcp.trainer.api.planning.get_cluster_resources")
    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    def test_pre_flight_tool_selection_torchtune_without_gpus(
        self, mock_compat, mock_cluster, mock_runtimes
    ):
        mock_compat.return_value = {
            "success": True,
            "data": {"compatible": True, "blockers": [], "platform": "kubernetes"},
        }
        mock_cluster.return_value = {"success": True, "data": {"gpu_total": 0}}
        mock_runtimes.return_value = {
            "success": True,
            "data": {"runtimes": [{"name": "torchtune"}]},
        }

        res = pre_flight()
        data = res["data"]
        assert data["tool_selection"]["recommended"] == "run_custom_training"
        assert data["tool_selection"]["fallback_from"] == "fine_tune"

    @patch("kubeflow_mcp.trainer.api.discovery.list_runtimes")
    @patch("kubeflow_mcp.trainer.api.planning.get_cluster_resources")
    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    def test_pre_flight_tool_selection_non_torchtune_with_gpus(
        self, mock_compat, mock_cluster, mock_runtimes
    ):
        mock_compat.return_value = {
            "success": True,
            "data": {"compatible": True, "blockers": [], "platform": "kubernetes"},
        }
        mock_cluster.return_value = {"success": True, "data": {"gpu_total": 2}}
        mock_runtimes.return_value = {
            "success": True,
            "data": {"runtimes": [{"name": "torch-distributed"}]},
        }

        res = pre_flight()
        data = res["data"]
        assert data["tool_selection"]["recommended"] == "run_custom_training"
        assert "No torchtune runtime" in data["tool_selection"]["reason"]

    @patch("kubeflow_mcp.trainer.api.discovery.list_runtimes")
    @patch("kubeflow_mcp.trainer.api.planning.get_cluster_resources")
    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    def test_pre_flight_tool_selection_cpu_only(self, mock_compat, mock_cluster, mock_runtimes):
        mock_compat.return_value = {
            "success": True,
            "data": {"compatible": True, "blockers": [], "platform": "kubernetes"},
        }
        mock_cluster.return_value = {"success": True, "data": {"gpu_total": 0}}
        mock_runtimes.return_value = {
            "success": True,
            "data": {"runtimes": [{"name": "torch-distributed"}]},
        }

        res = pre_flight()
        data = res["data"]
        assert data["tool_selection"]["recommended"] == "run_custom_training"
        assert "CPU-only" in data["tool_selection"]["reason"]

    @patch("kubeflow_mcp.trainer.api.discovery.list_runtimes")
    @patch("kubeflow_mcp.trainer.api.planning.get_cluster_resources")
    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    def test_pre_flight_tool_selection_no_runtimes(self, mock_compat, mock_cluster, mock_runtimes):
        mock_compat.return_value = {
            "success": True,
            "data": {"compatible": True, "blockers": [], "platform": "kubernetes"},
        }
        mock_cluster.return_value = {"success": True, "data": {"gpu_total": 4}}
        mock_runtimes.return_value = {
            "success": True,
            "data": {"runtimes": []},
        }

        res = pre_flight()
        data = res["data"]
        assert data["tool_selection"]["recommended"] is None
        assert "No runtimes installed" in data["tool_selection"]["reason"]

    @pytest.mark.parametrize("gpu_total", [4, 0])
    @patch("kubeflow_mcp.trainer.api.discovery.get_trainer_client")
    @patch("kubeflow_mcp.trainer.api.planning.get_cluster_resources")
    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    def test_pre_flight_reports_unknown_runtimes_when_runtime_list_fails(
        self, mock_compat, mock_cluster, mock_client, gpu_total
    ):
        # A failed runtime list (e.g. an RBAC 403) says nothing about which runtimes
        # exist, so pre_flight must not tell the caller to install one.
        mock_compat.return_value = {
            "success": True,
            "data": {"compatible": True, "blockers": [], "platform": "kubernetes"},
        }
        mock_cluster.return_value = {"success": True, "data": {"gpu_total": gpu_total}}
        mock_client.return_value.list_runtimes.side_effect = RuntimeError(
            "403 Forbidden: cannot list clustertrainingruntimes"
        )

        res = pre_flight()
        data = res["data"]
        assert data["runtimes"]["success"] is False
        assert "403 Forbidden" in data["runtimes"]["error"]
        assert data["tool_selection"]["recommended"] is None
        assert data["tool_selection"]["reason"] == (
            "Runtimes unknown — the runtime list could not be read"
        )
        assert not any("No runtimes installed" in step for step in data["next_steps"])
        assert (
            "Could not list runtimes (see runtimes.error) — installed runtimes are unknown"
            in data["next_steps"]
        )

    @patch("kubeflow_mcp.trainer.api.discovery.list_runtimes")
    @patch("kubeflow_mcp.trainer.api.planning.estimate_resources")
    @patch("kubeflow_mcp.trainer.api.planning.get_cluster_resources")
    @patch("kubeflow_mcp.trainer.api.planning.check_compatibility")
    def test_pre_flight_unparseable_gpu_memory_fallback(
        self, mock_compat, mock_cluster, mock_estimate, mock_runtimes
    ):
        mock_compat.return_value = {
            "success": True,
            "data": {"compatible": True, "blockers": [], "platform": "kubernetes"},
        }
        mock_cluster.return_value = {"success": True, "data": {"gpu_total": 8}}
        mock_estimate.return_value = {
            "success": True,
            "data": {"gpu_memory_required": "unknownGB"},
        }
        mock_runtimes.return_value = {
            "success": True,
            "data": {"runtimes": [{"name": "torchtune"}]},
        }

        res = pre_flight(model="meta-llama/Llama-3.1-70B")
        data = res["data"]
        assert not any("consider QLoRA" in rec for rec in data["recommendations"])


@pytest.mark.parametrize("tool", [get_cluster_resources, check_compatibility])
@pytest.mark.parametrize("status", [401, 403, 500])
@pytest.mark.parametrize("wrapped", [False, True])
def test_planning_classifies_kubernetes_errors(tool: Any, status: int, wrapped: bool) -> None:
    from kubernetes.client.exceptions import ApiException

    error: Exception = ApiException(status=status)
    if wrapped:
        cause = error
        error = RuntimeError("Kubernetes request failed")
        error.__cause__ = cause
    target = (
        "kubeflow_mcp.common.utils.get_core_v1_api"
        if tool is get_cluster_resources
        else "kubeflow_mcp.trainer.api.planning._check_k8s_version"
    )
    with patch(target, side_effect=error):
        result = tool()
    assert result["error_code"] == (
        "PERMISSION_DENIED" if status in (401, 403) else "KUBERNETES_ERROR"
    )


@pytest.mark.parametrize("check", ["version", "crd"])
@pytest.mark.parametrize("status", [401, 403, 404, 500])
@pytest.mark.parametrize("wrapping", ["direct", "cause", "context", "nested"])
def test_compatibility_classifies_required_api_failures(
    mock_k8s_apis: dict[str, MagicMock], check: str, status: int, wrapping: str
) -> None:
    from kubernetes.client.exceptions import ApiException

    error: Exception = ApiException(status=status)
    if wrapping != "direct":
        cause = error
        error = RuntimeError("Kubernetes request failed")
        if wrapping == "context":
            error.__context__ = cause
        else:
            error.__cause__ = cause
        if wrapping == "nested":
            cause = error
            error = RuntimeError("Compatibility check failed")
            error.__cause__ = cause
    version_api = MagicMock()
    version_api.get_code.return_value = SimpleNamespace(git_version="v1.34.2")
    crd_api = MagicMock()
    crd_api.read_custom_resource_definition.return_value = SimpleNamespace(
        spec=SimpleNamespace(versions=[SimpleNamespace(name=MIN_TRAINER_CRD_VERSION, served=True)])
    )
    operation = (
        version_api.get_code if check == "version" else crd_api.read_custom_resource_definition
    )
    operation.side_effect = error
    with (
        patch("kubeflow_mcp.common.utils.get_version_api", return_value=version_api),
        patch("kubernetes.client.ApiextensionsV1Api", return_value=crd_api),
    ):
        result = check_compatibility()
    if status in (401, 403):
        assert result["success"] is False
        assert result["error_code"] == "PERMISSION_DENIED"
    else:
        assert result["success"] is True
        assert result["data"]["compatible"] is False
        key = "kubernetes_version" if check == "version" else "trainer_crd"
        assert result["data"]["checks"][key]["status"] == "fail"


@pytest.mark.parametrize("step", range(4), ids=["compatibility", "cluster", "estimate", "runtimes"])
@pytest.mark.parametrize("code", ["PERMISSION_DENIED", "SDK_ERROR", "VALIDATION_ERROR"])
def test_pre_flight_propagates_failed_checks(step: int, code: str) -> None:
    from kubeflow_mcp.core.server import _inject_meta

    error = {"success": False, "error": "Check failed", "error_code": code}
    with (
        patch(
            "kubeflow_mcp.trainer.api.planning.check_compatibility",
            return_value={"success": True, "data": {"compatible": True, "blockers": []}},
        ) as compat,
        patch(
            "kubeflow_mcp.trainer.api.planning.get_cluster_resources",
            return_value={"success": True, "data": {"gpu_total": 1}},
        ) as cluster,
        patch(
            "kubeflow_mcp.trainer.api.planning.estimate_resources",
            return_value={"success": True, "data": {}},
        ) as estimate,
        patch(
            "kubeflow_mcp.trainer.api.discovery.list_runtimes",
            return_value={"success": True, "data": {"runtimes": []}},
        ) as runtimes,
    ):
        checks = [compat, cluster, estimate, runtimes]
        checks[step].return_value = error
        result = pre_flight(model="test/model")
        for later in checks[step + 1 :]:
            later.assert_not_called()
    assert result == error
    assert "_meta" not in _inject_meta(result, "pre_flight")
