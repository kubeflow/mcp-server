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

"""Planning tools for resource estimation and cluster inspection."""

import logging
import re
from typing import Any

from kubeflow_mcp.common.constants import (
    MIN_K8S_VERSION,
    MIN_TRAINER_CRD_VERSION,
    TRAINER_CRD_NAME,
    ErrorCode,
)
from kubeflow_mcp.common.types import ToolError, ToolResponse, exception_details

logger = logging.getLogger(__name__)

_HF_MODEL_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")


def _suggest_hf_model_ids(model: str, limit: int = 3) -> list[str]:
    """Suggest real HuggingFace model IDs for a malformed model reference.

    Normalizes common non-Hub formats into a search term (drops an ``hf://``
    prefix and any Ollama-style ``:tag`` suffix such as ``qwen3:8b``) and asks
    the Hub for close matches. Returns an empty list when the lookup errors or
    finds nothing, so the suggestions stay best-effort and the validation path
    never depends on network access.
    """
    search = model.strip().removeprefix("hf://").split(":", 1)[0].strip()
    if not search:
        return []
    try:
        from huggingface_hub import list_models

        return [m.id for m in list_models(search=search, limit=limit, full=False)]
    except Exception:  # noqa: BLE001 - suggestions are best-effort, never fatal
        return []


def _get_model_info_from_hf(model: str) -> dict[str, Any] | None:
    """Fetch model info from HuggingFace Hub."""
    try:
        if not _HF_MODEL_ID_RE.match(model):
            result: dict[str, Any] = {"error": f"Invalid HuggingFace model ID format: '{model}'"}
            suggestions = _suggest_hf_model_ids(model)
            if suggestions:
                result["suggestions"] = suggestions
            return result

        from huggingface_hub import model_info

        info = model_info(model, timeout=10)

        # Get parameter count from safetensors metadata
        params = None
        if info.safetensors:
            params = info.safetensors.total

        # Try card_data for parameter count
        if not params and info.card_data:
            params = getattr(info.card_data, "num_parameters", None)

        return {
            "model_id": info.id,
            "params": params,
            "library": getattr(info, "library_name", None),
            "pipeline": getattr(info, "pipeline_tag", None),
        }

    except Exception as e:
        return {"error": str(e)}


QUANTIZATION_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}
OVERHEAD_GB = 2


def _estimate_from_params(
    params: float, batch_size: int = 4, quantization: str = "bf16"
) -> dict[str, Any]:
    """Estimate resources based on parameter count.

    Rules of thumb for LoRA fine-tuning:
    - GPU memory = weights + LoRA adapters + activations + overhead
    - Weight bytes depend on quantization (bf16=2, int8=1, int4=0.5)
    - Full fine-tuning needs 4-6x more (gradients + optimizer states)
    """
    params_b = params / 1e9
    bpw = QUANTIZATION_BYTES.get(quantization, 2.0)

    weight_gb = round(params_b * bpw, 2)
    # LoRA adapters are ~2-5% of weights; use 5% as safe upper bound
    adapter_gb = round(weight_gb * 0.05, 2)
    activation_gb = round(weight_gb * 0.3 * (1 + (batch_size - 1) * 0.1), 2)

    gpu_memory_gb = int(weight_gb + adapter_gb + activation_gb + OVERHEAD_GB)
    gpu_memory_gb = max(gpu_memory_gb, 1)

    if gpu_memory_gb <= 8:
        gpu_count = 1
        gpu_type = "8GB (RTX 3070/4070)"
    elif gpu_memory_gb <= 16:
        gpu_count = 1
        gpu_type = "16GB (T4/RTX 4080)"
    elif gpu_memory_gb <= 24:
        gpu_count = 1
        gpu_type = "24GB (A10/RTX 3090)"
    elif gpu_memory_gb <= 40:
        gpu_count = 1
        gpu_type = "40GB (A100-40GB)"
    elif gpu_memory_gb <= 80:
        gpu_count = 1
        gpu_type = "80GB (A100-80GB/H100)"
    else:
        gpu_count = max(2, (gpu_memory_gb + 79) // 80)
        gpu_type = f"{gpu_count}x 80GB GPUs"

    system_memory = max(16, gpu_memory_gb * 2)

    return {
        "gpu_count": gpu_count,
        "gpu_memory_gb": gpu_memory_gb,
        "gpu_type": gpu_type,
        "system_memory_gi": system_memory,
        "params_billions": round(params_b, 2),
        "quantization": quantization,
        "breakdown": {
            "weights_gb": weight_gb,
            "lora_adapters_gb": adapter_gb,
            "activations_gb": activation_gb,
            "overhead_gb": OVERHEAD_GB,
        },
    }


def _parse_k8s_version(git_version: str) -> tuple[int, int] | None:
    """Extract (major, minor) from a Kubernetes git version string like 'v1.29.3'."""
    m = re.match(r"v?(\d+)\.(\d+)", git_version)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _check_k8s_version(
    checks: dict[str, dict[str, Any]],
    blockers: list[str],
    get_version_api: Any,
    timeout: int,
) -> None:
    """Check Kubernetes server version meets minimum requirement."""
    try:
        version_api = get_version_api()
        version_info = version_api.get_code(_request_timeout=timeout)
        k8s_version_str = version_info.git_version
        parsed = _parse_k8s_version(k8s_version_str)
        min_label = f"v{MIN_K8S_VERSION[0]}.{MIN_K8S_VERSION[1]}"
        if parsed and parsed >= MIN_K8S_VERSION:
            checks["kubernetes_version"] = {
                "status": "pass",
                "version": k8s_version_str,
                "minimum": min_label,
            }
        elif parsed:
            checks["kubernetes_version"] = {
                "status": "fail",
                "version": k8s_version_str,
                "minimum": min_label,
            }
            blockers.append(f"Kubernetes {k8s_version_str} is below minimum {min_label}")
        else:
            checks["kubernetes_version"] = {
                "status": "warn",
                "version": k8s_version_str,
                "detail": "Could not parse version",
            }
    except Exception as e:
        checks["kubernetes_version"] = {"status": "fail", "error": str(e)}
        blockers.append(f"Cannot reach Kubernetes API: {e}")


def _check_trainer_crd(
    checks: dict[str, dict[str, Any]],
    blockers: list[str],
    get_custom_objects_api: Any,
    timeout: int,
) -> bool:
    """Check Trainer CRD existence and API version. Returns whether CRD is installed."""
    try:
        api_ext = get_custom_objects_api()
        from kubernetes import client as k8s_client

        api_ext_v1 = k8s_client.ApiextensionsV1Api(api_ext.api_client)
        crd = api_ext_v1.read_custom_resource_definition(TRAINER_CRD_NAME, _request_timeout=timeout)
        served_versions = [v.name for v in crd.spec.versions if v.served]

        checks["trainer_crd"] = {
            "status": "pass",
            "crd_name": TRAINER_CRD_NAME,
            "served_versions": served_versions,
        }
        if MIN_TRAINER_CRD_VERSION in served_versions:
            checks["trainer_api_version"] = {
                "status": "pass",
                "required": MIN_TRAINER_CRD_VERSION,
                "served": served_versions,
            }
        else:
            checks["trainer_api_version"] = {
                "status": "fail",
                "required": MIN_TRAINER_CRD_VERSION,
                "served": served_versions,
            }
            blockers.append(
                f"Trainer CRD does not serve {MIN_TRAINER_CRD_VERSION} (served: {', '.join(served_versions)})"
            )
        return True
    except Exception as e:
        checks["trainer_crd"] = {"status": "fail", "error": str(e)}
        checks["trainer_api_version"] = {"status": "fail", "error": "CRD not found"}
        blockers.append(
            "Kubeflow Trainer CRD not installed. "
            "Install the Trainer operator first: "
            "https://www.kubeflow.org/docs/components/trainer/overview/"
        )
        return False


def _check_sdk_version(checks: dict[str, dict[str, Any]], blockers: list[str]) -> None:
    """Check kubeflow-trainer package is installed and meets minimum version."""
    import importlib.metadata

    try:
        sdk_version = importlib.metadata.version("kubeflow-trainer")
        from packaging.version import Version

        sdk_min = "0.4.0"
        sdk_ok = Version(sdk_version) >= Version(sdk_min)
        checks["kubeflow_sdk"] = {
            "status": "pass" if sdk_ok else "fail",
            "version": sdk_version,
            "minimum": sdk_min,
        }
        if not sdk_ok:
            blockers.append(f"kubeflow-trainer {sdk_version} is below minimum {sdk_min}")
    except importlib.metadata.PackageNotFoundError:
        try:
            sdk_version = importlib.metadata.version("kubeflow")
            checks["kubeflow_sdk"] = {
                "status": "pass",
                "version": sdk_version,
                "package": "kubeflow",
            }
        except importlib.metadata.PackageNotFoundError:
            checks["kubeflow_sdk"] = {"status": "fail", "error": "Not installed"}
            blockers.append("kubeflow-trainer not installed (pip install kubeflow-trainer)")
    except ImportError:
        checks["kubeflow_sdk"] = {
            "status": "warn",
            "detail": "packaging library not available, version comparison skipped",
        }


def _detect_platform(
    checks: dict[str, dict[str, Any]],
    recommendations: list[str],
    get_core_v1_api: Any,
    timeout: int,
) -> str:
    """Detect cluster platform from node labels."""
    _label_prefixes = [
        ("node.openshift.io", "openshift"),
        ("eks.amazonaws.com", "eks"),
        ("cloud.google.com/gke", "gke"),
    ]
    try:
        v1 = get_core_v1_api()
        nodes = v1.list_node(limit=3, _request_timeout=timeout)
        for node in nodes.items:
            labels = node.metadata.labels or {}
            for prefix, name in _label_prefixes:
                if any(k.startswith(prefix) for k in labels):
                    checks["platform"] = {"status": "info", "detected": name}
                    recommendations.append(
                        f"{name} detected — read trainer://guides/platform-fixes for platform-specific guidance"
                    )
                    return name
        checks["platform"] = {"status": "info", "detected": "kubernetes"}
        return "kubernetes"
    except Exception as e:
        checks["platform"] = {"status": "warn", "error": str(e)}
        logger.warning(f"Platform detection failed: {e}")
        return "unknown"


def check_compatibility() -> dict[str, Any]:
    """Pre-flight check: verify K8s version, Trainer CRD, installed packages, and platform.

    Run this FIRST before any other tool to confirm the MCP server is
    connected to a compatible environment. Returns a structured report
    with pass/fail for each check and a list of blockers.

    Checks performed:
        1. **Kubernetes version** — minimum {min_k8s} required
        2. **Trainer CRD installed** — ``trainjobs.trainer.kubeflow.org`` must exist
        3. **CRD API version** — ``{crd_version}`` must be served
        4. **Kubeflow training package** — minimum ``0.4.0`` required
        5. **Platform detection** — identifies platform from node labels

    Returns:
        dict: Response containing:

        - ``compatible`` (bool): True if all checks pass
        - ``checks`` (dict): Per-check results with status and details
        - ``blockers`` (list[str]): Human-readable list of blocking issues
        - ``platform`` (str): Detected platform (``openshift``, ``eks``, ``gke``, ``kubernetes``)
        - ``recommendations`` (list[str]): Suggested actions

    Example:
        >>> check_compatibility()
        {{"data": {{"compatible": true, "checks": {{...}}, "platform": "kubernetes"}}}}
    """.format(
        min_k8s=f"{MIN_K8S_VERSION[0]}.{MIN_K8S_VERSION[1]}", crd_version=MIN_TRAINER_CRD_VERSION
    )
    try:
        from kubeflow_mcp.common.utils import (
            K8S_TIMEOUT,
            get_core_v1_api,
            get_custom_objects_api,
            get_version_api,
        )

        blockers: list[str] = []
        recommendations: list[str] = []
        checks: dict[str, dict[str, Any]] = {}

        _check_k8s_version(checks, blockers, get_version_api, K8S_TIMEOUT)
        crd_installed = _check_trainer_crd(checks, blockers, get_custom_objects_api, K8S_TIMEOUT)
        _check_sdk_version(checks, blockers)
        platform = _detect_platform(checks, recommendations, get_core_v1_api, K8S_TIMEOUT)

        if not blockers and crd_installed:
            recommendations.append("Environment compatible — proceed with get_cluster_resources()")

        return ToolResponse(
            data={
                "compatible": len(blockers) == 0,
                "checks": checks,
                "blockers": blockers,
                "platform": platform,
                "recommendations": recommendations,
            }
        ).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.KUBERNETES_ERROR,
            details=exception_details(e),
        ).model_dump()


def get_cluster_resources() -> dict[str, Any]:
    """Check cluster GPU and compute availability.

    Call this before submitting training jobs to verify resources exist.

    Returns:
        dict: Response containing:

        - ``gpu_total`` (int): Total GPUs across all nodes
        - ``nodes_with_gpu`` (int): Number of nodes with GPUs
        - ``node_count`` (int): Total node count
        - ``nodes`` (list): Per-node details (name, memory, cpu, gpus)

    Example:
        >>> get_cluster_resources()
        {"data": {"gpu_total": 4, "nodes_with_gpu": 2, ...}}

    Note:
        Returns ``gpu_total=0`` if no GPUs available - LLM fine-tuning requires GPUs.
    """
    try:
        from kubeflow_mcp.common.utils import K8S_TIMEOUT, get_core_v1_api

        v1 = get_core_v1_api()
        nodes = v1.list_node(_request_timeout=K8S_TIMEOUT)
        gpu_total = 0
        node_info = []

        for node in nodes.items:
            alloc = node.status.allocatable or {}
            gpu = int(alloc.get("nvidia.com/gpu", 0))
            gpu_total += gpu

            node_data = {
                "name": node.metadata.name,
                "memory": alloc.get("memory"),
                "cpu": alloc.get("cpu"),
            }

            if gpu > 0:
                node_data["gpus"] = gpu

            node_info.append(node_data)

        return ToolResponse(
            data={
                "gpu_total": gpu_total,
                "nodes_with_gpu": sum(1 for n in node_info if n.get("gpus", 0) > 0),
                "node_count": len(node_info),
                "nodes": node_info,
            }
        ).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.KUBERNETES_ERROR,
            details=exception_details(e),
        ).model_dump()


def estimate_resources(
    model: str,
    num_workers: int = 1,
    batch_size: int = 4,
    quantization: str = "bf16",
) -> dict[str, Any]:
    """Estimate GPU and memory requirements for training a model.

    Fetches model metadata from HuggingFace Hub and calculates resource
    requirements based on parameter count for LoRA fine-tuning.

    Args:
        model: HuggingFace model ID. Accepts ``google/gemma-2b`` or ``hf://google/gemma-2b``.
        num_workers: Number of distributed workers. Defaults to 1.
        batch_size: Per-GPU batch size. Defaults to 4.
        quantization: Weight precision — ``bf16``, ``fp16``, ``int8``, ``int4``, ``fp32``.
            Defaults to ``bf16``.

    Returns:
        dict: Response containing:

        - ``params_billions`` (float): Model size in billions of parameters
        - ``gpu_per_worker`` (int): GPUs needed per worker
        - ``gpu_memory_required`` (str): Estimated GPU memory (e.g., "16GB")
        - ``gpu_type_recommended`` (str): Suggested GPU type
        - ``total_gpu`` (int): Total GPUs needed (gpu_per_worker * num_workers)
        - ``quantization`` (str): Precision used for estimation
        - ``breakdown`` (dict): Memory breakdown (weights, adapters, activations, overhead)
        - ``recommendation`` (str): Human-readable suggestion

    Example:
        >>> estimate_resources("google/gemma-2b", batch_size=2, quantization="int8")
        {"data": {"params_billions": 2.0, "gpu_memory_required": "8GB", ...}}
    """
    try:
        if num_workers < 1 or num_workers > 100:
            return ToolError(
                error=f"num_workers must be between 1 and 100, got {num_workers}",
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()
        if batch_size < 1 or batch_size > 1024:
            return ToolError(
                error=f"batch_size must be between 1 and 1024, got {batch_size}",
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()
        valid_quantizations = list(QUANTIZATION_BYTES.keys())
        if quantization not in QUANTIZATION_BYTES:
            return ToolError(
                error=f"Invalid quantization '{quantization}'. Must be one of: {valid_quantizations}",
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()

        # Strip hf:// prefix if present (fine_tune uses hf://, but HF API needs raw ID)
        model_id = model.removeprefix("hf://")

        # Fetch model info from HuggingFace
        hf_info = _get_model_info_from_hf(model_id)

        if not hf_info or "error" in hf_info:
            error_msg = hf_info.get("error", "Unknown error") if hf_info else "API failed"
            details: dict[str, Any] = {
                "hint": "Ensure model path is correct (e.g., 'meta-llama/Llama-3.2-1B')"
            }
            # Surface any "did you mean" suggestions from the format check so the
            # caller (and pre_flight, which delegates here) can self-correct.
            if hf_info and hf_info.get("suggestions"):
                details["suggestions"] = hf_info["suggestions"]
            return ToolError(
                error=f"Could not fetch model info from HuggingFace: {error_msg}",
                error_code=ErrorCode.SDK_ERROR,
                details=details,
            ).model_dump()

        params = hf_info.get("params")
        if not params:
            return ToolError(
                error="Model found but parameter count not available in HuggingFace metadata",
                error_code=ErrorCode.SDK_ERROR,
                details={
                    "model_id": hf_info.get("model_id"),
                    "hint": "Try a different model or check HuggingFace model card",
                },
            ).model_dump()

        estimates = _estimate_from_params(params, batch_size, quantization)

        gpu_per_worker = estimates["gpu_count"]
        total_gpu = gpu_per_worker * num_workers

        return ToolResponse(
            data={
                "model": model,
                "params_billions": estimates["params_billions"],
                "gpu_per_worker": gpu_per_worker,
                "gpu_memory_required": f"{estimates['gpu_memory_gb']}GB",
                "gpu_type_recommended": estimates["gpu_type"],
                "memory_per_worker": f"{estimates['system_memory_gi']}Gi",
                "total_gpu": total_gpu,
                "num_workers": num_workers,
                "batch_size": batch_size,
                "quantization": estimates["quantization"],
                "breakdown": estimates["breakdown"],
                "training_type": f"LoRA ({quantization})",
                "recommendation": f"Request {total_gpu} GPU(s) - {estimates['gpu_type']}",
            }
        ).model_dump()

    except Exception as e:
        return ToolError(
            error=str(e),
            error_code=ErrorCode.SDK_ERROR,
            details=exception_details(e),
        ).model_dump()


def pre_flight(
    model: str = "",
    num_workers: int = 1,
    batch_size: int = 4,
    quantization: str = "bf16",
) -> dict[str, Any]:
    """One-shot environment check: compatibility + cluster + estimate + runtimes.

    Combines four planning/discovery calls into a single round-trip:
    ``check_compatibility()``, ``get_cluster_resources()``,
    ``estimate_resources()`` (when *model* is provided), and
    ``list_runtimes()``.

    Args:
        model: Optional HuggingFace model ID (e.g., ``meta-llama/Llama-3.2-1B``).
            When provided, includes GPU memory estimates in the response.
            Accepts ``hf://`` prefix.
        num_workers: Number of distributed workers for estimation. Defaults to 1.
        batch_size: Per-GPU batch size for estimation. Defaults to 4.
        quantization: Weight precision for estimation — ``bf16``, ``fp16``,
            ``int8``, ``int4``, ``fp32``. Defaults to ``bf16``.

    Returns:
        dict: Combined response with sections:

        - ``compatibility`` — from ``check_compatibility()``
        - ``cluster`` — from ``get_cluster_resources()``
        - ``estimate`` — from ``estimate_resources()`` (only when *model* given)
        - ``runtimes`` — from ``list_runtimes()``
        - ``next_steps`` — context-aware recommendations for the agent

    Example:
        >>> pre_flight(model="meta-llama/Llama-3.2-1B")
        {{"data": {{"compatibility": {{...}}, "cluster": {{...}}, "estimate": {{...}}, ...}}}}
    """
    result: dict[str, Any] = {}
    next_steps: list[str] = []

    compat = check_compatibility()
    compat_data = compat.get("data", compat)
    result["compatibility"] = compat_data

    if compat_data.get("blockers"):
        return ToolResponse(
            data={
                **result,
                "next_steps": [f"Blocker: {b}" for b in compat_data["blockers"]],
            }
        ).model_dump()

    platform = compat_data.get("platform", "kubernetes")
    if platform != "kubernetes":
        next_steps.append(
            f"{platform} detected — read trainer://guides/platform-fixes "
            "for platform-specific guidance"
        )

    cluster = get_cluster_resources()
    cluster_data = cluster.get("data", cluster)
    result["cluster"] = cluster_data

    gpu_total = cluster_data.get("gpu_total", 0)
    if gpu_total == 0:
        next_steps.append("No GPUs detected — LLM fine-tuning requires GPUs")

    if model:
        estimate = estimate_resources(
            model=model,
            num_workers=num_workers,
            batch_size=batch_size,
            quantization=quantization,
        )
        estimate_data = estimate.get("data", estimate)
        result["estimate"] = estimate_data

    from kubeflow_mcp.trainer.api.discovery import list_runtimes as _list_runtimes

    runtimes_resp = _list_runtimes()
    runtimes_data = runtimes_resp.get("data", runtimes_resp)
    result["runtimes"] = runtimes_data

    runtime_names = [r.get("name", "") for r in runtimes_data.get("runtimes", [])]
    has_torchtune = any(
        n.startswith("torchtune") or n.startswith("torch-tune") for n in runtime_names
    )

    tool_selection: dict[str, Any] = {}
    recommendations: list[str] = []

    if has_torchtune and gpu_total > 0:
        tool_selection["recommended"] = "fine_tune"
        tool_selection["reason"] = "torchtune runtime + GPUs available"
        next_steps.append("torchtune runtime available — fine_tune() is supported")
    elif has_torchtune and gpu_total == 0:
        tool_selection["recommended"] = "run_custom_training"
        tool_selection["reason"] = "torchtune needs GPUs (NCCL), cluster has none — use gloo"
        tool_selection["fallback_from"] = "fine_tune"
        next_steps.append(
            "torchtune runtime found but NO GPUs — fine_tune() will fail. "
            "Use run_custom_training() with gloo backend instead"
        )
        recommendations.append("Read trainer://guides/training-patterns for LoRA code patterns")
    elif runtime_names and gpu_total > 0:
        tool_selection["recommended"] = "run_custom_training"
        tool_selection["reason"] = "No torchtune runtime — write LoRA script with gloo/nccl"
        next_steps.append("No torchtune runtime — use run_custom_training() with a LoRA script")
        recommendations.append("Read trainer://guides/training-patterns for LoRA code patterns")
    elif runtime_names:
        tool_selection["recommended"] = "run_custom_training"
        tool_selection["reason"] = "CPU-only cluster, no torchtune runtime"
        next_steps.append(
            "CPU-only cluster, no torchtune — use run_custom_training() with gloo backend"
        )
        recommendations.append("Read trainer://guides/training-patterns for distributed patterns")
    else:
        tool_selection["recommended"] = None
        tool_selection["reason"] = "No runtimes installed"
        next_steps.append("No runtimes installed — install a ClusterTrainingRuntime first")

    if platform != "kubernetes":
        recommendations.append(
            "Read trainer://guides/platform-fixes for platform-specific guidance"
        )

    if model and "estimate" in result:
        est = result["estimate"]
        mem_gb = est.get("gpu_memory_required", "")
        if isinstance(mem_gb, str) and mem_gb.endswith("GB"):
            try:
                mem_val = int(mem_gb.rstrip("GB"))
            except ValueError:
                mem_val = 0
            if mem_val > 24:
                recommendations.append(
                    f"Model needs ~{mem_val}GB GPU — consider QLoRA (quantize_base=True) "
                    "to halve memory"
                )

    result["tool_selection"] = tool_selection
    result["recommendations"] = recommendations
    result["next_steps"] = next_steps
    return ToolResponse(data=result).model_dump()
