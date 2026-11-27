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

"""Resource quota pre-flight verification for training jobs (ROADMAP Phase 4).

Validates requested compute and pod counts against Kubernetes ResourceQuota
objects in the target namespace before jobs are admitted or submitted.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from kubernetes.client.rest import ApiException
from kubernetes.utils.quantity import parse_quantity

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError
from kubeflow_mcp.common.utils import K8S_TIMEOUT, get_core_v1_api, get_trainer_effective_namespace

logger = logging.getLogger(__name__)

# Standard resource aliases mapped to ResourceQuota keys
_CPU_KEYS = ("requests.cpu", "cpu", "limits.cpu")
_MEMORY_KEYS = ("requests.memory", "memory", "limits.memory")
_POD_KEYS = ("count/pods", "pods")
_GPU_KEYS = (
    "requests.nvidia.com/gpu",
    "nvidia.com/gpu",
    "requests.amd.com/gpu",
    "amd.com/gpu",
    "requests.intel.com/gpu",
    "intel.com/gpu",
    "requests.gpu.intel.com/i915",
    "gpu.intel.com/i915",
)


def _to_decimal(val: Any) -> Decimal:
    """Safely parse a Kubernetes quantity string or number into a Decimal."""
    if isinstance(val, (int, float, Decimal)):
        return Decimal(str(val))
    return parse_quantity(str(val).strip())


def _format_quantity(val: Decimal) -> str:
    """Format decimal quantity cleanly for error messages."""
    if val == val.to_integral_value():
        return str(int(val))
    return str(val.normalize())


def _extract_quota_items(quota_list: Any) -> list[Any]:
    """Safely extract list of quota items handling both K8s models and dictionaries."""
    if isinstance(quota_list, dict):
        items = quota_list.get("items", [])
    else:
        items = getattr(quota_list, "items", [])
    if callable(items) or not isinstance(items, (list, tuple)):
        return []
    return list(items)


def _extract_quota_data(q: Any) -> tuple[str, dict[str, Any], dict[str, Any], list[str]]:
    """Safely extract quota name, hard, used dicts, and scopes from a model or dict."""
    if isinstance(q, dict):
        q_name = q.get("metadata", {}).get("name", "quota")
        status = q.get("status", {}) or {}
        hard = status.get("hard", {}) or {}
        used = status.get("used", {}) or {}
        scopes = q.get("spec", {}).get("scopes", []) or []
        return q_name, hard, used, list(scopes)
    q_name = getattr(q.metadata, "name", "quota") if hasattr(q, "metadata") else "quota"
    status = getattr(q, "status", None)
    hard = getattr(status, "hard", {}) or {} if status else {}
    used = getattr(status, "used", {}) or {} if status else {}
    spec = getattr(q, "spec", None)
    scopes = getattr(spec, "scopes", []) or [] if spec else []
    return q_name, hard, used, list(scopes)


def get_namespace_resource_quotas(
    namespace: str | None = None,
    *,
    core_v1_api: Any = None,
) -> list[dict[str, Any]]:
    """List and parse active ResourceQuota objects in the target namespace.

    Returns an empty list if no quotas exist or if the caller lacks permission
    to list ResourceQuotas.
    """
    try:
        effective_ns = get_trainer_effective_namespace(namespace)
        api = core_v1_api or get_core_v1_api()
        quota_list = api.list_namespaced_resource_quota(
            namespace=effective_ns,
            _request_timeout=K8S_TIMEOUT,
        )
    except ApiException as e:
        logger.debug(
            "ResourceQuota listing skipped for namespace '%s' (status %s): %s",
            namespace,
            e.status,
            e.reason,
        )
        return []
    except Exception as e:
        logger.debug(
            "Could not query ResourceQuota in namespace '%s': %s",
            namespace,
            e,
        )
        return []

    quotas_info: list[dict[str, Any]] = []
    items = _extract_quota_items(quota_list)
    for q in items:
        name, hard, used, _ = _extract_quota_data(q)

        quota_entry: dict[str, Any] = {
            "name": name,
            "namespace": effective_ns,
            "hard": {},
            "used": {},
            "available": {},
        }

        for res_key, hard_val_str in hard.items():
            try:
                hard_dec = _to_decimal(hard_val_str)
                used_val_str = used.get(res_key, "0")
                used_dec = _to_decimal(used_val_str)
                avail_dec = max(Decimal(0), hard_dec - used_dec)

                quota_entry["hard"][res_key] = hard_val_str
                quota_entry["used"][res_key] = used_val_str
                quota_entry["available"][res_key] = _format_quantity(avail_dec)
            except Exception as parse_err:
                logger.debug(
                    "Failed to parse quantity for %s in quota %s: %s", res_key, name, parse_err
                )

        quotas_info.append(quota_entry)

    return quotas_info


def _extract_requested_resources(
    resources: dict[str, Any],
    nodes_dec: Decimal,
) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    """Extract and scale total CPU, memory, and GPU requirements."""
    req_cpu: Decimal | None = None
    for k in ("cpu", "CPU", "requests.cpu", "limits.cpu"):
        if k in resources and resources[k] is not None:
            req_cpu = _to_decimal(resources[k]) * nodes_dec
            break

    req_mem: Decimal | None = None
    for k in ("memory", "requests.memory", "limits.memory"):
        if k in resources and resources[k] is not None:
            req_mem = _to_decimal(resources[k]) * nodes_dec
            break

    req_gpu: Decimal | None = None
    for k in (
        "gpu",
        "nvidia.com/gpu",
        "requests.nvidia.com/gpu",
        "amd.com/gpu",
        "requests.amd.com/gpu",
        "intel.com/gpu",
        "requests.intel.com/gpu",
        "gpu.intel.com/i915",
        "requests.gpu.intel.com/i915",
    ):
        if k in resources and resources[k] is not None:
            req_gpu = _to_decimal(resources[k]) * nodes_dec
            break

    return req_cpu, req_mem, req_gpu


def _check_resource_limit(
    q_name: str,
    effective_ns: str,
    hard: dict[str, Any],
    used: dict[str, Any],
    keys: tuple[str, ...],
    requested: Decimal,
    unit_label: str,
    remediation_hint: str,
) -> ToolError | None:
    """Check a specific resource category against a ResourceQuota limit."""
    for k in keys:
        if k in hard:
            hard_val = _to_decimal(hard[k])
            used_val = _to_decimal(used.get(k, "0"))
            avail_val = max(Decimal(0), hard_val - used_val)
            if requested > avail_val:
                return ToolError(
                    error=(
                        f"Resource quota '{q_name}' exceeded in namespace '{effective_ns}': "
                        f"requested {requested} {unit_label}, but only {avail_val} available "
                        f"(hard: {hard_val}, used: {used_val})"
                    ),
                    error_code=ErrorCode.QUOTA_EXCEEDED,
                    details={
                        "quota_name": q_name,
                        "namespace": effective_ns,
                        "resource": k,
                        "requested": _format_quantity(requested),
                        "available": _format_quantity(avail_val),
                        "hard": _format_quantity(hard_val),
                        "used": _format_quantity(used_val),
                        "remediation": remediation_hint,
                    },
                )
    return None


def _check_single_quota(
    q_name: str,
    effective_ns: str,
    hard: dict[str, Any],
    used: dict[str, Any],
    nodes_dec: Decimal,
    req_cpu: Decimal | None,
    req_mem: Decimal | None,
    req_gpu: Decimal | None,
) -> ToolError | None:
    """Check a single ResourceQuota object for capacity violations."""
    # 1. Pod count
    pod_err = _check_resource_limit(
        q_name=q_name,
        effective_ns=effective_ns,
        hard=hard,
        used=used,
        keys=_POD_KEYS,
        requested=nodes_dec,
        unit_label="pod(s)",
        remediation_hint=(
            f"Reduce num_nodes (currently {nodes_dec}), delete finished pods, "
            f"or request a quota increase in namespace '{effective_ns}'."
        ),
    )
    if pod_err:
        return pod_err

    # 2. CPU requests
    if req_cpu is not None:
        cpu_err = _check_resource_limit(
            q_name=q_name,
            effective_ns=effective_ns,
            hard=hard,
            used=used,
            keys=_CPU_KEYS,
            requested=req_cpu,
            unit_label=f"CPU(s) across {nodes_dec} node(s)",
            remediation_hint=(
                f"Reduce requested cpu in resources_per_node or reduce num_nodes, "
                f"or request a quota increase in namespace '{effective_ns}'."
            ),
        )
        if cpu_err:
            return cpu_err

    # 3. Memory requests
    if req_mem is not None:
        mem_err = _check_resource_limit(
            q_name=q_name,
            effective_ns=effective_ns,
            hard=hard,
            used=used,
            keys=_MEMORY_KEYS,
            requested=req_mem,
            unit_label=f"bytes memory across {nodes_dec} node(s)",
            remediation_hint=(
                f"Reduce requested memory in resources_per_node or reduce num_nodes, "
                f"or request a quota increase in namespace '{effective_ns}'."
            ),
        )
        if mem_err:
            return mem_err

    # 4. GPU requests
    if req_gpu is not None:
        gpu_err = _check_resource_limit(
            q_name=q_name,
            effective_ns=effective_ns,
            hard=hard,
            used=used,
            keys=_GPU_KEYS,
            requested=req_gpu,
            unit_label=f"GPU(s) across {nodes_dec} node(s)",
            remediation_hint=(
                f"Reduce requested gpu count or reduce num_nodes, "
                f"or request a quota increase in namespace '{effective_ns}'."
            ),
        )
        if gpu_err:
            return gpu_err

    return None


def _get_namespace_limit_range_defaults(
    effective_ns: str,
    api: Any,
) -> tuple[Decimal | None, Decimal | None, bool]:
    """Retrieve default CPU and memory requests from namespace LimitRange resources.

    If a namespace has a LimitRange defining defaultRequest (or default limits) for
    CPU or memory, pods submitted without explicit compute requests are mutated at
    admission to receive those defaults, altering their effective QoS class away from BestEffort.

    Returns:
        tuple of (default_cpu, default_mem, is_forbidden)
    """
    try:
        lr_list = api.list_namespaced_limit_range(
            namespace=effective_ns,
            _request_timeout=K8S_TIMEOUT,
        )
        items = getattr(lr_list, "items", []) or []
        default_cpu: Decimal | None = None
        default_mem: Decimal | None = None
        for lr in items:
            spec = getattr(lr, "spec", None)
            limits = getattr(spec, "limits", []) or []
            for item in limits:
                default_req = (
                    getattr(item, "default_request", None)
                    or getattr(item, "defaultRequest", None)
                    or {}
                )
                if isinstance(default_req, dict):
                    if "cpu" in default_req and default_cpu is None:
                        default_cpu = _to_decimal(default_req["cpu"])
                    if "memory" in default_req and default_mem is None:
                        default_mem = _to_decimal(default_req["memory"])
                default_lim = getattr(item, "default", None) or {}
                if isinstance(default_lim, dict):
                    if "cpu" in default_lim and default_cpu is None:
                        default_cpu = _to_decimal(default_lim["cpu"])
                    if "memory" in default_lim and default_mem is None:
                        default_mem = _to_decimal(default_lim["memory"])
        return default_cpu, default_mem, False
    except ApiException as exc:
        if getattr(exc, "status", None) in (401, 403):
            logger.debug(
                "Cannot list LimitRange in namespace '%s' (%s: %s). "
                "Default request injection could not be inspected.",
                effective_ns,
                exc.status,
                exc.reason,
            )
            return None, None, True
        return None, None, False
    except Exception:
        return None, None, False


def check_resource_quota(
    namespace: str | None = None,
    resources_per_node: dict[str, Any] | None = None,
    num_nodes: int = 1,
    *,
    core_v1_api: Any = None,
) -> ToolError | None:
    """Pre-flight check verifying requested resources against namespace ResourceQuota.

    Calculates total requirements across all worker nodes (`num_nodes * per_node`)
    and verifies that CPU, memory, GPU, and pod counts do not exceed remaining
    unallocated quota.

    Returns:
        ToolError with error_code=QUOTA_EXCEEDED if quota would be exceeded.
        None if all quotas are satisfied or if no quotas are configured.
    """
    try:
        effective_ns = get_trainer_effective_namespace(namespace)
        api = core_v1_api or get_core_v1_api()
        quota_list = api.list_namespaced_resource_quota(
            namespace=effective_ns,
            _request_timeout=K8S_TIMEOUT,
        )
    except ApiException as e:
        logger.debug(
            "ResourceQuota check skipped for namespace '%s' (status %s): %s",
            namespace,
            e.status,
            e.reason,
        )
        return None
    except Exception as e:
        logger.debug(
            "Failed to retrieve ResourceQuota for namespace '%s': %s",
            namespace,
            e,
        )
        return None

    items = _extract_quota_items(quota_list)
    if not items:
        return None

    nodes_dec = Decimal(max(1, num_nodes))
    req_cpu, req_mem, req_gpu = _extract_requested_resources(resources_per_node or {}, nodes_dec)

    effective_cpu = req_cpu
    effective_mem = req_mem
    lr_forbidden = False
    if effective_cpu is None or effective_mem is None:
        def_cpu, def_mem, lr_forbidden = _get_namespace_limit_range_defaults(effective_ns, api)
        if effective_cpu is None and def_cpu is not None:
            effective_cpu = def_cpu * nodes_dec
        if effective_mem is None and def_mem is not None:
            effective_mem = def_mem * nodes_dec

    for q in items:
        q_name, hard, used, scopes = _extract_quota_data(q)
        if not hard:
            continue
        # Skip BestEffort-only quotas if:
        # 1. Workload requests compute resources (or receives them via LimitRange defaults).
        # 2. LimitRange lookup was forbidden (cannot determine if defaults will be injected at admission,
        #    so skip BestEffort quota to avoid false rejections).
        # In Kubernetes, pods without compute requests or limits have BestEffort QoS.
        if scopes == ["BestEffort"] and (effective_cpu or effective_mem or lr_forbidden):
            continue

        err = _check_single_quota(
            q_name=q_name,
            effective_ns=effective_ns,
            hard=hard,
            used=used,
            nodes_dec=nodes_dec,
            req_cpu=effective_cpu,
            req_mem=effective_mem,
            req_gpu=req_gpu,
        )
        if err:
            return err

    return None
