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

"""Kueue Workload discovery and status resolution for Kubeflow Trainer jobs."""

import logging
import threading
import time
from typing import Any

from kubernetes import client as k8s_client

from kubeflow_mcp.common.utils import (
    K8S_TIMEOUT,
    _get_api_client,
    get_custom_objects_api,
)

logger = logging.getLogger(__name__)

_KUEUE_GROUP = "kueue.x-k8s.io"
_WORKLOAD_PLURAL = "workloads"
_TRAINJOB_GROUP = "trainer.kubeflow.org"
_TRAINJOB_VERSION = "v1alpha1"
_TRAINJOB_PLURAL = "trainjobs"
_QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
_JOB_UID_LABEL = "kueue.x-k8s.io/job-uid"

MAX_QUEUE_MESSAGE_LENGTH = 500
NEGATIVE_CACHE_TTL = 60.0

# Cache: host -> (served_version | None, expire_monotonic_timestamp | None)
# Positive: (version_str, None) - valid until reset
# Negative: (None, expire_time) - cached for NEGATIVE_CACHE_TTL
_discovery_cache: dict[str, tuple[str | None, float | None]] = {}
_cache_lock = threading.Lock()


def reset_kueue_cache() -> None:
    """Reset the Kueue API discovery cache (for testing or context rotation)."""
    with _cache_lock:
        _discovery_cache.clear()


def discover_kueue_version() -> str | None:
    """Check if kueue.x-k8s.io API group is served and return preferred version.

    Uses tenant-safe /apis discovery accessible to all authenticated ServiceAccounts.
    Caches positive hits persistently and negative hits (404) with a short TTL.
    Transient errors (timeouts, 5xx) are never cached.
    """
    now = time.monotonic()
    try:
        api_client = _get_api_client()
        host = getattr(api_client.configuration, "host", "default")
    except Exception:
        host = "default"

    with _cache_lock:
        cached = _discovery_cache.get(host)
        if cached is not None:
            version, expires_at = cached
            if version is not None:
                return version
            if expires_at is not None and now < expires_at:
                return None

    try:
        apis_api = k8s_client.ApisApi(_get_api_client())
        group_list = apis_api.get_api_versions(_request_timeout=K8S_TIMEOUT)
        served_version: str | None = None
        for group in getattr(group_list, "groups", []):
            if getattr(group, "name", None) == _KUEUE_GROUP:
                pref = getattr(group, "preferred_version", None)
                if pref and getattr(pref, "version", None):
                    served_version = pref.version
                else:
                    versions = getattr(group, "versions", [])
                    if versions and getattr(versions[0], "version", None):
                        served_version = versions[0].version
                    else:
                        served_version = "v1beta1"
                break

        with _cache_lock:
            if served_version:
                _discovery_cache[host] = (served_version, None)
            else:
                _discovery_cache[host] = (None, now + NEGATIVE_CACHE_TTL)
        return served_version

    except Exception as e:
        status = getattr(e, "status", None)
        if status == 404:
            with _cache_lock:
                _discovery_cache[host] = (None, now + NEGATIVE_CACHE_TTL)
        else:
            logger.debug("Kueue API discovery error: %s", e)
        return None


def get_trainjob_queue_status(
    name: str,
    namespace: str,
    job_status: str | None = None,
) -> dict[str, Any] | None:
    """Resolve live Kueue Workload admission status for a TrainJob.

    Returns None if:
    - The job is already in a terminal state (Complete, Failed)
    - Kueue is not installed or unreachable
    - The job is not managed by Kueue
    - Caller lacks RBAC to read Workloads (fail-open)
    """
    if job_status in ("Complete", "Failed"):
        return None

    try:
        served_version = discover_kueue_version()
        if not served_version:
            return None

        try:
            api = get_custom_objects_api()
            job_cr = api.get_namespaced_custom_object(
                group=_TRAINJOB_GROUP,
                version=_TRAINJOB_VERSION,
                namespace=namespace,
                plural=_TRAINJOB_PLURAL,
                name=name,
                _request_timeout=K8S_TIMEOUT,
            )
        except Exception as e:
            logger.debug("Failed to read TrainJob %s/%s for Kueue check: %s", namespace, name, e)
            return None

        metadata = job_cr.get("metadata", {})
        job_uid = metadata.get("uid")
        if not job_uid:
            return None

        labels = metadata.get("labels", {})
        queue_name = labels.get(_QUEUE_LABEL)

        try:
            workloads = api.list_namespaced_custom_object(
                group=_KUEUE_GROUP,
                version=served_version,
                namespace=namespace,
                plural=_WORKLOAD_PLURAL,
                label_selector=f"{_JOB_UID_LABEL}={job_uid}",
                limit=10,
                _request_timeout=K8S_TIMEOUT,
            )
        except Exception as e:
            logger.debug("Failed to list Kueue Workloads for %s/%s: %s", namespace, name, e)
            return None

        items = workloads.get("items", [])
        if not items:
            if queue_name and (job_cr.get("spec") or {}).get("suspend") is True:
                return {
                    "queue_name": queue_name,
                    "state": "queued",
                    "reason": "Reconciling",
                    "message": "Workload is reconciling in queue",
                }
            return None

        return _parse_workload_status(items[0], queue_name)
    except Exception as e:
        logger.debug("Unexpected error resolving Kueue status for %s/%s: %s", namespace, name, e)
        return None


def _parse_workload_status(wl: dict[str, Any], queue_name: str | None) -> dict[str, Any]:
    """Parse Workload conditions and metadata into a normalized status dictionary."""
    wl_queue = wl.get("spec", {}).get("queueName") or queue_name
    cluster_queue = wl.get("status", {}).get("admission", {}).get("clusterQueue")
    requeue_count = wl.get("status", {}).get("requeueState", {}).get("count")

    # Map status conditions per Kueue workload_types.go
    conditions = wl.get("status", {}).get("conditions", [])
    cond_map: dict[str, dict[str, Any]] = {}
    for c in conditions:
        if isinstance(c, dict) and "type" in c:
            cond_map[c["type"]] = c

    evicted = cond_map.get("Evicted")
    admitted = cond_map.get("Admitted")
    quota = cond_map.get("QuotaReserved")

    state = "queued"
    reason = "Reconciling"
    message = ""

    if evicted and str(evicted.get("status", "")).lower() == "true":
        state = "evicted"
        reason = evicted.get("reason", "Evicted")
        message = evicted.get("message", "")
    elif wl.get("spec", {}).get("active") is False:
        state = "evicted"
        reason = "Deactivated"
        message = "The workload is deactivated"
    elif admitted and str(admitted.get("status", "")).lower() == "true":
        state = "admitted"
        reason = admitted.get("reason", "Admitted")
        message = admitted.get("message", "")
    elif (
        quota
        and str(quota.get("status", "")).lower() == "false"
        and quota.get("reason") == "Inadmissible"
    ):
        state = "inadmissible"
        reason = "Inadmissible"
        message = quota.get("message", "")
    elif (
        quota
        and str(quota.get("status", "")).lower() == "false"
        and quota.get("reason") == "Pending"
    ):
        state = "queued"
        reason = "Pending"
        message = quota.get("message", "")
    elif admitted and str(admitted.get("status", "")).lower() == "false":
        state = "queued"
        reason = admitted.get("reason", "Pending")
        message = admitted.get("message", "")
    else:
        state = "queued"
        reason = "Reconciling"
        message = "Workload conditions pending"

    if message:
        message = message[:MAX_QUEUE_MESSAGE_LENGTH]

    result: dict[str, Any] = {
        "queue_name": wl_queue,
        "state": state,
        "reason": reason,
    }
    if message:
        result["message"] = message
    if cluster_queue:
        result["cluster_queue"] = cluster_queue
    if requeue_count is not None:
        result["requeue_count"] = requeue_count

    return result
