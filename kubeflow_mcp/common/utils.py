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

"""SDK client factories with caching and timeout configuration.

Mirrors kubeflow SDK's client structure:
- TrainerClient from kubeflow.trainer
"""

import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Any

# Import at module level to avoid import deadlocks when tools are called rapidly
from kubeflow.trainer import TrainerClient

if TYPE_CHECKING:
    from kubernetes import client as k8s_client

K8S_TIMEOUT = 5


@lru_cache(maxsize=1)
def _get_api_client() -> "k8s_client.ApiClient":
    """Create and cache a single ApiClient with strict timeouts.

    Cached so all higher-level API objects share one connection pool.
    Call ``reset_clients()`` to force re-creation (e.g. after kubeconfig change).
    """
    from kubernetes import client, config

    config.load_config()
    configuration = client.Configuration.get_default_copy()
    configuration.retries = 1
    configuration.socket_options = None  # rely on OS defaults
    configuration.connect_timeout = K8S_TIMEOUT
    configuration.read_timeout = K8S_TIMEOUT
    return client.ApiClient(configuration)


def get_core_v1_api() -> "k8s_client.CoreV1Api":
    """Get CoreV1Api backed by the shared, timeout-configured ApiClient."""
    from kubernetes import client

    return client.CoreV1Api(_get_api_client())


def get_version_api() -> "k8s_client.VersionApi":
    """Get VersionApi backed by the shared, timeout-configured ApiClient."""
    from kubernetes import client

    return client.VersionApi(_get_api_client())


def get_custom_objects_api() -> "k8s_client.CustomObjectsApi":
    """Get CustomObjectsApi backed by the shared, timeout-configured ApiClient."""
    from kubernetes import client

    return client.CustomObjectsApi(_get_api_client())


def get_apiextensions_api() -> "k8s_client.ApiextensionsV1Api":
    """Get ApiextensionsV1Api backed by the shared, timeout-configured ApiClient."""
    from kubernetes import client

    return client.ApiextensionsV1Api(_get_api_client())


def get_trainer_effective_namespace(namespace: str | None = None) -> str:
    """Namespace for TrainJob operations: explicit arg, then SDK backend, else ``default``.

    Aligns direct CustomObjects calls with :class:`TrainerClient` (Kubernetes backend).
    """
    if namespace:
        return namespace
    client = get_trainer_client()
    backend = client.backend
    ns = getattr(backend, "namespace", None)
    if ns is not None:
        return str(ns)
    return "default"


def get_trainer_custom_objects_api() -> "k8s_client.CustomObjectsApi":
    """CustomObjectsApi configured with the same kubeconfig as the MCP server.

    Always creates a fresh API instance from the shared :func:`_get_api_client`
    rather than extracting the SDK backend's internal client, which may hold
    stale connections or divergent auth configuration in long-running processes.
    """
    return get_custom_objects_api()


@lru_cache(maxsize=1)
def get_trainer_client() -> TrainerClient:
    """Get or create TrainerClient singleton.

    Uses default KubernetesBackendConfig with current kubeconfig context.
    """
    return TrainerClient()


_ns_client_cache: dict[str, Any] = {}
_ns_client_lock = threading.Lock()
_NS_CLIENT_CACHE_MAX = 64


def get_trainer_client_for_namespace(namespace: str | None = None) -> Any:
    """Return a TrainerClient targeting the given namespace.

    When *namespace* is ``None`` the shared singleton (default kubeconfig
    namespace) is returned.  When a namespace is explicitly provided a
    cached ``TrainerClient`` scoped to that namespace is returned.
    """
    if namespace is None:
        return get_trainer_client()
    with _ns_client_lock:
        if namespace in _ns_client_cache:
            return _ns_client_cache[namespace]
        from kubeflow.common.types import KubernetesBackendConfig

        client = TrainerClient(backend_config=KubernetesBackendConfig(namespace=namespace))
        if len(_ns_client_cache) >= _NS_CLIENT_CACHE_MAX:
            oldest = next(iter(_ns_client_cache))
            del _ns_client_cache[oldest]
        _ns_client_cache[namespace] = client
        return client


MCP_MANAGED_LABEL = "kubeflow-mcp/managed-by"
MCP_MANAGED_VALUE = "mcp"
_TRAINJOB_GROUP = "trainer.kubeflow.org"
_TRAINJOB_VERSION = "v1alpha1"
_TRAINJOB_PLURAL = "trainjobs"


def is_mcp_managed(name: str, namespace: str) -> bool | None:
    """Check if a TrainJob was created through MCP (has the ownership label).

    Returns:
        True if the job has the MCP ownership label.
        False if the job exists but lacks the label.
        None if the check could not be performed (API error, permissions).
    """
    try:
        api = get_custom_objects_api()
        obj = api.get_namespaced_custom_object(
            group=_TRAINJOB_GROUP,
            version=_TRAINJOB_VERSION,
            namespace=namespace,
            plural=_TRAINJOB_PLURAL,
            name=name,
            _request_timeout=K8S_TIMEOUT,
        )
        labels = obj.get("metadata", {}).get("labels", {})
        return labels.get(MCP_MANAGED_LABEL) == MCP_MANAGED_VALUE
    except Exception as e:
        from kubeflow_mcp.common.types import is_k8s_not_found

        if is_k8s_not_found(e):
            return False
        return None


def reset_clients() -> None:
    """Reset all cached clients (for testing or kubeconfig rotation)."""
    get_trainer_client.cache_clear()
    _get_api_client.cache_clear()
    with _ns_client_lock:
        _ns_client_cache.clear()
