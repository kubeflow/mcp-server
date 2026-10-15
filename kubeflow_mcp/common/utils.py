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

By default every client is built from the process kubeconfig (or in-cluster
ServiceAccount) and cached for the life of the process. Inside a
:func:`request_client_configuration` block, every helper here instead builds
from the configuration supplied for that block and caches nothing beyond it, so
a deployment can run each request as a different Kubernetes identity. See
``RequestClientConfigurationMiddleware`` in :mod:`kubeflow_mcp.core.middleware`.
"""

import contextlib
import contextvars
import logging
import os
import threading
from collections.abc import Iterator
from functools import lru_cache
from typing import TYPE_CHECKING, Any

# Import at module level to avoid import deadlocks when tools are called rapidly
from kubeflow.trainer import TrainerClient

if TYPE_CHECKING:
    from kubernetes import client as k8s_client

logger = logging.getLogger(__name__)

K8S_TIMEOUT = 5


class _RequestClients:
    """Clients built from one request's configuration, closed when the request ends.

    Each ``ApiClient`` starts a thread pool on its first ``async_req`` call (the
    Trainer SDK makes them) and registers it with ``atexit``, so a client that is
    never closed is kept alive until the process exits. Closing them here keeps
    per-request clients from accumulating.
    """

    def __init__(self, configuration: "k8s_client.Configuration") -> None:
        self.configuration = configuration
        self.trainer_clients: dict[str | None, TrainerClient] = {}
        self._api_clients: list[Any] = []
        self._lock = threading.Lock()

    def api_client(self) -> "k8s_client.ApiClient":
        from kubernetes import client

        api = client.ApiClient(self.configuration)
        self.track(api)
        return api

    def trainer_client(self, namespace: str | None) -> TrainerClient:
        with self._lock:
            if namespace in self.trainer_clients:
                return self.trainer_clients[namespace]
        from kubeflow.common.types import KubernetesBackendConfig

        trainer = TrainerClient(
            backend_config=KubernetesBackendConfig(
                namespace=namespace, client_configuration=self.configuration
            )
        )
        core_api = getattr(trainer.backend, "core_api", None)
        if core_api is not None:
            self.track(core_api.api_client)
        with self._lock:
            return self.trainer_clients.setdefault(namespace, trainer)

    def track(self, api: Any) -> None:
        with self._lock:
            self._api_clients.append(api)

    def close(self) -> None:
        with self._lock:
            api_clients, self._api_clients = self._api_clients, []
            self.trainer_clients.clear()
        for api in api_clients:
            try:
                api.close()
            except Exception:
                logger.debug("Failed to close a per-request ApiClient", exc_info=True)


_request_clients: contextvars.ContextVar[_RequestClients | None] = contextvars.ContextVar(
    "kubeflow_mcp_request_clients", default=None
)


@contextlib.contextmanager
def request_client_configuration(
    configuration: "k8s_client.Configuration",
) -> Iterator[None]:
    """Build every Kubernetes and Trainer client in this block from *configuration*.

    Inside the block, the ``get_*_api`` helpers and the ``get_trainer_client*``
    factories ignore the process kubeconfig and the process-wide caches. Clients
    are built from *configuration* and shared only within the block (one
    ``TrainerClient`` per namespace), then closed when it exits, so nothing
    built for one identity is reused for another.

    The configuration is used as given. The SDK is never asked to load a
    kubeconfig or in-cluster config for it, because a ``TrainerClient`` given a
    ``client_configuration`` skips that step.

    Args:
        configuration: Where the API server is and how to authenticate to it,
            for example a bearer token in ``api_key["authorization"]``.
    """
    scope = _RequestClients(configuration)
    token = _request_clients.set(scope)
    try:
        yield
    finally:
        _request_clients.reset(token)
        scope.close()


def _current_api_client() -> "k8s_client.ApiClient":
    """The per-request ApiClient if a request configuration is set, else the shared one."""
    scope = _request_clients.get()
    if scope is not None:
        return scope.api_client()
    return _get_api_client()


@lru_cache(maxsize=1)
def _get_api_client() -> "k8s_client.ApiClient":
    """Create and cache a single ApiClient with strict timeouts.

    Cached so all higher-level API objects share one connection pool.
    Call ``reset_clients()`` to force re-creation (e.g. after kubeconfig change).
    """
    from kubernetes import client, config

    # Not config.load_config(): it print()s its in-cluster fallback notice to
    # stdout, the MCP wire under the stdio transport. KUBECONFIG is read here
    # rather than via config.KUBE_CONFIG_DEFAULT_LOCATION, which is fixed at
    # import time and treats a path list or an empty value as missing.
    paths = os.getenv("KUBECONFIG") or "~/.kube/config"
    if any(os.path.exists(os.path.expanduser(p)) for p in paths.split(os.pathsep) if p):
        config.load_kube_config(config_file=paths)
    else:
        logger.info("No kubeconfig at %s, using in-cluster config", paths)
        config.load_incluster_config()
    configuration = client.Configuration.get_default_copy()
    configuration.retries = 1
    configuration.socket_options = None  # rely on OS defaults
    configuration.connect_timeout = K8S_TIMEOUT
    configuration.read_timeout = K8S_TIMEOUT
    return client.ApiClient(configuration)


def get_core_v1_api() -> "k8s_client.CoreV1Api":
    """Get CoreV1Api backed by the shared ApiClient, or the request's if one is set."""
    from kubernetes import client

    return client.CoreV1Api(_current_api_client())


def get_version_api() -> "k8s_client.VersionApi":
    """Get VersionApi backed by the shared ApiClient, or the request's if one is set."""
    from kubernetes import client

    return client.VersionApi(_current_api_client())


def get_custom_objects_api() -> "k8s_client.CustomObjectsApi":
    """Get CustomObjectsApi backed by the shared ApiClient, or the request's if one is set."""
    from kubernetes import client

    return client.CustomObjectsApi(_current_api_client())


def get_apiextensions_api() -> "k8s_client.ApiextensionsV1Api":
    """Get ApiextensionsV1Api backed by the shared ApiClient, or the request's if one is set."""
    from kubernetes import client

    return client.ApiextensionsV1Api(_current_api_client())


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


def get_trainer_client() -> TrainerClient:
    """Get or create TrainerClient singleton.

    Uses default KubernetesBackendConfig with current kubeconfig context, or
    the request's configuration inside :func:`request_client_configuration`.
    """
    scope = _request_clients.get()
    if scope is not None:
        return scope.trainer_client(None)
    return _get_process_trainer_client()


@lru_cache(maxsize=1)
def _get_process_trainer_client() -> TrainerClient:
    return TrainerClient()


_ns_client_cache: dict[str, Any] = {}
_ns_client_lock = threading.Lock()
_NS_CLIENT_CACHE_MAX = 64


def get_trainer_client_for_namespace(namespace: str | None = None) -> Any:
    """Return a TrainerClient targeting the given namespace.

    When *namespace* is ``None`` the shared singleton (default kubeconfig
    namespace) is returned.  When a namespace is explicitly provided a
    cached ``TrainerClient`` scoped to that namespace is returned. Inside
    :func:`request_client_configuration` the client comes from the request
    instead and the process-wide cache is neither read nor written.
    """
    if namespace is None:
        return get_trainer_client()
    scope = _request_clients.get()
    if scope is not None:
        return scope.trainer_client(namespace)
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


# Outcomes of an ownership check. "missing" is kept distinct from "unmanaged"
# because the two need opposite responses: a job that does not exist is a
# not-found, while collapsing it into "unmanaged" tells the caller to go fix
# permissions that were never the problem.
OWNERSHIP_MANAGED = "managed"
OWNERSHIP_UNMANAGED = "unmanaged"
OWNERSHIP_MISSING = "missing"


def get_trainer_ownership(name: str, namespace: str) -> str | None:
    """Classify a TrainJob for the MCP ownership gate.

    Returns:
        ``OWNERSHIP_MANAGED`` if the job carries the MCP ownership label,
        ``OWNERSHIP_UNMANAGED`` if it exists without the label,
        ``OWNERSHIP_MISSING`` if no such job exists, or
        ``None`` if the check could not be performed (API error, permissions).
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
        managed = labels.get(MCP_MANAGED_LABEL) == MCP_MANAGED_VALUE
        return OWNERSHIP_MANAGED if managed else OWNERSHIP_UNMANAGED
    except Exception as e:
        from kubeflow_mcp.common.types import is_k8s_not_found

        if is_k8s_not_found(e):
            return OWNERSHIP_MISSING
        return None


def reset_clients() -> None:
    """Reset all cached clients (for testing or kubeconfig rotation)."""
    _get_process_trainer_client.cache_clear()
    _get_api_client.cache_clear()
    with _ns_client_lock:
        _ns_client_cache.clear()
