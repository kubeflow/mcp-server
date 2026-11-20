# Copyright The Kubeflow Authors.
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

"""Per-caller authorization with Kubernetes SubjectAccessReview (#565).

Personas decide which tools a caller sees, but every call runs with the server's own
Kubernetes account. With ``authz.mode: sar`` the server first asks the API server
whether the *caller* may make each request the tool declares in
``CLIENT_TOOL_PERMISSIONS``, and refuses the call with the missing permissions if not.

- The caller is the verified token's user and groups, never a tool argument or request
  metadata, so this needs JWT auth.
- It only narrows: the server's own RBAC still bounds every call.
- It fails closed: a check that cannot be made refuses the call.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

import anyio
from fastmcp.server.middleware import Middleware
from fastmcp.tools import ToolResult
from mcp.types import TextContent

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolPermission
from kubeflow_mcp.core.config import AuthzConfig
from kubeflow_mcp.core.security import validate_namespace

logger = logging.getLogger(__name__)

# Kubernetes reserves these names; a token claiming one could borrow a system
# identity (system:masters, a kube-system ServiceAccount) instead of the caller's.
_RESERVED_PREFIX = "system:"

# Callers choose namespace strings, so the cache is bounded: expired entries go first,
# then everything if a burst of fresh ones still exceeds the cap.
MAX_CACHE_ENTRIES = 10_000

# A permission to check, and the namespace to check it in (None: cluster-wide).
Check = tuple[ToolPermission, str | None]


class AuthzError(Exception):
    """A decision could not be made, so the call is refused."""


class InvalidArguments(ValueError):
    """The call's arguments are malformed; refused before any review is sent."""


@dataclass(frozen=True)
class CallerIdentity:
    user: str
    groups: tuple[str, ...]


def identity_from_claims(claims: Mapping[str, Any], config: AuthzConfig) -> CallerIdentity | None:
    """Map verified token claims to the Kubernetes user and groups to ask about.

    Returns None when there is no usable user: missing, empty, or a reserved
    ``system:`` name. Reserved group names are dropped.
    """
    user = claims.get(config.user_claim)
    if not isinstance(user, str) or not user:
        return None
    user = config.username_prefix + user
    if user.startswith(_RESERVED_PREFIX):
        return None
    groups = claims.get(config.groups_claim) or []
    if isinstance(groups, str):
        groups = [groups]
    named = (config.groups_prefix + g for g in groups if isinstance(g, str) and g)
    # The API server adds this group to every authenticated request, but a
    # SubjectAccessReview only knows the groups it is sent.
    return CallerIdentity(
        user=user,
        groups=(
            *(g for g in named if not g.startswith(_RESERVED_PREFIX)),
            "system:authenticated",
        ),
    )


def required_checks(
    permissions: tuple[ToolPermission, ...],
    arguments: Mapping[str, Any],
    resolve_namespace: Callable[[str | None], str | None],
) -> list[Check]:
    """Return the permissions this call needs and the namespace of each.

    A permission with ``when`` is needed only if that argument is truthy in this call.
    A resolver that returns None (the tool finds its namespace itself) means the
    namespaced permissions are checked cluster-wide.

    Raises:
        InvalidArguments: the namespace argument is not a valid namespace name. The
            middleware runs before the tool validates it, so it is checked here first.
    """
    needed = [p for p in permissions if p.when is None or arguments.get(p.when)]
    if not any(not p.cluster_wide for p in needed):
        return [(p, None) for p in needed]
    namespace = arguments.get("namespace")
    if namespace is not None:
        if not isinstance(namespace, str):
            raise InvalidArguments("namespace must be a string")
        err = validate_namespace(namespace)
        if err is not None:
            raise InvalidArguments(err.error)
    effective = resolve_namespace(namespace)
    return [(p, None if p.cluster_wide else effective) for p in needed]


def describe(permission: ToolPermission, namespace: str | None) -> str:
    """Say what is missing the way ``kubectl auth can-i`` names it."""
    resource = permission.resource
    if permission.group:
        base, _, sub = resource.partition("/")
        resource = f"{base}.{permission.group}" + (f"/{sub}" if sub else "")
    where = f"in namespace '{namespace}'" if namespace else "cluster-wide"
    return f"{permission.verb} {resource} {where}"


class SubjectAccessReviewer:
    """Ask the API server whether a caller may make a request, with a short cache."""

    def __init__(
        self,
        api_factory: Callable[[], Any],
        cache_ttl: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._api_factory = api_factory
        self._ttl = cache_ttl
        self._clock = clock
        self._cache: dict[tuple[Any, ...], tuple[float, bool]] = {}
        self._lock = threading.Lock()

    def missing(self, identity: CallerIdentity, checks: list[Check]) -> list[str]:
        """Return the checks the caller fails, described. Raises AuthzError if unsure."""
        return [describe(p, ns) for p, ns in checks if not self._allowed(identity, p, ns)]

    def _allowed(self, identity: CallerIdentity, p: ToolPermission, namespace: str | None) -> bool:
        key = (identity, p.verb, p.group, p.resource, namespace)
        now = self._clock()
        if self._ttl > 0:
            with self._lock:
                hit = self._cache.get(key)
            if hit is not None and now - hit[0] < self._ttl:
                return hit[1]
        allowed = self._ask(identity, p, namespace)
        if self._ttl > 0:
            with self._lock:
                if len(self._cache) >= MAX_CACHE_ENTRIES:
                    self._evict(now)
                self._cache[key] = (now, allowed)
        return allowed

    def _evict(self, now: float) -> None:
        self._cache = {k: v for k, v in self._cache.items() if now - v[0] < self._ttl}
        if len(self._cache) >= MAX_CACHE_ENTRIES:
            self._cache.clear()

    def _ask(self, identity: CallerIdentity, p: ToolPermission, namespace: str | None) -> bool:
        from kubernetes import client

        from kubeflow_mcp.common.utils import K8S_TIMEOUT

        resource, _, subresource = p.resource.partition("/")
        review = client.V1SubjectAccessReview(
            spec=client.V1SubjectAccessReviewSpec(
                user=identity.user,
                groups=list(identity.groups),
                resource_attributes=client.V1ResourceAttributes(
                    verb=p.verb,
                    group=p.group,
                    resource=resource,
                    subresource=subresource or None,
                    namespace=namespace,
                ),
            )
        )
        try:
            result = self._api_factory().create_subject_access_review(
                review, _request_timeout=K8S_TIMEOUT
            )
        except Exception as e:
            raise AuthzError(f"SubjectAccessReview failed: {e}") from e
        status = getattr(result, "status", None)
        if status is None:
            raise AuthzError("SubjectAccessReview returned no status")
        return bool(status.allowed) and not bool(status.denied)


def _refuse(
    error: str,
    details: dict[str, Any] | None = None,
    error_code: ErrorCode = ErrorCode.PERMISSION_DENIED,
) -> ToolResult:
    body = ToolError(error=error, error_code=error_code, details=details).model_dump()
    return ToolResult(
        content=[TextContent(type="text", text=error)],
        structured_content=body,
        is_error=True,
    )


class SarAuthorizationMiddleware(Middleware):
    """Refuse a tool call the caller's own RBAC would not allow, before it runs.

    ``execute_tool`` (progressive and semantic modes) is checked as the tool it runs.
    Tools without declared permissions, such as the mode meta-tools, pass through.
    """

    def __init__(
        self,
        config: AuthzConfig,
        permissions: Mapping[str, tuple[ToolPermission, ...]],
        namespace_resolvers: Mapping[str, Callable[[str | None], str]],
        reviewer: SubjectAccessReviewer,
        get_claims: Callable[[], Mapping[str, Any] | None],
    ) -> None:
        self._config = config
        self._permissions = permissions
        self._resolvers = namespace_resolvers
        self._reviewer = reviewer
        self._get_claims = get_claims

    async def on_call_tool(self, context: Any, call_next: Any) -> ToolResult:
        tool = context.message.name
        arguments = context.message.arguments or {}
        if tool == "execute_tool":
            tool = arguments.get("tool_name")
            arguments = arguments.get("arguments") or {}
            # Malformed input must not slip past the check, so it is refused here.
            if not isinstance(tool, str) or not isinstance(arguments, dict):
                return _refuse(
                    "execute_tool needs tool_name as a string and arguments as an object",
                    error_code=ErrorCode.VALIDATION_ERROR,
                )
        if tool not in self._permissions:
            return await call_next(context)

        claims = self._get_claims()
        identity = identity_from_claims(claims or {}, self._config)
        if identity is None:
            return _refuse(
                f"Cannot authorize '{tool}': the request carries no usable verified "
                f"'{self._config.user_claim}' claim (missing, or a reserved 'system:' name)"
            )
        try:
            missing = await anyio.to_thread.run_sync(self._check, tool, arguments, identity)
        except InvalidArguments as e:
            return _refuse(str(e), error_code=ErrorCode.VALIDATION_ERROR)
        except AuthzError as e:
            logger.warning(
                "authz_unavailable", extra={"audit": True, "tool": tool, "error": str(e)}
            )
            return _refuse(f"Cannot authorize '{tool}': {e}")
        if missing:
            logger.info(
                "authz_denied",
                extra={
                    "audit": True,
                    "tool": tool,
                    "user": identity.user,
                    "missing_permissions": missing,
                },
            )
            return _refuse(
                f"'{identity.user}' may not call '{tool}' here: cannot {'; cannot '.join(missing)}",
                {"missing_permissions": missing, "user": identity.user},
            )
        return await call_next(context)

    def _check(
        self, tool: str, arguments: Mapping[str, Any], identity: CallerIdentity
    ) -> list[str]:
        resolver = self._resolvers.get(tool) or _no_default_namespace
        try:
            checks = required_checks(self._permissions[tool], arguments, resolver)
        except InvalidArguments:
            raise
        except Exception as e:
            raise AuthzError(f"cannot work out what this call needs ({e})") from e
        return self._reviewer.missing(identity, checks)


def _no_default_namespace(namespace: str | None) -> str:
    if not namespace:
        raise ValueError("no default namespace for this tool")
    return namespace


def _namespace_resolvers() -> dict[str, Callable[[str | None], str]]:
    from kubeflow_mcp.common import utils

    return {
        "trainer": utils.get_trainer_effective_namespace,
        "spark": utils.get_spark_effective_namespace,
        "pipelines": utils.get_pipelines_effective_namespace,
        # Health tools use the Trainer client's namespace.
        "health": utils.get_trainer_effective_namespace,
    }


def _controller_namespace(namespace: str | None) -> str | None:
    # Where inspect_controller reads: the argument, then the configured controller
    # namespace, else None (the tool searches for it, so check cluster-wide).
    from kubeflow_mcp.trainer.api.platform import _get_controller_namespace

    return namespace or _get_controller_namespace()


def _verified_claims() -> Mapping[str, Any] | None:
    from fastmcp.server.dependencies import get_access_token

    token = get_access_token()
    return token.claims if token is not None else None


def build_authz_middleware(
    config: AuthzConfig,
    loaded_modules: Mapping[str, Any],
    auth_provider: Any,
    registered: Collection[str],
) -> SarAuthorizationMiddleware:
    """Build the middleware for the loaded clients; refuse a setup that cannot work.

    Raises:
        ValueError: when there is no per-caller identity to check (no auth, or a shared
            API key), since every call would then be refused.
    """
    from kubeflow_mcp.common.utils import get_authorization_api
    from kubeflow_mcp.core.auth import APIKeyVerifier
    from kubeflow_mcp.core.health import HEALTH_TOOL_PERMISSIONS

    if auth_provider is None or isinstance(auth_provider, APIKeyVerifier):
        raise ValueError(
            "authz mode 'sar' needs JWT auth (auth.jwks_uri): the caller's user comes "
            "from the verified token, and an API key carries no per-user identity"
        )
    resolvers = _namespace_resolvers()
    permissions: dict[str, tuple[ToolPermission, ...]] = {}
    tool_resolvers: dict[str, Callable[[str | None], str]] = {}
    tables = [
        (name, getattr(m, "CLIENT_TOOL_PERMISSIONS", {})) for name, m in loaded_modules.items()
    ]
    for client, table in [*tables, ("health", HEALTH_TOOL_PERMISSIONS)]:
        for tool, declared in table.items():
            if tool not in registered:
                continue
            permissions[tool] = declared
            if client in resolvers:
                tool_resolvers[tool] = resolvers[client]
    if "inspect_controller" in permissions:
        tool_resolvers["inspect_controller"] = _controller_namespace
    return SarAuthorizationMiddleware(
        config,
        permissions,
        tool_resolvers,
        SubjectAccessReviewer(get_authorization_api, config.cache_ttl),
        _verified_claims,
    )
