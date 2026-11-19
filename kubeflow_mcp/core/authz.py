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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import anyio
from fastmcp.server.middleware import Middleware
from fastmcp.tools import ToolResult
from mcp.types import TextContent

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError, ToolPermission
from kubeflow_mcp.core.config import AuthzConfig

logger = logging.getLogger(__name__)

# Without a namespace argument these tools find their namespace themselves (the
# controller's), so the check cannot know it in advance and asks cluster-wide instead.
_SELF_RESOLVED_NAMESPACE = {"inspect_controller"}

# A permission to check, and the namespace to check it in (None: cluster-wide).
Check = tuple[ToolPermission, str | None]


class AuthzError(Exception):
    """A decision could not be made, so the call is refused."""


@dataclass(frozen=True)
class CallerIdentity:
    user: str
    groups: tuple[str, ...]


def identity_from_claims(claims: Mapping[str, Any], config: AuthzConfig) -> CallerIdentity | None:
    """Map verified token claims to the Kubernetes user and groups to ask about."""
    user = claims.get(config.user_claim)
    if not isinstance(user, str) or not user:
        return None
    groups = claims.get(config.groups_claim) or []
    if isinstance(groups, str):
        groups = [groups]
    return CallerIdentity(
        user=config.username_prefix + user,
        groups=tuple(config.groups_prefix + g for g in groups if isinstance(g, str) and g),
    )


def required_checks(
    tool: str,
    permissions: tuple[ToolPermission, ...],
    arguments: Mapping[str, Any],
    resolve_namespace: Callable[[str | None], str],
) -> list[Check]:
    """Return the permissions this call needs and the namespace of each.

    A permission with ``when`` is needed only if that argument is truthy in this call.
    """
    needed = [p for p in permissions if p.when is None or arguments.get(p.when)]
    if not any(not p.cluster_wide for p in needed):
        return [(p, None) for p in needed]
    namespace = arguments.get("namespace")
    # Middleware runs before argument validation, so never trust the type.
    if namespace is not None and not isinstance(namespace, str):
        raise ValueError("namespace must be a string")
    if namespace is None and tool in _SELF_RESOLVED_NAMESPACE:
        effective = None
    else:
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
                self._cache[key] = (now, allowed)
        return allowed

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


def _refuse(error: str, details: dict[str, Any] | None = None) -> ToolResult:
    body = ToolError(
        error=error, error_code=ErrorCode.PERMISSION_DENIED, details=details
    ).model_dump()
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
        if tool not in self._permissions:
            return await call_next(context)

        claims = self._get_claims()
        identity = identity_from_claims(claims or {}, self._config)
        if identity is None:
            return _refuse(
                f"Cannot authorize '{tool}': the request carries no verified "
                f"'{self._config.user_claim}' claim"
            )
        try:
            missing = await anyio.to_thread.run_sync(self._check, tool, arguments, identity)
        except AuthzError as e:
            logger.warning("authz_unavailable", extra={"tool": tool, "error": str(e)})
            return _refuse(f"Cannot authorize '{tool}': {e}")
        if missing:
            logger.info(
                "authz_denied", extra={"tool": tool, "user": identity.user, "missing": missing}
            )
            return _refuse(
                f"'{identity.user}' may not call '{tool}' here: cannot {'; cannot '.join(missing)}",
                {"missing_permissions": missing, "user": identity.user},
            )
        return await call_next(context)

    def _check(
        self, tool: str, arguments: Mapping[str, Any], identity: CallerIdentity
    ) -> list[str]:
        resolver = self._resolvers.get(tool)
        try:
            checks = required_checks(
                tool, self._permissions[tool], arguments, resolver or _no_default_namespace
            )
        except Exception as e:
            raise AuthzError(
                f"cannot resolve the namespace ({e}); pass namespace= explicitly"
            ) from e
        return self._reviewer.missing(identity, checks)


def _no_default_namespace(namespace: str | None) -> str:
    if not namespace:
        raise ValueError("no default namespace for this tool")
    return namespace
