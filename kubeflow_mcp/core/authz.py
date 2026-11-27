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
from typing import Any, NamedTuple, get_type_hints

import anyio
from fastmcp.server.middleware import Middleware
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import TypeAdapter, ValidationError

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

# Lax, like the tools' own argument validation: "false" and 0 are False.
_BOOL = TypeAdapter(bool)


class Check(NamedTuple):
    """A permission to check for one call."""

    permission: ToolPermission
    namespace: str | None  # None: cluster-wide
    name: str | None = None  # the object requested, when the permission names one


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

    Returns None when there is no usable identity: the user is missing, empty or a
    reserved ``system:`` name, or the groups claim is neither a string nor a list.
    Reserved group names are dropped.
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
    if not isinstance(groups, (list, tuple)):
        return None
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


def _given(param: str, arguments: Mapping[str, Any], bool_params: Collection[str]) -> bool:
    """Whether the call sets *param*, read the way the tool's argument validation will.

    A boolean parameter counts when it validates to True ("false" does not); any
    other parameter counts when it is present and not None (an empty name is given).
    """
    value = arguments.get(param)
    if param not in bool_params:
        return value is not None
    if value is None:
        return False
    try:
        return _BOOL.validate_python(value)
    except ValidationError as e:
        raise InvalidArguments(f"{param} must be a boolean") from e


def required_checks(
    permissions: tuple[ToolPermission, ...],
    arguments: Mapping[str, Any],
    resolve_namespace: Callable[[str | None], str | None],
    bool_params: Collection[str] = frozenset(),
) -> list[Check]:
    """Return the permissions this call needs and the namespace of each.

    A permission with ``when`` is needed only if the call sets that argument, one with
    ``unless`` only if it does not. A resolver that returns None (the tool finds its
    namespace itself) means the namespaced permissions are checked cluster-wide.

    Raises:
        InvalidArguments: the namespace argument is not a valid namespace name. The
            middleware runs before the tool validates it, so it is checked here first.
    """
    needed = [
        p
        for p in permissions
        if (p.when is None or _given(p.when, arguments, bool_params))
        and (p.unless is None or not _given(p.unless, arguments, bool_params))
    ]
    if not any(not p.cluster_wide for p in needed):
        return [Check(p, None, _object_name(p, arguments)) for p in needed]
    namespace = arguments.get("namespace")
    if namespace is not None:
        if not isinstance(namespace, str):
            raise InvalidArguments("namespace must be a string")
        err = validate_namespace(namespace)
        if err is not None:
            raise InvalidArguments(err.error)
    effective = resolve_namespace(namespace)
    return [
        Check(p, None if p.cluster_wide else effective, _object_name(p, arguments)) for p in needed
    ]


def _object_name(permission: ToolPermission, arguments: Mapping[str, Any]) -> str | None:
    """The name of the object the request is for, so rules with resourceNames can match.

    Anything but a non-empty string is left out: the tool rejects it before it sends
    the request, and a review without a name matches only rules for every object.
    """
    value = arguments.get(permission.name) if permission.name else None
    return value if isinstance(value, str) and value else None


def describe(permission: ToolPermission, namespace: str | None, name: str | None = None) -> str:
    """Say what is missing the way ``kubectl auth can-i`` names it."""
    resource = permission.resource
    if permission.group:
        base, _, sub = resource.partition("/")
        resource = f"{base}.{permission.group}" + (f"/{sub}" if sub else "")
    named = f" '{name}'" if name else ""
    where = f"in namespace '{namespace}'" if namespace else "cluster-wide"
    return f"{permission.verb} {resource}{named} {where}"


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
        return [describe(*check) for check in checks if not self._allowed(identity, check)]

    def _allowed(self, identity: CallerIdentity, check: Check) -> bool:
        p = check.permission
        key = (identity, p.verb, p.group, p.resource, check.namespace, check.name)
        now = self._clock()
        if self._ttl > 0:
            with self._lock:
                hit = self._cache.get(key)
            if hit is not None and now - hit[0] < self._ttl:
                return hit[1]
        allowed = self._ask(identity, check)
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

    def _ask(self, identity: CallerIdentity, check: Check) -> bool:
        from kubernetes import client

        from kubeflow_mcp.common.utils import K8S_TIMEOUT

        p = check.permission
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
                    namespace=check.namespace,
                    name=check.name,
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
        bool_params: Mapping[str, Collection[str]] | None = None,
    ) -> None:
        self._config = config
        self._permissions = permissions
        self._resolvers = namespace_resolvers
        self._reviewer = reviewer
        self._get_claims = get_claims
        self._bool_params = bool_params or {}

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
        # Nothing declared (unknown, hidden, or no Kubernetes request): nothing to ask.
        if not self._permissions.get(tool):
            return await call_next(context)

        claims = self._get_claims()
        identity = identity_from_claims(claims or {}, self._config)
        if identity is None:
            # Logged without the claims themselves, which may carry personal data.
            logger.warning(
                "authz_no_identity",
                extra={"audit": True, "tool": tool, "error": "no usable verified identity"},
            )
            return _refuse(
                f"Cannot authorize '{tool}': the request carries no usable verified "
                f"identity (the '{self._config.user_claim}' claim is missing or reserved, "
                f"or the '{self._config.groups_claim}' claim is malformed)"
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
            checks = required_checks(
                self._permissions[tool], arguments, resolver, self._bool_params.get(tool, ())
            )
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
        "optimizer": utils.get_optimizer_effective_namespace,
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
    from kubeflow_mcp.core.health import HEALTH_TOOL_PERMISSIONS, HEALTH_TOOLS

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
    functions = [f for m in loaded_modules.values() for f in getattr(m, "TOOLS", [])]
    bool_params = {
        f.__name__: frozenset(n for n, t in get_type_hints(f).items() if t is bool)
        for f in [*functions, *HEALTH_TOOLS]
        if f.__name__ in permissions
    }
    return SarAuthorizationMiddleware(
        config,
        permissions,
        tool_resolvers,
        SubjectAccessReviewer(get_authorization_api, config.cache_ttl),
        _verified_claims,
        bool_params,
    )
