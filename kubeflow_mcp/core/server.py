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

"""MCP server factory with dynamic client loading.

Designed for extensibility:
- Phase 1: trainer only
- Phase 2+: Contributors add optimizer, hub
"""

import functools
import importlib
import logging
import re
import time
from typing import Any

from fastmcp import FastMCP

from kubeflow_mcp.common.constants import (
    TOOL_NEXT_HINTS,
    TOOL_TO_PHASE,
    ErrorCode,
    is_infrastructure_error,
)
from kubeflow_mcp.core.dynamic_tools import get_mode_tools, init_dynamic_tools
from kubeflow_mcp.core.health import (
    HEALTH_TOOL_ANNOTATIONS,
    HEALTH_TOOL_DESCRIPTIONS,
    HEALTH_TOOLS,
)
from kubeflow_mcp.core.logging import with_correlation_id
from kubeflow_mcp.core.policy import (
    apply_policy_filters,
    get_allowed_tools,
    get_effective_persona,
    is_read_only,
)
from kubeflow_mcp.core.resilience import RateLimiter, get_breaker
from kubeflow_mcp.core.resources import register_resources
from kubeflow_mcp.core.security import mask_sensitive_data
from kubeflow_mcp.core.telemetry import get_tracer

try:
    from opentelemetry.trace import SpanKind
    from opentelemetry.trace import Status as _Status
    from opentelemetry.trace import StatusCode as _StatusCode
except ImportError:  # pragma: no cover
    SpanKind = None  # type: ignore[assignment,misc]
    _Status = None  # type: ignore[assignment,misc]
    _StatusCode = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_rate_limiter: RateLimiter | None = None


def configure_resilience(
    rate_limit: float = 10.0,
    rate_capacity: float = 20.0,
) -> None:
    """Initialize the global rate limiter from config. Called once at startup."""
    global _rate_limiter
    _rate_limiter = RateLimiter(rate=rate_limit, capacity=rate_capacity)


def _inject_meta(result: Any, tool_name: str) -> Any:
    """Inject _meta (phase + next hint) into successful tool responses.

    Provides workflow guidance for clients that don't consume server
    instructions or MCP resources (e.g. Ollama, custom agents).
    """
    if not isinstance(result, dict):
        return result
    if "error" in result or "error_code" in result:
        return result
    phase = TOOL_TO_PHASE.get(tool_name)
    hint = TOOL_NEXT_HINTS.get(tool_name)
    if phase or hint:
        meta: dict[str, str] = {}
        if phase:
            meta["phase"] = phase
        if hint:
            meta["next"] = hint
        result["_meta"] = meta
    return result


def _audit_wrap(tool_func):
    """Wrap a tool function with rate limiting, circuit breaking, audit logging, and response metadata."""
    tracer = get_tracer("kubeflow_mcp.tools")

    @functools.wraps(tool_func)
    def wrapper(**kwargs):
        tool_name = tool_func.__name__
        cid = with_correlation_id()
        persona = get_effective_persona()
        start = time.monotonic()

        span_kwargs: dict[str, Any] = {}
        if SpanKind is not None:
            span_kwargs["kind"] = SpanKind.CLIENT

        with tracer.start_as_current_span(
            f"tool:{tool_name}", **span_kwargs
        ) as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("kubeflow.persona", persona)
            span.set_attribute("correlation_id", cid)

            if _rate_limiter is not None and not _rate_limiter.acquire():
                duration_ms = int((time.monotonic() - start) * 1000)
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.duration_ms", duration_ms)
                logger.warning("rate_limited", extra={"tool": tool_name})
                return {
                    "error": "Rate limit exceeded. Retry after a brief pause.",
                    "error_code": ErrorCode.RATE_LIMITED,
                }

            breaker = get_breaker(tool_name)
            if not breaker.can_execute():
                duration_ms = int((time.monotonic() - start) * 1000)
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.duration_ms", duration_ms)
                logger.warning("circuit_open", extra={"tool": tool_name})
                return {
                    "error": f"Circuit breaker open for '{tool_name}' — K8s API may be degraded. Retries automatically after recovery timeout.",
                    "error_code": ErrorCode.CIRCUIT_OPEN,
                }

            masked = mask_sensitive_data(kwargs) if kwargs else {}
            try:
                result = tool_func(**kwargs)
                duration_ms = int((time.monotonic() - start) * 1000)
                is_success = (
                    "error_code" not in result and "error" not in result
                    if isinstance(result, dict)
                    else True
                )
                span.set_attribute("tool.success", is_success)
                span.set_attribute("tool.duration_ms", duration_ms)
                if is_success:
                    breaker.record_success()
                elif is_infrastructure_error(result):
                    breaker.record_failure()

                logger.info(
                    "tool_call",
                    extra={
                        "audit": True,
                        "correlation_id": cid,
                        "tool": tool_name,
                        "parameters": masked,
                        "success": is_success,
                        "duration_ms": duration_ms,
                    },
                )
                return _inject_meta(result, tool_name)
            except Exception as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                breaker.record_failure()
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.duration_ms", duration_ms)
                if _StatusCode is not None:
                    span.set_status(_Status(_StatusCode.ERROR, str(exc)))
                logger.error(
                    "tool_call_failed",
                    extra={
                        "audit": True,
                        "correlation_id": cid,
                        "tool": tool_name,
                        "parameters": masked,
                        "success": False,
                        "duration_ms": duration_ms,
                    },
                    exc_info=True,
                )
                raise

    return wrapper


CLIENT_MODULES = {
    "trainer": "kubeflow_mcp.trainer",
    "optimizer": "kubeflow_mcp.optimizer",
    "hub": "kubeflow_mcp.hub",
}

_GLOBAL_HEADER = """\
Kubeflow MCP Server - AI Model Training on Kubernetes

PREREQUISITES:
- Kubeflow Trainer v2.2.0+ installed (TrainJob CRD must exist)
- Kubeflow SDK 0.4.0+ (bundled with MCP server)
- Kubernetes 1.27+
- Any platform: vanilla K8s, Kind, Minikube, OpenShift, EKS, GKE

CRITICAL WORKFLOW - Follow these steps IN ORDER:
1. PLANNING -> 2. DISCOVERY -> 3. TRAINING -> 4. MONITORING

IMPORTANT:
- ALWAYS preview before submitting (confirmed=False first)
- Use get_training_events() to debug stuck/failed jobs
"""


def _derive_tier(full_text: str, tier: str) -> str:
    """Derive compact/minimal instruction content from full tier."""
    if tier == "full":
        return full_text
    if tier == "compact":
        return "\n".join(
            line for line in full_text.splitlines() if line.strip() and "trainer://" not in line
        )
    tools = re.findall(r"\b(\w+)\(", full_text)
    return f"Tools: {', '.join(dict.fromkeys(tools))}"


def _sections_for_persona(persona: str) -> list[str]:
    """Derive which instruction sections a persona needs from its tool access."""
    allowed = get_allowed_tools(persona) or set(TOOL_TO_PHASE.keys())
    phases_used = {TOOL_TO_PHASE[t] for t in allowed if t in TOOL_TO_PHASE}

    section_order = ["planning", "monitoring", "training", "platform"]

    sections_needed: set[str] = set()
    for module_path in CLIENT_MODULES.values():
        try:
            module = importlib.import_module(module_path)
            phase_to_section = getattr(module, "PHASE_TO_SECTION", {})
            for phase in phases_used:
                section = phase_to_section.get(phase)
                if section:
                    sections_needed.add(section)
        except ImportError:
            continue

    return [s for s in section_order if s in sections_needed]


def _build_server_instructions(
    loaded_modules: dict[str, Any],
    persona: str,
    tier: str,
) -> str:
    """Build server instructions from client module sections, filtered by persona and tier."""
    sections = _sections_for_persona(persona)

    fragments = []
    for module in loaded_modules.values():
        instruction_sections = getattr(module, "INSTRUCTION_SECTIONS", {})
        for section_name in sections:
            section = instruction_sections.get(section_name, {})
            full_text = section.get("full", "")
            if full_text:
                fragments.append(_derive_tier(full_text, tier))

    resource_lines = []
    for module in loaded_modules.values():
        for uri, (_, desc) in getattr(module, "CLIENT_RESOURCES", {}).items():
            resource_lines.append(f"- {uri} -> {desc}")

    result = _GLOBAL_HEADER
    if fragments:
        result += "\n" + "\n\n".join(fragments) + "\n"
    if resource_lines and tier == "full":
        result += "\nRESOURCES (read on demand):\n" + "\n".join(resource_lines) + "\n"

    return result


def create_server(  # noqa: C901
    clients: list[str] | None = None,
    persona: str = "readonly",
    mode: str = "full",
    instruction_tier: str = "full",
    auth_provider: Any = None,
) -> FastMCP:
    """Create MCP server with dynamic client loading.

    Args:
        clients: List of client modules to load (default: ["trainer"])
                 Options: "trainer", "optimizer", "hub"
        persona: User role for tool filtering
        mode: Tool loading mode:
              - "full": register all tools directly (default)
              - "progressive": 3 meta-tools (list_tools, describe_tools, execute_tool)
              - "semantic": 2 meta-tools (find_tools, execute_tool)
        instruction_tier: Instruction verbosity — "full", "compact", or "minimal"
        auth_provider: Optional FastMCP auth provider for HTTP transport.

    Returns:
        Configured FastMCP server instance
    """
    if clients is None:
        clients = ["trainer"]

    from kubeflow_mcp.core.policy import set_effective_persona

    set_effective_persona(persona)

    # Single import per client — cache module refs for reuse
    loaded_modules: dict[str, Any] = {}
    for client_name in clients:
        if client_name not in CLIENT_MODULES:
            continue
        try:
            loaded_modules[client_name] = importlib.import_module(CLIENT_MODULES[client_name])
        except ImportError as e:
            logger.warning(f"Failed to import client '{client_name}': {e}")

    # Build instructions (persona + tier aware)
    instructions = _build_server_instructions(loaded_modules, persona, instruction_tier)
    mcp_kwargs: dict[str, Any] = {"instructions": instructions}
    if auth_provider is not None:
        mcp_kwargs["auth"] = auth_provider
        logger.info("HTTP auth provider attached to server")
    mcp: FastMCP = FastMCP("kubeflow-mcp", **mcp_kwargs)

    # Merge tool metadata from client modules
    all_descriptions: dict[str, str] = {}
    all_annotations: dict[str, dict] = {}
    for module in loaded_modules.values():
        all_descriptions.update(getattr(module, "CLIENT_TOOL_DESCRIPTIONS", {}))
        all_annotations.update(getattr(module, "CLIENT_TOOL_ANNOTATIONS", {}))
    all_descriptions.update(HEALTH_TOOL_DESCRIPTIONS)
    all_annotations.update(HEALTH_TOOL_ANNOTATIONS)

    # Stage 1: Get tools allowed by persona
    allowed_tools = get_allowed_tools(persona)

    # Collect tools from client modules plus server-level health tools
    # (core.health).
    tools_by_name: dict[str, Any] = {}
    for module in loaded_modules.values():
        for tool_func in getattr(module, "TOOLS", []):
            tools_by_name[tool_func.__name__] = tool_func
    for tool_func in HEALTH_TOOLS:
        tools_by_name.setdefault(tool_func.__name__, tool_func)
    all_tool_names = set(tools_by_name.keys())
    all_tool_funcs = list(tools_by_name.values())

    # Stage 2: Apply policy filters (allow/deny lists from ~/.kf-mcp-policy.yaml)
    if allowed_tools is not None:
        final_allowed = allowed_tools
    else:
        final_allowed = all_tool_names

    final_allowed = apply_policy_filters(final_allowed)

    # Stage 3: Enforce read_only policy by removing write tools.
    # Fail-closed: tools without an explicit readOnlyHint=True annotation
    # are treated as write tools and excluded. New tools MUST have
    # annotations in CLIENT_TOOL_ANNOTATIONS to be available in read_only mode.
    read_only = is_read_only()
    if read_only:
        read_tools = {
            name for name, ann in all_annotations.items() if ann.get("readOnlyHint", False)
        }
        write_tools = final_allowed - read_tools
        final_allowed = final_allowed & read_tools
        logger.info(f"read_only policy active, excluded {len(write_tools)} write tools")

    # Filter tool funcs to only those allowed by persona + policy
    allowed_funcs = [f for f in all_tool_funcs if f.__name__ in final_allowed]

    logger.debug(f"Final allowed tools after policy: {len(final_allowed)}")

    # --- Mode dispatch ---
    valid_modes = ("full", "progressive", "semantic")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}")

    if mode in ("progressive", "semantic"):
        init_dynamic_tools(allowed_funcs, all_descriptions)
        meta_tools = get_mode_tools(mode)

        for meta_func in meta_tools:
            audited = _audit_wrap(meta_func)
            mcp.tool()(audited)
            logger.debug(f"Registered meta-tool: {meta_func.__name__}")

        logger.info(
            f"Mode '{mode}': {len(meta_tools)} meta-tools proxying {len(allowed_funcs)} tools"
        )
    else:
        registered = 0
        for tool_func in allowed_funcs:
            tool_name = tool_func.__name__
            annotations = all_annotations.get(tool_name)
            description = all_descriptions.get(tool_name)
            audited = _audit_wrap(tool_func)

            if annotations and description:
                mcp.tool(description=description, annotations=annotations)(audited)
            elif description:
                mcp.tool(description=description)(audited)
            elif annotations:
                mcp.tool(annotations=annotations)(audited)
            else:
                mcp.tool()(audited)
            registered += 1
            logger.debug(f"Registered tool: {tool_name}")

        logger.info(f"Full mode: registered {registered} tools")

    # Register MCP resources from client modules (all resources, always)
    register_resources(mcp, loaded_modules)

    return mcp
