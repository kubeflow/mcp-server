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

"""Persona and policy definitions for tool access control.

Built-in personas define which tools are available to different user roles:
- readonly: View-only access for monitoring
- data-scientist: Training job submission
- ml-engineer: Full training management
- platform-admin: Unrestricted access

Custom policies can be defined in ~/.kf-mcp-policy.yaml:

.. code-block:: yaml

        policy:
            allow:
                - category:discovery
                - category:monitoring
                - category:planning
                - fine_tune
            deny:
                - risk:destructive
                - delete_*
            namespaces:
                - ml-team-dev
                - ml-team-prod
            read_only: false

        # Custom persona definitions
        personas:
            my-custom-role:
                inherit: readonly
                tools:
                    - fine_tune
                    - estimate_resources
"""

import fnmatch
import functools
import logging
import threading
from pathlib import Path
from typing import Any

from kubeflow_mcp.common.constants import TOOL_PHASES

logger = logging.getLogger(__name__)

_reload_lock = threading.Lock()


def _get_policy_paths() -> list[Path]:
    """Policy file locations, searched in order (home directory only for security)."""
    return [
        Path.home() / ".kf-mcp-policy.yaml",
        Path.home() / ".kf-mcp-policy.yml",
        Path.home() / ".config" / "kubeflow-mcp" / "policy.yaml",
    ]


PERSONAS: dict[str, dict[str, Any]] = {
    "readonly": {
        "tools": [
            "pre_flight",
            "check_compatibility",
            "get_cluster_resources",
            "estimate_resources",
            "list_training_jobs",
            "get_training_job",
            "list_runtimes",
            "get_runtime",
            "get_training_logs",
            "get_training_events",
            "health_check",
            "get_server_logs",
        ]
    },
    "data-scientist": {
        "inherit": "readonly",
        "tools": [
            "fine_tune",
            "run_custom_training",
            "wait_for_training",
            "delete_training_job",
        ],
    },
    "ml-engineer": {
        "inherit": "data-scientist",
        "tools": [
            "run_container_training",
            "update_training_job",
            "inspect_crd",
            "inspect_controller",
        ],
    },
    "platform-admin": {"tools": "*"},
}

DESTRUCTIVE_TOOLS = {"delete_training_job", "delete_runtime"}

# ─── Runtime persona ───────────────────────
# Set once at server startup by create_server(); tools read via get_effective_persona().

_effective_persona: str = "readonly"


def set_effective_persona(persona: str) -> None:
    """Store the resolved persona at server startup."""
    global _effective_persona
    _effective_persona = persona


def get_effective_persona() -> str:
    """Return the persona resolved at server startup."""
    return _effective_persona


def _find_policy_file() -> Path | None:
    """Find the first existing policy file."""
    for path in _get_policy_paths():
        if path.exists():
            return path
    return None


def _load_policy_file() -> dict[str, Any]:
    """Load policy from YAML file."""
    path = _find_policy_file()
    if not path:
        return {}

    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
            logger.debug(f"Loaded policy from {path}")
            return data if data else {}
    except ImportError:
        logger.debug("PyYAML not installed, skipping policy file")
        return {}
    except Exception as e:
        logger.warning(f"Failed to load policy from {path}: {e}")
        return {}


@functools.lru_cache(maxsize=1)
def _get_cached_policy() -> dict[str, Any]:
    """Get full policy file contents (cached, thread-safe via lru_cache)."""
    return _load_policy_file()


def _get_custom_personas_dict() -> dict[str, dict[str, Any]]:
    """Get custom personas from policy file (derived from cached policy)."""
    return _get_cached_policy().get("personas", {})


def _expand_category(category: str) -> list[str]:
    """Expand category:name to list of tools."""
    if category.startswith("category:"):
        cat_name = category[9:]
        return TOOL_PHASES.get(cat_name, [])
    return [category]


def _matches_pattern(tool: str, pattern: str) -> bool:
    """Check if tool matches a pattern (supports wildcards)."""
    if pattern.startswith("risk:"):
        risk = pattern[5:]
        if risk == "destructive":
            return tool in DESTRUCTIVE_TOOLS
        return False
    return fnmatch.fnmatch(tool, pattern)


def get_allowed_tools(persona: str, _seen: set[str] | None = None) -> set[str] | None:
    """Returns tool names allowed for persona. None means all.

    Args:
        persona: Persona name (built-in or custom)

    Returns:
        Set of allowed tool names, or None for unrestricted access

    Raises:
        ValueError: If persona is not found or inheritance cycle detected
    """
    if _seen is None:
        _seen = set()
    if persona in _seen:
        raise ValueError(f"Inheritance cycle detected: {' -> '.join(_seen)} -> {persona}")
    _seen.add(persona)

    custom = _get_custom_personas_dict()
    if persona in custom:
        config = custom[persona]
    elif persona in PERSONAS:
        config = PERSONAS[persona]
    else:
        raise ValueError(f"Unknown persona: {persona}")

    if config.get("tools") == "*":
        return None

    tools = set(config.get("tools", []))

    if "inherit" in config:
        parent = get_allowed_tools(config["inherit"], _seen)
        if parent:
            tools.update(parent)

    return tools


def apply_policy_filters(
    tools: set[str],
    policy: dict[str, Any] | None = None,
) -> set[str]:
    """Apply policy-based filtering to a set of tools.

    Args:
        tools: Set of tool names to filter
        policy: Policy dict with 'allow' and 'deny' lists

    Returns:
        Filtered set of tool names
    """
    if policy is None:
        policy = _get_cached_policy().get("policy", {})

    if not policy:
        return tools

    result = set(tools)

    allow = policy.get("allow", [])
    if allow:
        allowed = set()
        for pattern in allow:
            for item in _expand_category(pattern):
                for tool in tools:
                    if _matches_pattern(tool, item):
                        allowed.add(tool)
        result = result & allowed

    deny = policy.get("deny", [])
    for pattern in deny:
        for item in _expand_category(pattern):
            result = {t for t in result if not _matches_pattern(t, item)}

    return result


def get_allowed_namespaces() -> list[str] | None:
    """Get allowed namespaces from policy file.

    Returns:
        List of allowed namespaces, or None for unrestricted
    """
    policy = _get_cached_policy().get("policy", {})
    namespaces = policy.get("namespaces")
    if namespaces is None:
        return None
    return list(namespaces)


def is_read_only() -> bool:
    """Check if policy enforces read-only mode."""
    policy = _get_cached_policy().get("policy", {})
    return bool(policy.get("read_only", False))


def reload_policy() -> None:
    """Force reload of policy file (clears cache, thread-safe)."""
    with _reload_lock:
        _get_cached_policy.cache_clear()
