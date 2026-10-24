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
import reprlib
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

# Tools matched by each ``risk:<level>`` policy pattern.
RISK_LEVELS: dict[str, set[str]] = {"destructive": DESTRUCTIVE_TOOLS}

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


class PolicyFileError(RuntimeError):
    """The policy file exists but cannot be used, so the server must not start.

    A ``RuntimeError`` so existing callers keep working; ``serve()`` catches it to print
    the message without a traceback."""


def _load_policy_file() -> dict[str, Any]:
    """Load policy from YAML file.

    A missing file means no policy. A file that exists but cannot be loaded, or that
    loads with values of the wrong shape, raises instead of returning ``{}``: an empty
    policy is "no restrictions", so ignoring the file would silently drop ``read_only``,
    ``namespaces`` and ``deny`` (fail open).
    """
    path = _find_policy_file()
    if not path:
        return {}

    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
    except ImportError as e:
        raise PolicyFileError(f"Cannot enforce policy file {path}: PyYAML is not installed") from e
    except Exception as e:
        raise PolicyFileError(f"Failed to load policy file {path}: {e}") from e

    logger.debug(f"Loaded policy from {path}")
    return _validate_policy_shape(data, path)


def _shape_error(path: Path, key: str, expected: str, value: Any) -> PolicyFileError:
    """Build the error for a policy value of the wrong type."""
    return PolicyFileError(
        f"Invalid policy file {path}: '{key}' must be {expected}, "
        f"got {type(value).__name__} {reprlib.repr(value)}"
    )


def _mapping_or_empty(value: Any, key: str, path: Path) -> dict[str, Any]:
    """Return a mapping section; an empty YAML section (``None``) means an empty mapping."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _shape_error(path, key, "a mapping", value)
    return value


def _check_string_list(value: Any, key: str, path: Path) -> None:
    """Require a list of strings.

    A lone string is the dangerous case: the readers iterate it one character at a
    time, so ``deny: risk:destructive`` would deny nothing.
    """
    if isinstance(value, str):
        raise PolicyFileError(
            f"Invalid policy file {path}: '{key}' must be a list of strings, got the single "
            f"string {value!r}. Write it as a YAML list item: '- {value}'"
        )
    if not isinstance(value, list):
        raise _shape_error(path, key, "a list of strings", value)
    for item in value:
        if not isinstance(item, str):
            raise _shape_error(path, f"{key}[]", "a string", item)


def _validate_categories(patterns: list[str], key: str, path: Path) -> None:
    """Reject unknown ``category:`` references before a policy can be applied."""
    valid_categories = ", ".join(sorted(TOOL_PHASES))
    for pattern in patterns:
        if pattern.startswith("category:"):
            category = pattern.removeprefix("category:")
            if category not in TOOL_PHASES:
                raise PolicyFileError(
                    f"Invalid policy file {path}: unknown category {category!r} in '{key}'. "
                    f"Valid categories: {valid_categories}"
                )


def _validate_risks(patterns: list[str], key: str, path: Path) -> None:
    """Reject unknown ``risk:`` references, which would otherwise match no tool."""
    valid_risks = ", ".join(sorted(RISK_LEVELS))
    for pattern in patterns:
        if pattern.startswith("risk:"):
            risk = pattern.removeprefix("risk:")
            if risk not in RISK_LEVELS:
                raise PolicyFileError(
                    f"Invalid policy file {path}: unknown risk {risk!r} in '{key}'. "
                    f"Valid risks: {valid_risks}"
                )


def _validate_policy_section(policy: dict[str, Any], path: Path) -> None:
    """Check the ``policy:`` mapping in place."""
    for key in ("allow", "deny"):
        if key in policy:
            if policy[key] is None:
                policy[key] = []
            else:
                _check_string_list(policy[key], f"policy.{key}", path)
                _validate_categories(policy[key], f"policy.{key}", path)
                _validate_risks(policy[key], f"policy.{key}", path)

    # An empty ``namespaces:`` keeps meaning "no namespace restriction".
    if policy.get("namespaces") is not None:
        _check_string_list(policy["namespaces"], "policy.namespaces", path)

    read_only = policy.get("read_only")
    if read_only is not None and not isinstance(read_only, bool):
        # bool("false") is True, so a quoted value would silently flip the setting.
        raise _shape_error(path, "policy.read_only", "true or false", read_only)


def _validate_personas_section(personas: dict[str, Any], path: Path) -> None:
    """Check each custom persona in place."""
    for name, body in personas.items():
        key = f"personas.{name}"
        body = _mapping_or_empty(body, key, path)
        personas[name] = body

        if "inherit" in body and not isinstance(body["inherit"], str):
            raise _shape_error(path, f"{key}.inherit", "a persona name", body["inherit"])

        tools = body.get("tools")
        if tools is not None and tools != "*":
            _check_string_list(tools, f"{key}.tools", path)


def _validate_policy_shape(data: Any, path: Path) -> dict[str, Any]:
    """Check that a parsed policy file has the shape the readers below rely on.

    ``yaml.safe_load`` accepts any document, and a value of the wrong type is misread
    rather than rejected. Raise at load time instead, naming the file and the key, the
    same way an unparseable file fails closed. Empty sections are normalized so the
    readers never see ``None`` where they expect a mapping or a list.
    """
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise PolicyFileError(
            f"Invalid policy file {path}: the top level must be a mapping, "
            f"got {type(data).__name__} {reprlib.repr(data)}"
        )

    if "policy" in data:
        data["policy"] = _mapping_or_empty(data["policy"], "policy", path)
        _validate_policy_section(data["policy"], path)

    if "personas" in data:
        data["personas"] = _mapping_or_empty(data["personas"], "personas", path)
        _validate_personas_section(data["personas"], path)

    return data


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
        return tool in RISK_LEVELS.get(pattern[5:], set())
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

    # Fall back to empty list if tools is unspecified or explicitly null in YAML
    tools = set(config.get("tools") or [])

    if "inherit" in config:
        parent = get_allowed_tools(config["inherit"], _seen)
        if parent is None:
            return None
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
