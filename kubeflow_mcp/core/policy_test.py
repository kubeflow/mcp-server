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

"""Tests for core/policy.py — persona gating, policy filters, namespace rules."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from tests.common import TestCase

from kubeflow_mcp.core.policy import (
    PolicyFileError,
    _expand_category,
    _load_policy_file,
    _matches_pattern,
    apply_policy_filters,
    get_allowed_namespaces,
    get_allowed_tools,
    get_effective_persona,
    is_read_only,
    reload_policy,
    set_effective_persona,
)

# ─── Effective persona ──────────────────────────────────────────────────────


def test_default_persona_is_readonly():
    set_effective_persona("readonly")
    assert get_effective_persona() == "readonly"


def test_set_and_get_persona():
    set_effective_persona("platform-admin")
    assert get_effective_persona() == "platform-admin"
    set_effective_persona("readonly")


# ─── expand / matches helpers ────────────────────────────────────────────────


def test_category_prefix_expands():
    result = _expand_category("category:planning")
    assert isinstance(result, list)
    assert len(result) > 0


def test_non_category_returns_as_is():
    assert _expand_category("fine_tune") == ["fine_tune"]


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="exact match",
            config={"tool": "fine_tune", "pattern": "fine_tune"},
            expected_output=True,
        ),
        TestCase(
            name="wildcard match",
            config={"tool": "delete_training_job", "pattern": "delete_*"},
            expected_output=True,
        ),
        TestCase(
            name="risk:destructive matches delete",
            config={"tool": "delete_training_job", "pattern": "risk:destructive"},
            expected_output=True,
        ),
        TestCase(
            name="risk:destructive does not match fine_tune",
            config={"tool": "fine_tune", "pattern": "risk:destructive"},
            expected_output=False,
        ),
    ],
)
def test_matches_pattern(test_case):
    result = _matches_pattern(test_case.config["tool"], test_case.config["pattern"])
    assert result is test_case.expected_output


# ─── apply_policy_filters ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="empty policy returns all tools",
            config={
                "tools": {"fine_tune", "list_training_jobs"},
                "policy": {},
            },
            expected_output={"fine_tune", "list_training_jobs"},
        ),
        TestCase(
            name="allow restricts to specified tools",
            config={
                "tools": {"fine_tune", "list_training_jobs", "delete_training_job"},
                "policy": {"allow": ["fine_tune", "list_training_jobs"]},
            },
            expected_output={"fine_tune", "list_training_jobs"},
        ),
        TestCase(
            name="deny removes matching tools",
            config={
                "tools": {"fine_tune", "delete_training_job"},
                "policy": {"deny": ["risk:destructive"]},
            },
        ),
        TestCase(
            name="deny wildcard removes matching tools",
            config={
                "tools": {"delete_training_job", "delete_runtime", "fine_tune"},
                "policy": {"deny": ["delete_*"]},
            },
            expected_output={"fine_tune"},
        ),
        TestCase(
            name="allow with category:planning keeps only planning tools",
            config={
                "tools": {
                    "pre_flight",
                    "check_compatibility",
                    "get_cluster_resources",
                    "estimate_resources",
                    "fine_tune",
                    "delete_training_job",
                },
                "policy": {"allow": ["category:planning"]},
            },
            expected_output={
                "pre_flight",
                "check_compatibility",
                "get_cluster_resources",
                "estimate_resources",
            },
        ),
        TestCase(
            name="combined allow and deny applies deny after allow",
            config={
                "tools": {
                    "fine_tune",
                    "run_custom_training",
                    "run_container_training",
                    "delete_training_job",
                    "list_training_jobs",
                },
                "policy": {"allow": ["category:training"], "deny": ["fine_tune"]},
            },
            expected_output={"run_custom_training", "run_container_training"},
        ),
        TestCase(
            name="deny with category:training removes all training tools",
            config={
                "tools": {
                    "fine_tune",
                    "run_custom_training",
                    "run_container_training",
                    "get_training_logs",
                    "delete_training_job",
                },
                "policy": {"deny": ["category:training"]},
            },
            expected_output={"get_training_logs", "delete_training_job"},
        ),
    ],
)
def test_apply_policy_filters(test_case):
    result = apply_policy_filters(test_case.config["tools"], test_case.config["policy"])
    if test_case.expected_output is not None:
        assert result == test_case.expected_output
    else:
        assert "fine_tune" in result
        assert "delete_training_job" not in result


# ─── Namespace policy ────────────────────────────────────────────────────────


def test_no_policy_returns_none():
    with patch(
        "kubeflow_mcp.core.policy._get_cached_policy",
        return_value={},
    ):
        assert get_allowed_namespaces() is None


def test_returns_namespace_list():
    with patch(
        "kubeflow_mcp.core.policy._get_cached_policy",
        return_value={"policy": {"namespaces": ["ns-a", "ns-b"]}},
    ):
        assert get_allowed_namespaces() == ["ns-a", "ns-b"]


def test_read_only_default_false():
    with patch(
        "kubeflow_mcp.core.policy._get_cached_policy",
        return_value={},
    ):
        assert is_read_only() is False


def test_read_only_true_when_set():
    with patch(
        "kubeflow_mcp.core.policy._get_cached_policy",
        return_value={"policy": {"read_only": True}},
    ):
        assert is_read_only() is True


def test_reload_policy_clears_cache():
    with patch("kubeflow_mcp.core.policy._load_policy_file") as load_policy:
        load_policy.return_value = {"policy": {"namespaces": ["old-ns"]}}
        assert get_allowed_namespaces() == ["old-ns"]

        load_policy.return_value = {"policy": {"namespaces": ["new-ns"]}}
        reload_policy()
        assert get_allowed_namespaces() == ["new-ns"]


# ─── Policy file loading ─────────────────────────────────────────────────────


def test_missing_policy_file_means_no_policy(tmp_policy_file):
    assert _load_policy_file() == {}


def test_valid_policy_file_is_loaded(tmp_policy_file):
    policy = {"policy": {"read_only": True, "namespaces": ["team-a"]}}
    tmp_policy_file(policy)
    assert _load_policy_file() == policy


def test_malformed_policy_file_fails_closed(tmp_policy_file):
    """A typo must not silently drop read_only, namespaces and deny (fail open)."""
    policy_path = tmp_policy_file({})
    policy_path.write_text('policy:\n  read_only: true\n  deny: ["delete_*"\n')

    with pytest.raises(RuntimeError, match="Failed to load policy file"):
        is_read_only()
    with pytest.raises(RuntimeError, match="Failed to load policy file"):
        get_allowed_namespaces()


def test_policy_file_without_pyyaml_fails_closed(tmp_policy_file):
    tmp_policy_file({"policy": {"read_only": True}})

    with patch.dict("sys.modules", {"yaml": None}):
        with pytest.raises(RuntimeError, match="PyYAML is not installed"):
            _load_policy_file()


# ─── Policy file shape ───────────────────────────────────────────────────────


@pytest.fixture
def write_policy_yaml(tmp_policy_file):
    """Write raw YAML so tests use the exact text an admin would type."""

    def _write(text: str):
        policy_path = tmp_policy_file({})
        policy_path.write_text(text)
        return policy_path

    return _write


@pytest.mark.parametrize(
    "yaml_text, message",
    [
        pytest.param(
            "policy:\n  deny: risk:destructive\n",
            r"'policy\.deny' must be a list of strings, got the single string "
            r"'risk:destructive'\. Write it as a YAML list item: '- risk:destructive'",
            id="deny-single-string-denied-nothing",
        ),
        pytest.param(
            "policy:\n  allow: list_*\n",
            r"'policy\.allow' must be a list of strings, got the single string",
            id="allow-single-string-allowed-everything",
        ),
        pytest.param(
            "policy:\n  namespaces: ml-team-dev\n",
            r"'policy\.namespaces' must be a list of strings, got the single string",
            id="namespaces-single-string-split-into-characters",
        ),
        pytest.param(
            "policy:\n  deny:\n    - delete_*\n    - 42\n",
            r"'policy\.deny\[\]' must be a string, got int 42",
            id="deny-non-string-item",
        ),
        pytest.param(
            "policy:\n  deny:\n    risk: destructive\n",
            r"'policy\.deny' must be a list of strings, got dict",
            id="deny-mapping",
        ),
        pytest.param(
            'policy:\n  read_only: "false"\n',
            r"'policy\.read_only' must be true or false, got str 'false'",
            id="read-only-quoted-string-read-as-true",
        ),
        pytest.param(
            "policy:\n  - read_only\n",
            r"'policy' must be a mapping, got list",
            id="policy-section-list",
        ),
        pytest.param(
            "personas: auditor\n",
            r"'personas' must be a mapping, got str",
            id="personas-section-string",
        ),
        pytest.param(
            "personas:\n  auditor: readonly\n",
            r"'personas\.auditor' must be a mapping, got str",
            id="persona-body-string",
        ),
        pytest.param(
            "personas:\n  auditor:\n    tools: fine_tune\n",
            r"'personas\.auditor\.tools' must be a list of strings, got the single string",
            id="persona-tools-single-string",
        ),
        pytest.param(
            "personas:\n  auditor:\n    inherit: [readonly]\n",
            r"'personas\.auditor\.inherit' must be a persona name, got list",
            id="persona-inherit-list-crashed-cycle-check",
        ),
        pytest.param(
            "personas:\n  auditor:\n    inherit:\n",
            r"'personas\.auditor\.inherit' must be a persona name, got NoneType",
            id="persona-inherit-empty",
        ),
        pytest.param(
            "- deny\n",
            r"the top level must be a mapping, got list",
            id="top-level-list",
        ),
        pytest.param(
            "read_only\n",
            r"the top level must be a mapping, got str",
            id="top-level-string",
        ),
    ],
)
def test_wrong_policy_shape_fails_closed(write_policy_yaml, yaml_text, message):
    """A file that parses but has the wrong shape must not be applied (issue #405)."""
    policy_path = write_policy_yaml(yaml_text)

    with pytest.raises(RuntimeError, match=message) as exc_info:
        _load_policy_file()
    assert str(policy_path) in str(exc_info.value)


@pytest.mark.parametrize("reader", [is_read_only, get_allowed_namespaces, apply_policy_filters])
def test_wrong_policy_shape_blocks_every_reader(write_policy_yaml, reader):
    """No policy reader may fall back to an unrestricted value for a bad file."""
    write_policy_yaml("policy:\n  deny: risk:destructive\n")
    args = ({"delete_training_job"},) if reader is apply_policy_filters else ()

    with pytest.raises(RuntimeError, match="Invalid policy file"):
        reader(*args)


def test_known_policy_categories_are_accepted(write_policy_yaml):
    write_policy_yaml("policy:\n  allow: [category:planning]\n  deny: [category:monitoring]\n")

    assert _load_policy_file() == {
        "policy": {"allow": ["category:planning"], "deny": ["category:monitoring"]}
    }


@pytest.mark.parametrize("key", ["allow", "deny"])
def test_unknown_policy_category_fails_closed(write_policy_yaml, key):
    policy_path = write_policy_yaml(f"policy:\n  {key}: [category:trainig]\n")

    with pytest.raises(
        PolicyFileError,
        match=r"unknown category 'trainig'.*Valid categories:.*training",
    ) as exc_info:
        _load_policy_file()
    assert str(policy_path) in str(exc_info.value)


def test_known_policy_risks_are_accepted(write_policy_yaml):
    write_policy_yaml("policy:\n  allow: [risk:destructive]\n  deny: [risk:destructive]\n")

    assert _load_policy_file() == {
        "policy": {"allow": ["risk:destructive"], "deny": ["risk:destructive"]}
    }


@pytest.mark.parametrize("key", ["allow", "deny"])
@pytest.mark.parametrize("risk", ["Destructive", "mutating", ""])
def test_unknown_policy_risk_fails_closed(write_policy_yaml, key, risk):
    policy_path = write_policy_yaml(f"policy:\n  {key}: ['risk:{risk}']\n")

    with pytest.raises(
        PolicyFileError,
        match=rf"unknown risk '{risk}'.*Valid risks: destructive",
    ) as exc_info:
        _load_policy_file()
    assert str(policy_path) in str(exc_info.value)


def test_well_formed_policy_file_is_unchanged(write_policy_yaml):
    write_policy_yaml(
        "policy:\n"
        "  allow: [category:discovery, fine_tune]\n"
        "  deny: [risk:destructive]\n"
        "  namespaces: [ml-team-dev]\n"
        "  read_only: false\n"
        "personas:\n"
        "  auditor:\n"
        "    inherit: readonly\n"
        "    tools: [fine_tune]\n"
        "  admin-like:\n"
        "    tools: '*'\n"
    )

    assert _load_policy_file() == {
        "policy": {
            "allow": ["category:discovery", "fine_tune"],
            "deny": ["risk:destructive"],
            "namespaces": ["ml-team-dev"],
            "read_only": False,
        },
        "personas": {
            "auditor": {"inherit": "readonly", "tools": ["fine_tune"]},
            "admin-like": {"tools": "*"},
        },
    }
    assert get_allowed_namespaces() == ["ml-team-dev"]
    assert apply_policy_filters({"delete_training_job", "fine_tune"}) == {"fine_tune"}


@pytest.mark.parametrize(
    "yaml_text",
    [
        pytest.param("", id="empty-file"),
        pytest.param("policy:\n", id="empty-policy"),
        pytest.param("policy:\n  deny:\n", id="empty-deny"),
        pytest.param("policy:\n  allow:\n", id="empty-allow"),
        pytest.param("policy:\n  namespaces:\n  read_only:\n", id="empty-namespaces-read-only"),
        pytest.param("personas:\n", id="empty-personas"),
    ],
)
def test_empty_sections_mean_no_restriction(write_policy_yaml, yaml_text):
    """An empty section used to crash startup; it now means the same as leaving it out."""
    write_policy_yaml(yaml_text)
    tools = {"delete_training_job", "fine_tune"}

    assert is_read_only() is False
    assert get_allowed_namespaces() is None
    assert apply_policy_filters(tools) == tools
    assert get_allowed_tools("readonly") is not None


def test_empty_persona_body_has_no_tools(write_policy_yaml):
    """A persona with nothing under it used to crash when selected."""
    write_policy_yaml("personas:\n  auditor:\n")

    assert get_allowed_tools("auditor") == set()


# ─── Persona inheritance & gating ───────────────────────────────────────────


def test_custom_persona_inherits_unrestricted_parent():
    """Inheriting from platform-admin or another unrestricted persona must return None."""
    with patch(
        "kubeflow_mcp.core.policy._get_custom_personas_dict",
        return_value={
            "super-admin": {"inherit": "platform-admin"},
            "audit-admin": {"inherit": "platform-admin", "tools": ["audit_tool"]},
            "wildcard-custom": {"tools": "*"},
            "child-of-wildcard": {"inherit": "wildcard-custom"},
        },
    ):
        assert get_allowed_tools("super-admin") is None
        assert get_allowed_tools("audit-admin") is None
        assert get_allowed_tools("wildcard-custom") is None
        assert get_allowed_tools("child-of-wildcard") is None


def test_custom_persona_null_tools_yaml_fallback():
    """tools: null in YAML policy file should not crash set() and should inherit parent tools."""
    with patch(
        "kubeflow_mcp.core.policy._get_custom_personas_dict",
        return_value={
            "null-tools-child": {"inherit": "readonly", "tools": None},
        },
    ):
        readonly_tools = get_allowed_tools("readonly")
        assert readonly_tools is not None
        assert get_allowed_tools("null-tools-child") == readonly_tools


def test_multi_level_inheritance_chain_unrestricted():
    """Multi-level chain ending in unrestricted persona must propagate None to all descendants."""
    with patch(
        "kubeflow_mcp.core.policy._get_custom_personas_dict",
        return_value={
            "tier-1": {"inherit": "platform-admin", "tools": ["tier1_tool"]},
            "tier-2": {"inherit": "tier-1", "tools": ["tier2_tool"]},
            "tier-3": {"inherit": "tier-2"},
        },
    ):
        assert get_allowed_tools("platform-admin") is None
        assert get_allowed_tools("tier-1") is None
        assert get_allowed_tools("tier-2") is None
        assert get_allowed_tools("tier-3") is None


def test_multi_level_inheritance_chain_restricted():
    """Multi-level chain with restricted personas must properly accumulate tools from all ancestors."""
    with patch(
        "kubeflow_mcp.core.policy._get_custom_personas_dict",
        return_value={
            "analyst-base": {"inherit": "readonly", "tools": ["run_query"]},
            "analyst-senior": {"inherit": "analyst-base", "tools": ["export_report"]},
            "analyst-lead": {"inherit": "analyst-senior", "tools": ["manage_dashboards"]},
        },
    ):
        readonly_tools = get_allowed_tools("readonly")
        lead_tools = get_allowed_tools("analyst-lead")

        assert readonly_tools is not None
        assert lead_tools is not None
        # Must contain all ancestor tools plus own tools
        assert readonly_tools.issubset(lead_tools)
        assert {"run_query", "export_report", "manage_dashboards"}.issubset(lead_tools)


def test_inheritance_cycle_detection_raises_value_error():
    """Inheritance cycles must raise ValueError with cycle path."""
    with patch(
        "kubeflow_mcp.core.policy._get_custom_personas_dict",
        return_value={
            "cycle-a": {"inherit": "cycle-b"},
            "cycle-b": {"inherit": "cycle-a"},
        },
    ):
        with pytest.raises(ValueError, match="Inheritance cycle detected"):
            get_allowed_tools("cycle-a")

    with patch(
        "kubeflow_mcp.core.policy._get_custom_personas_dict",
        return_value={
            "node-1": {"inherit": "node-2"},
            "node-2": {"inherit": "node-3"},
            "node-3": {"inherit": "node-1"},
        },
    ):
        with pytest.raises(ValueError, match="Inheritance cycle detected"):
            get_allowed_tools("node-1")
