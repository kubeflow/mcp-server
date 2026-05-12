"""Tests for persona policy and access control."""

from pathlib import Path
from unittest.mock import patch

import pytest

from kubeflow_mcp.core.policy import (
    _expand_category,
    _matches_pattern,
    apply_policy_filters,
    get_allowed_namespaces,
    get_allowed_tools,
    is_read_only,
    reload_policy,
)


class TestBuiltinPersonas:
    def test_readonly(self):
        tools = get_allowed_tools("readonly")
        assert "list_training_jobs" in tools
        assert "fine_tune" not in tools

    def test_data_scientist_inherits_readonly(self):
        tools = get_allowed_tools("data-scientist")
        assert "list_training_jobs" in tools
        assert "fine_tune" in tools
        assert "run_container_training" not in tools

    def test_ml_engineer_inherits_data_scientist(self):
        tools = get_allowed_tools("ml-engineer")
        assert "list_training_jobs" in tools
        assert "fine_tune" in tools
        assert "run_container_training" in tools

    def test_platform_admin_unrestricted(self):
        assert get_allowed_tools("platform-admin") is None

    def test_unknown_persona_raises(self):
        with pytest.raises(ValueError, match="Unknown persona"):
            get_allowed_tools("unknown")


class TestCategoryExpansion:
    def test_known_category(self):
        tools = _expand_category("category:training")
        assert "fine_tune" in tools
        assert "run_custom_training" in tools

    def test_unknown_category(self):
        assert _expand_category("category:nonexistent") == []

    def test_non_category_passthrough(self):
        assert _expand_category("fine_tune") == ["fine_tune"]


class TestPatternMatching:
    def test_exact_match(self):
        assert _matches_pattern("fine_tune", "fine_tune") is True

    def test_wildcard(self):
        assert _matches_pattern("delete_training_job", "delete_*") is True
        assert _matches_pattern("fine_tune", "delete_*") is False

    def test_risk_destructive(self):
        assert _matches_pattern("delete_training_job", "risk:destructive") is True
        assert _matches_pattern("fine_tune", "risk:destructive") is False

    def test_unknown_risk_tag(self):
        assert _matches_pattern("fine_tune", "risk:unknown") is False


class TestApplyPolicyFilters:
    def test_allow_list(self):
        tools = {"fine_tune", "list_training_jobs", "delete_training_job"}
        result = apply_policy_filters(tools, policy={"allow": ["fine_tune", "list_training_jobs"]})
        assert result == {"fine_tune", "list_training_jobs"}

    def test_deny_list(self):
        tools = {"fine_tune", "list_training_jobs", "delete_training_job"}
        result = apply_policy_filters(tools, policy={"deny": ["risk:destructive"]})
        assert "delete_training_job" not in result
        assert "fine_tune" in result

    def test_deny_wildcard(self):
        tools = {"delete_training_job", "fine_tune", "list_training_jobs"}
        result = apply_policy_filters(tools, policy={"deny": ["delete_*"]})
        assert "delete_training_job" not in result

    def test_allow_by_category(self):
        tools = {"fine_tune", "run_custom_training", "list_training_jobs"}
        result = apply_policy_filters(tools, policy={"allow": ["category:training"]})
        assert "fine_tune" in result
        assert "run_custom_training" in result
        assert "list_training_jobs" not in result

    def test_empty_policy_returns_all(self):
        tools = {"fine_tune", "list_training_jobs"}
        assert apply_policy_filters(tools, policy={}) == tools


class TestNamespacesAndReadOnly:
    def test_no_policy_namespaces_returns_none(self):
        with patch("kubeflow_mcp.core.policy._load_policy_file", return_value={}):
            reload_policy()
            assert get_allowed_namespaces() is None
        reload_policy()

    def test_namespaces_from_policy(self):
        policy = {"policy": {"namespaces": ["ml-dev", "ml-prod"]}}
        with patch("kubeflow_mcp.core.policy._load_policy_file", return_value=policy):
            reload_policy()
            assert get_allowed_namespaces() == ["ml-dev", "ml-prod"]
        reload_policy()

    def test_read_only_false_by_default(self):
        with patch("kubeflow_mcp.core.policy._load_policy_file", return_value={}):
            reload_policy()
            assert is_read_only() is False
        reload_policy()

    def test_read_only_true(self):
        policy = {"policy": {"read_only": True}}
        with patch("kubeflow_mcp.core.policy._load_policy_file", return_value=policy):
            reload_policy()
            assert is_read_only() is True
        reload_policy()


class TestCustomPersonas:
    def test_custom_persona_from_policy_file(self):
        policy = {"personas": {"custom-role": {"tools": ["fine_tune", "list_training_jobs"]}}}
        with patch("kubeflow_mcp.core.policy._load_policy_file", return_value=policy):
            reload_policy()
            tools = get_allowed_tools("custom-role")
        assert "fine_tune" in tools
        assert "list_training_jobs" in tools
        reload_policy()

    def test_custom_persona_with_inheritance(self):
        policy = {
            "personas": {"extended-readonly": {"inherit": "readonly", "tools": ["fine_tune"]}}
        }
        with patch("kubeflow_mcp.core.policy._load_policy_file", return_value=policy):
            reload_policy()
            tools = get_allowed_tools("extended-readonly")
        assert "fine_tune" in tools
        assert "list_training_jobs" in tools  # inherited from readonly
        reload_policy()


class TestPolicyFileLoading:
    def test_load_from_yaml(self, tmp_path):
        policy_file = tmp_path / ".kf-mcp-policy.yaml"
        policy_file.write_text(
            "policy:\n"
            "  allow:\n"
            "    - fine_tune\n"
            "    - list_training_jobs\n"
            "  namespaces:\n"
            "    - ml-dev\n"
        )
        with patch("kubeflow_mcp.core.policy._get_policy_paths", return_value=[policy_file]):
            reload_policy()
            assert get_allowed_namespaces() == ["ml-dev"]
        reload_policy()

    def test_missing_file_returns_none(self):
        with patch(
            "kubeflow_mcp.core.policy._get_policy_paths",
            return_value=[Path("/nonexistent/policy.yaml")],
        ):
            reload_policy()
            assert get_allowed_namespaces() is None
        reload_policy()


class TestReloadPolicy:
    def test_picks_up_new_personas(self):
        """reload_policy() must clear cache so a subsequent call re-reads the file."""
        policy_v1 = {"personas": {"role-v1": {"tools": ["list_training_jobs"]}}}
        policy_v2 = {"personas": {"role-v2": {"tools": ["fine_tune"]}}}

        with patch("kubeflow_mcp.core.policy._load_policy_file", return_value=policy_v1):
            reload_policy()
            assert "list_training_jobs" in get_allowed_tools("role-v1")

        with patch("kubeflow_mcp.core.policy._load_policy_file", return_value=policy_v2):
            reload_policy()
            with pytest.raises(ValueError, match="Unknown persona"):
                get_allowed_tools("role-v1")
            assert "fine_tune" in get_allowed_tools("role-v2")

        reload_policy()  # leave clean

    def test_without_personas_clears_stale_cache(self):
        """reload_policy() on a file with no personas must not leave stale cache."""
        with patch(
            "kubeflow_mcp.core.policy._load_policy_file",
            return_value={"personas": {"temp-role": {"tools": ["fine_tune"]}}},
        ):
            reload_policy()
            get_allowed_tools("temp-role")  # warms cache

        with patch("kubeflow_mcp.core.policy._load_policy_file", return_value={}):
            reload_policy()
            with pytest.raises(ValueError, match="Unknown persona"):
                get_allowed_tools("temp-role")

        reload_policy()
