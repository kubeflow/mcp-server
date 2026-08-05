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
    _expand_category,
    _matches_pattern,
    apply_policy_filters,
    get_allowed_namespaces,
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
    ],
)
def test_apply_policy_filters(test_case):
    result = apply_policy_filters(test_case.config["tools"], test_case.config["policy"])
    if test_case.expected_output is not None:
        assert result == test_case.expected_output
    else:
        assert "fine_tune" in result
        assert "delete_training_job" not in result


# TODO(test): test allow with category:planning
# TODO(test): test combined allow + deny
# TODO(test): test deny with category:training


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


# TODO(test): test reload_policy + get_allowed_namespaces reflects new file
# TODO(test): test custom persona from YAML with inherit chain
# TODO(test): test inheritance cycle detection raises ValueError
# TODO(test): test malformed policy YAML handled gracefully
