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

"""Tests for core/config.py — config loading, env overrides, defaults."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml
from tests.common import TestCase

from kubeflow_mcp.core.config import (
    ServerConfig,
    _find_config_file,
    load_config,
)


def test_server_config_defaults():
    cfg = ServerConfig()
    assert cfg.persona == "readonly"
    assert cfg.transport == "stdio"
    assert cfg.instruction_tier == "full"
    assert cfg.clients == ["trainer"]


def test_server_config_custom_values():
    cfg = ServerConfig(persona="ml-engineer", transport="http", instruction_tier="compact")
    assert cfg.persona == "ml-engineer"
    assert cfg.transport == "http"
    assert cfg.instruction_tier == "compact"


def test_load_config_defaults_when_no_file():
    with patch("kubeflow_mcp.core.config._find_config_file", return_value=None):
        cfg = load_config()
    assert cfg.server.persona == "readonly"
    assert cfg.resilience.rate_limit == 10.0


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="KUBEFLOW_MCP_PERSONA overrides persona",
            config={"env": {"KUBEFLOW_MCP_PERSONA": "platform-admin"}},
            expected_output={"field": "server.persona", "value": "platform-admin"},
        ),
        TestCase(
            name="KUBEFLOW_MCP_INSTRUCTION_TIER overrides tier",
            config={"env": {"KUBEFLOW_MCP_INSTRUCTION_TIER": "minimal"}},
            expected_output={"field": "server.instruction_tier", "value": "minimal"},
        ),
        TestCase(
            name="KUBEFLOW_MCP_RATE_LIMIT overrides rate limit",
            config={"env": {"KUBEFLOW_MCP_RATE_LIMIT": "5.0"}},
            expected_output={"field": "resilience.rate_limit", "value": 5.0},
        ),
        TestCase(
            name="KUBEFLOW_MCP_CB_FAILURE_THRESHOLD overrides threshold",
            config={"env": {"KUBEFLOW_MCP_CB_FAILURE_THRESHOLD": "10"}},
            expected_output={"field": "resilience.cb_failure_threshold", "value": 10},
        ),
    ],
)
def test_load_config_env_overrides(test_case):
    with (
        patch("kubeflow_mcp.core.config._find_config_file", return_value=None),
        patch.dict("os.environ", test_case.config["env"]),
    ):
        cfg = load_config()
    section, field = test_case.expected_output["field"].split(".")
    assert getattr(getattr(cfg, section), field) == test_case.expected_output["value"]


def test_load_config_from_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.dump(
            {
                "server": {"persona": "data-scientist", "transport": "http"},
                "trainer": {"default_runtime": "torchtune-llama"},
            }
        )
    )
    cfg = load_config(config_path=config_file)
    assert cfg.server.persona == "data-scientist"
    assert cfg.trainer.default_runtime == "torchtune-llama"


def test_load_config_env_overrides_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"server": {"persona": "data-scientist"}}))
    with patch.dict("os.environ", {"KUBEFLOW_MCP_PERSONA": "platform-admin"}):
        cfg = load_config(config_path=config_file)
    assert cfg.server.persona == "platform-admin"


def test_load_config_ignores_non_mapping_yaml(tmp_path, caplog):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- trainer\n- optimizer\n")

    cfg = load_config(config_path=config_file)

    assert cfg.server.clients == ["trainer"]
    assert "Config root must be a mapping" in caplog.text


def test_find_config_file_prefers_project_local_config(tmp_path):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    (home_dir / ".kubeflow-mcp.yaml").write_text("server:\n  persona: readonly\n")
    local_config = tmp_path / ".kubeflow-mcp.yaml"
    local_config.write_text("server:\n  persona: ml-engineer\n")

    with (
        patch("kubeflow_mcp.core.config.Path.home", return_value=home_dir),
        patch("kubeflow_mcp.core.config.Path.cwd", return_value=tmp_path),
    ):
        assert _find_config_file() == local_config


# TODO(test): test config_path that doesn't exist falls back to default search
# TODO(test): test malformed YAML handled gracefully
# TODO(test): test auth config env overrides (KUBEFLOW_MCP_AUTH_TOKEN, KUBEFLOW_MCP_JWKS_URI)
# TODO(test): test OTEL_EXPORTER_OTLP_ENDPOINT env override
# TODO(test): test KUBEFLOW_MCP_CONTROLLER_NAMESPACE env override
# TODO(test): test multiple clients from KUBEFLOW_MCP_CLIENTS env var
