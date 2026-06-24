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
"""Test CLI commands."""

import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from kubeflow_mcp.cli import cli


def test_cli_version():
    from kubeflow_mcp import __version__

    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_status_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "trainer" in result.output
    assert "implemented" in result.output


def test_status_shows_stubs():
    runner = CliRunner()
    result = runner.invoke(cli, ["status"])
    assert "optimizer" in result.output
    assert "stub" in result.output


# --- serve: input validation ---


def test_serve_rejects_invalid_persona():
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--persona", "hacker"])
    assert result.exit_code == 2
    assert "invalid value" in result.output.lower()


def test_serve_rejects_invalid_transport():
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--transport", "websocket"])
    assert result.exit_code == 2
    assert "invalid value" in result.output.lower()


# --- serve: wiring — create_server receives correct args, server.run is called ---


def _make_default_config(**overrides):
    """Build a fake Config with sensible defaults, applying *overrides* to server fields."""
    from kubeflow_mcp.core.config import Config, LoggingConfig, ServerConfig

    server_kwargs = dict(
        clients=["trainer"],
        persona="readonly",
        transport="stdio",
    )
    server_kwargs.update(overrides)
    return Config(server=ServerConfig(**server_kwargs), logging=LoggingConfig())


def _make_serve_mocks(config=None):
    """Return (mock_server, mock_create_server, sys.modules patch dict) for serve tests.

    serve() does lazy imports of core.logging, core.server, core.config,
    core.auth, and core.resilience — patch sys.modules so those imports
    resolve to mocks regardless of whether the modules exist on the current branch.
    """
    if config is None:
        config = _make_default_config()

    mock_server = MagicMock()
    mock_create_server = MagicMock(return_value=mock_server)
    mock_configure_resilience = MagicMock()
    mock_setup_logging = MagicMock(return_value=MagicMock())
    mock_load_config = MagicMock(return_value=config)
    mock_build_auth_provider = MagicMock(return_value=None)
    mock_configure_circuit_breaker = MagicMock()

    fake_server_mod = MagicMock()
    fake_server_mod.create_server = mock_create_server
    fake_server_mod.configure_resilience = mock_configure_resilience
    fake_logging_mod = MagicMock()
    fake_logging_mod.setup_logging = mock_setup_logging
    fake_config_mod = MagicMock()
    fake_config_mod.load_config = mock_load_config
    fake_auth_mod = MagicMock()
    fake_auth_mod.build_auth_provider = mock_build_auth_provider
    fake_resilience_mod = MagicMock()
    fake_resilience_mod.configure_circuit_breaker = mock_configure_circuit_breaker

    modules_patch = {
        "kubeflow_mcp.core.server": fake_server_mod,
        "kubeflow_mcp.core.logging": fake_logging_mod,
        "kubeflow_mcp.core.config": fake_config_mod,
        "kubeflow_mcp.core.auth": fake_auth_mod,
        "kubeflow_mcp.core.resilience": fake_resilience_mod,
    }
    return mock_server, mock_create_server, modules_patch


def test_serve_passes_clients_and_persona():
    mock_server, mock_create_server, modules_patch = _make_serve_mocks()

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(
            cli,
            ["serve", "--clients", "trainer,optimizer", "--persona", "data-scientist"],
        )

    mock_create_server.assert_called_once_with(
        clients=["trainer", "optimizer"],
        persona="data-scientist",
        mode="full",
        instruction_tier="full",
        auth_provider=None,
    )
    mock_server.run.assert_called_once()


def test_serve_http_transport_uses_streamable_http():
    mock_server, _, modules_patch = _make_serve_mocks()

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve", "--transport", "http"])

    _, kwargs = mock_server.run.call_args
    assert kwargs.get("transport") == "streamable-http"


def test_serve_sse_transport_uses_sse():
    mock_server, _, modules_patch = _make_serve_mocks()

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve", "--transport", "sse"])

    _, kwargs = mock_server.run.call_args
    assert kwargs.get("transport") == "sse"


def test_serve_sse_transport_calls_build_auth_provider():
    mock_server, _, modules_patch = _make_serve_mocks()

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve", "--transport", "sse"])

    fake_auth_mod = sys.modules.get(
        "kubeflow_mcp.core.auth", modules_patch["kubeflow_mcp.core.auth"]
    )
    fake_auth_mod.build_auth_provider.assert_called_once()


def test_serve_progressive_mode():
    mock_server, mock_create_server, modules_patch = _make_serve_mocks()

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve", "--mode", "progressive"])

    mock_create_server.assert_called_once_with(
        clients=["trainer"],
        persona="readonly",
        mode="progressive",
        instruction_tier="full",
        auth_provider=None,
    )
    mock_server.run.assert_called_once()


def test_serve_semantic_mode():
    mock_server, mock_create_server, modules_patch = _make_serve_mocks()

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve", "-m", "semantic"])

    mock_create_server.assert_called_once_with(
        clients=["trainer"],
        persona="readonly",
        mode="semantic",
        instruction_tier="full",
        auth_provider=None,
    )


def test_serve_rejects_invalid_mode():
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--mode", "turbo"])
    assert result.exit_code == 2
    assert "invalid value" in result.output.lower()


# --- serve: config / env var fallback ---


@dataclass
class ConfigFallbackCase:
    name: str
    config_persona: str
    cli_args: list[str] = field(default_factory=lambda: ["serve"])
    expected_persona: str = ""

    def __post_init__(self):
        if not self.expected_persona:
            self.expected_persona = self.config_persona


@pytest.mark.parametrize(
    "test_case",
    [
        ConfigFallbackCase(
            name="no CLI flag falls back to config persona",
            config_persona="ml-engineer",
            cli_args=["serve"],
            expected_persona="ml-engineer",
        ),
        ConfigFallbackCase(
            name="CLI flag overrides config persona",
            config_persona="ml-engineer",
            cli_args=["serve", "--persona", "platform-admin"],
            expected_persona="platform-admin",
        ),
        ConfigFallbackCase(
            name="config defaults to readonly when config has readonly",
            config_persona="readonly",
            cli_args=["serve"],
            expected_persona="readonly",
        ),
    ],
)
def test_serve_persona_config_fallback(test_case):
    config = _make_default_config(persona=test_case.config_persona)
    mock_server, mock_create_server, modules_patch = _make_serve_mocks(config=config)

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, test_case.cli_args)

    mock_create_server.assert_called_once()
    _, kwargs = mock_create_server.call_args
    assert kwargs["persona"] == test_case.expected_persona


def test_serve_clients_config_fallback():
    config = _make_default_config(clients=["trainer", "optimizer"])
    mock_server, mock_create_server, modules_patch = _make_serve_mocks(config=config)

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve"])

    mock_create_server.assert_called_once()
    _, kwargs = mock_create_server.call_args
    assert kwargs["clients"] == ["trainer", "optimizer"]


def test_serve_clients_cli_overrides_config():
    config = _make_default_config(clients=["trainer", "optimizer"])
    mock_server, mock_create_server, modules_patch = _make_serve_mocks(config=config)

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve", "--clients", "hub"])

    mock_create_server.assert_called_once()
    _, kwargs = mock_create_server.call_args
    assert kwargs["clients"] == ["hub"]


def test_serve_transport_config_fallback():
    config = _make_default_config(transport="http")
    mock_server, _, modules_patch = _make_serve_mocks(config=config)

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve"])

    _, kwargs = mock_server.run.call_args
    assert kwargs.get("transport") == "streamable-http"


# --- serve: auth token wiring ---


def test_serve_http_calls_build_auth_provider():
    """HTTP transport triggers build_auth_provider with config."""
    mock_server, mock_create_server, modules_patch = _make_serve_mocks()
    fake_auth_mod = modules_patch["kubeflow_mcp.core.auth"]

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve", "--transport", "http"])

    fake_auth_mod.build_auth_provider.assert_called_once()


def test_serve_auth_token_passed_to_create_server():
    """--auth-token with HTTP transport produces a non-None auth_provider."""
    mock_server, mock_create_server, modules_patch = _make_serve_mocks()
    fake_auth_mod = modules_patch["kubeflow_mcp.core.auth"]
    fake_auth_mod.build_auth_provider.return_value = MagicMock(name="auth_provider")

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve", "--transport", "http", "--auth-token", "secret123"])

    mock_create_server.assert_called_once()
    _, kwargs = mock_create_server.call_args
    assert kwargs["auth_provider"] is not None


def test_serve_stdio_skips_auth_provider():
    """stdio transport does not call build_auth_provider."""
    mock_server, mock_create_server, modules_patch = _make_serve_mocks()
    fake_auth_mod = modules_patch["kubeflow_mcp.core.auth"]

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve", "--transport", "stdio"])

    fake_auth_mod.build_auth_provider.assert_not_called()
    mock_create_server.assert_called_once()
    _, kwargs = mock_create_server.call_args
    assert kwargs["auth_provider"] is None


# --- serve: resilience wiring ---


def test_serve_calls_configure_resilience():
    """serve() wires rate limiter from config."""
    mock_server, _, modules_patch = _make_serve_mocks()
    fake_server_mod = modules_patch["kubeflow_mcp.core.server"]

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve"])

    fake_server_mod.configure_resilience.assert_called_once()


def test_serve_calls_configure_circuit_breaker():
    """serve() wires circuit breaker from config."""
    mock_server, _, modules_patch = _make_serve_mocks()
    fake_resilience_mod = modules_patch["kubeflow_mcp.core.resilience"]

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve"])

    fake_resilience_mod.configure_circuit_breaker.assert_called_once()


# --- serve: banner flag ---


def test_serve_no_banner_flag():
    """--no-banner passes show_banner=False to server.run."""
    mock_server, _, modules_patch = _make_serve_mocks()

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve", "--no-banner"])

    _, kwargs = mock_server.run.call_args
    assert kwargs.get("show_banner") is False


def test_serve_default_shows_banner():
    """Without --no-banner, show_banner=True."""
    mock_server, _, modules_patch = _make_serve_mocks()

    with patch.dict(sys.modules, modules_patch):
        runner = CliRunner()
        runner.invoke(cli, ["serve"])

    _, kwargs = mock_server.run.call_args
    assert kwargs.get("show_banner") is True


def test_agent_unknown_provider():
    runner = CliRunner()
    result = runner.invoke(cli, ["agent", "--provider", "nonexistent-xyz"])
    assert result.exit_code == 1
    assert "Unknown provider" in result.output
    assert "ollama" in result.output or "litellm" in result.output


def test_agent_invokes_registered_provider():
    mock_provider = MagicMock()
    mock_cls = MagicMock(return_value=mock_provider)
    fake_ep = MagicMock()
    fake_ep.name = "fake"
    fake_ep.load = MagicMock(return_value=mock_cls)

    with patch("kubeflow_mcp.cli._provider_entry_point_map", return_value={"fake": fake_ep}):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["agent", "--provider", "fake", "--model", "m1", "--mode", "full"],
        )

    assert result.exit_code == 0
    mock_cls.assert_called_once_with()
    mock_provider.run.assert_called_once()
    call_kw = mock_provider.run.call_args.kwargs
    assert call_kw["model"] == "m1"
    assert call_kw["mode"] == "full"
    assert call_kw["thinking"] is False


def test_agent_passes_url_to_provider():
    mock_provider = MagicMock()
    mock_cls = MagicMock(return_value=mock_provider)
    fake_ep = MagicMock()
    fake_ep.load = MagicMock(return_value=mock_cls)

    with patch("kubeflow_mcp.cli._provider_entry_point_map", return_value={"fake": fake_ep}):
        runner = CliRunner()
        runner.invoke(
            cli,
            ["agent", "--provider", "fake", "--url", "http://ollama:11434"],
        )

    kwargs = mock_provider.run.call_args.kwargs
    assert kwargs["url"] == "http://ollama:11434"
