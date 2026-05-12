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
from unittest.mock import MagicMock, patch

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
    assert result.exit_code != 0
    assert "invalid choice" in result.output.lower() or "hacker" in result.output


def test_serve_rejects_invalid_transport():
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--transport", "websocket"])
    assert result.exit_code != 0
    assert "invalid choice" in result.output.lower() or "websocket" in result.output


# --- serve: wiring — create_server receives correct args, server.run is called ---


def _make_serve_mocks():
    """Return (mock_server, sys.modules patch dict) for serve command tests.

    serve() does lazy imports of core.logging and core.server inside the
    function body — patch sys.modules so those imports resolve to mocks
    regardless of whether the modules exist on the current branch.
    """

    mock_server = MagicMock()
    mock_create_server = MagicMock(return_value=mock_server)
    mock_setup_logging = MagicMock(return_value=MagicMock())

    fake_server_mod = MagicMock()
    fake_server_mod.create_server = mock_create_server
    fake_logging_mod = MagicMock()
    fake_logging_mod.setup_logging = mock_setup_logging

    modules_patch = {
        "kubeflow_mcp.core.server": fake_server_mod,
        "kubeflow_mcp.core.logging": fake_logging_mod,
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
    assert result.exit_code != 0
    assert "invalid choice" in result.output.lower() or "turbo" in result.output
