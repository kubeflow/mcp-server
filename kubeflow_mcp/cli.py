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

"""Kubeflow MCP Server CLI."""

import warnings

import click

from kubeflow_mcp import __version__

# Suppress pydantic warnings from fastmcp/mcp dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Kubeflow MCP Server - AI interface for Kubeflow Training."""
    pass


@cli.command()
@click.option(
    "--clients",
    "-c",
    default=None,
    help="Comma-separated client modules (trainer, optimizer, hub). "
    "Falls back to KUBEFLOW_MCP_CLIENTS env var, config file, then 'trainer'.",
)
@click.option(
    "--persona",
    "-p",
    default=None,
    type=click.Choice(["readonly", "data-scientist", "ml-engineer", "platform-admin"]),
    help="Persona for tool filtering. "
    "Falls back to KUBEFLOW_MCP_PERSONA env var, config file, then 'readonly'.",
)
@click.option(
    "--transport",
    "-t",
    default=None,
    type=click.Choice(["stdio", "http", "sse"]),
    help="MCP transport protocol. Falls back to MCP_TRANSPORT env var, config file, then 'stdio'.",
)
@click.option(
    "--log-level",
    "-l",
    default=None,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level. Falls back to LOG_LEVEL env var, config file, then 'INFO'.",
)
@click.option(
    "--mode",
    "-m",
    default="full",
    type=click.Choice(["full", "progressive", "semantic"]),
    help="Tool loading mode: full (all tools), progressive (hierarchical discovery), semantic (embedding search)",
)
@click.option(
    "--log-format",
    default=None,
    type=click.Choice(["json", "console"]),
    help="Log format (auto-detects if not specified). Falls back to LOG_FORMAT env var, config file.",
)
@click.option(
    "--instruction-tier",
    default=None,
    type=click.Choice(["full", "compact", "minimal"]),
    help="Instruction verbosity: full (all guidance), compact (no resource refs), minimal (tool names only). "
    "Falls back to KUBEFLOW_MCP_INSTRUCTION_TIER env var, config file, then 'full'.",
)
@click.option(
    "--no-banner",
    is_flag=True,
    default=False,
    help="Hide FastMCP startup banner",
)
@click.option(
    "--auth-token",
    default=None,
    help="Bearer token for HTTP auth (dev/staging). "
    "Falls back to KUBEFLOW_MCP_AUTH_TOKEN env var, config file. "
    "Ignored for stdio transport.",
)
def serve(
    clients: str | None,
    persona: str | None,
    transport: str | None,
    mode: str,
    log_level: str | None,
    log_format: str | None,
    instruction_tier: str | None,
    no_banner: bool,
    auth_token: str | None,
) -> None:
    """Start the MCP server.

    Options fall back to env vars / config file (~/.kubeflow-mcp.yaml) when
    not provided on the command line.  See ``kubeflow_mcp.core.config`` for the
    full precedence chain: CLI flag > env var > config file > built-in default.
    """
    from kubeflow_mcp.core.auth import build_auth_provider
    from kubeflow_mcp.core.config import load_config
    from kubeflow_mcp.core.logging import setup_logging
    from kubeflow_mcp.core.resilience import configure_circuit_breaker
    from kubeflow_mcp.core.server import configure_resilience, create_server

    cfg = load_config()

    clients = clients or ",".join(cfg.server.clients)
    persona = persona or cfg.server.persona
    transport = transport or cfg.server.transport
    instruction_tier = instruction_tier or cfg.server.instruction_tier
    log_level = log_level or cfg.logging.level
    log_format = log_format or cfg.logging.format

    if auth_token:
        cfg.auth.auth_token = auth_token

    logger = setup_logging(level=log_level, format=log_format)
    logger.info(
        "Starting kubeflow-mcp",
        extra={
            "clients": clients,
            "persona": persona,
            "transport": transport,
            "mode": mode,
            "instruction_tier": instruction_tier,
        },
    )

    configure_resilience(
        rate_limit=cfg.resilience.rate_limit,
        rate_capacity=cfg.resilience.rate_capacity,
    )
    configure_circuit_breaker(
        failure_threshold=cfg.resilience.cb_failure_threshold,
        recovery_timeout=cfg.resilience.cb_recovery_timeout,
    )

    auth_provider = None
    if transport != "stdio":
        auth_provider = build_auth_provider(cfg.auth)
        if auth_provider is None:
            logger.warning(
                "HTTP transport with no auth configured — server is open. "
                "Set --auth-token or KUBEFLOW_MCP_AUTH_TOKEN for bearer auth, "
                "or KUBEFLOW_MCP_JWKS_URI for JWT verification."
            )

    client_list = [c.strip() for c in clients.split(",")]
    server = create_server(
        clients=client_list,
        persona=persona,
        mode=mode,
        instruction_tier=instruction_tier,
        auth_provider=auth_provider,
    )

    show_banner = not no_banner
    if transport == "stdio":
        server.run(show_banner=show_banner)
    elif transport == "sse":
        server.run(transport="sse", show_banner=show_banner)
    else:
        server.run(transport="streamable-http", show_banner=show_banner)


@cli.command()
def status() -> None:
    """Show server status and enabled tools."""
    from kubeflow_mcp.core.server import CLIENT_MODULES

    click.echo("Kubeflow MCP Server Status")
    click.echo("-" * 40)
    click.echo(f"Version: {__version__}")
    click.echo("\nAvailable clients:")
    for name, module_path in CLIENT_MODULES.items():
        try:
            import importlib

            module = importlib.import_module(module_path)
            info = getattr(module, "MODULE_INFO", {})
            status = info.get("status", "unknown")
            tools = len(getattr(module, "TOOLS", []))
            click.echo(f"  {name}: {status} ({tools} tools)")
        except ImportError:
            click.echo(f"  {name}: not installed")


@cli.command()
@click.option(
    "--backend",
    "-b",
    default="ollama",
    type=click.Choice(["ollama"]),
    help="Agent backend",
)
@click.option(
    "--model",
    "-m",
    default="qwen3:8b",
    help="Model name for the agent",
)
@click.option(
    "--mode",
    default="full",
    type=click.Choice(["full", "progressive", "semantic", "static", "mcp"]),
    help="Tool loading mode (static/mcp are legacy aliases for full)",
)
@click.option(
    "--thinking/--no-thinking",
    default=False,
    help="Enable thinking output for supported models",
)
def agent(backend: str, model: str, mode: str, thinking: bool) -> None:
    """Run an interactive AI agent."""
    if backend == "ollama":
        try:
            from kubeflow_mcp.agents.ollama import main as ollama_main
        except ImportError:
            click.echo("Error: Agent dependencies not installed.", err=True)
            click.echo("Install with: pip install ollama", err=True)
            raise SystemExit(1) from None

        import sys

        sys.argv = [
            "kubeflow-mcp-agent",
            "--model",
            model,
            "--mode",
            mode,
        ]
        if thinking:
            sys.argv.append("--thinking")
        ollama_main()
    else:
        click.echo(f"Backend '{backend}' not yet implemented.", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
