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

"""Configuration schema for kubeflow-mcp.

Supports configuration from:
1. Config file: ~/.kubeflow-mcp.yaml
2. Environment variables (override config file)
3. CLI arguments (override both)

Example config file (~/.kubeflow-mcp.yaml):

.. code-block:: yaml

        server:
            clients:
                - trainer
                - optimizer
            persona: ml-engineer
            transport: stdio

        auth:
            auth_token: my-secret-token   # simple API key (dev/staging)
            # OR for JWT:
            # jwks_uri: https://auth.example.com/.well-known/jwks.json
            # issuer: https://auth.example.com
            # audience: kubeflow-mcp

        resilience:
            rate_limit: 10.0
            rate_capacity: 20.0
            cb_failure_threshold: 5
            cb_recovery_timeout: 30.0

        trainer:
            default_runtime: torch-distributed

        logging:
            level: INFO
            format: json

        observability:
            otel_endpoint: http://localhost:4318/v1/traces

Namespace restrictions are enforced via ``~/.kf-mcp-policy.yaml``
(``policy.namespaces``), not through server config.
"""

import logging
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _get_config_paths() -> list[Path]:
    """Config file locations, searched in order. Evaluated lazily so cwd is fresh."""
    return [
        Path.home() / ".kubeflow-mcp.yaml",
        Path.home() / ".kubeflow-mcp.yml",
        Path.home() / ".config" / "kubeflow-mcp" / "config.yaml",
        Path.cwd() / ".kubeflow-mcp.yaml",
    ]


class ServerConfig(BaseModel):
    """Server configuration.

    Namespace restrictions are enforced via ``~/.kf-mcp-policy.yaml``
    (``policy.namespaces``), not here.
    """

    clients: list[str] = Field(default=["trainer"])
    persona: str = Field(default="readonly")
    transport: Literal["stdio", "http", "sse"] = Field(default="stdio")
    instruction_tier: Literal["full", "compact", "minimal"] = Field(default="full")


class TrainerConfig(BaseModel):
    """Trainer-specific configuration."""

    default_runtime: str | None = None
    default_namespace: str | None = None
    controller_namespace: str | None = None


class AuthConfig(BaseModel):
    """HTTP transport authentication configuration.

    Supports two modes:
    - **API key**: set ``auth_token`` for simple bearer token auth (dev/staging).
    - **JWT**: set ``jwks_uri`` (+ optional ``issuer``, ``audience``) for
      production JWT verification via FastMCP's ``JWTVerifier``.

    Ignored when ``transport=stdio`` (stdio inherits OS-level security).
    """

    auth_token: str | None = Field(default=None)
    jwks_uri: str | None = Field(default=None)
    issuer: str | None = Field(default=None)
    audience: str | None = Field(default=None)


class ResilienceConfig(BaseModel):
    """Rate limiter and circuit breaker tuning."""

    rate_limit: float = Field(default=10.0, description="Requests/second refill rate")
    rate_capacity: float = Field(default=20.0, description="Max burst capacity")
    cb_failure_threshold: int = Field(default=5, description="Failures before circuit opens")
    cb_recovery_timeout: float = Field(default=30.0, description="Seconds before half-open retry")


class OptimizerConfig(BaseModel):
    """Optimizer-specific configuration."""

    default_algorithm: str = Field(default="random")
    max_trials: int = Field(default=10)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    format: str | None = Field(default=None)


class ObservabilityConfig(BaseModel):
    """Observability configuration."""

    otel_endpoint: str | None = Field(default=None)


class Config(BaseModel):
    """Root configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


def _find_config_file() -> Path | None:
    """Find the first existing config file."""
    for path in _get_config_paths():
        if path.exists():
            return path
    return None


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load config from YAML file."""
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except ImportError:
        logger.warning("PyYAML not installed, skipping config file")
        return {}
    except Exception as e:
        logger.warning(f"Failed to load config from {path}: {e}")
        return {}


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from file and environment variables.

    Priority (highest to lowest):
    1. Environment variables
    2. Specified config file
    3. Default config file locations
    4. Default values

    Args:
        config_path: Optional explicit path to config file

    Returns:
        Merged configuration
    """
    # Start with defaults
    file_config: dict[str, Any] = {}

    # Load from file
    if config_path and config_path.exists():
        file_config = _load_yaml_config(config_path)
        logger.debug(f"Loaded config from {config_path}")
    else:
        default_path = _find_config_file()
        if default_path:
            file_config = _load_yaml_config(default_path)
            logger.debug(f"Loaded config from {default_path}")

    # Build server config with env overrides
    server_file = file_config.get("server", {})
    server = ServerConfig(
        clients=[
            c.strip()
            for c in os.getenv(
                "KUBEFLOW_MCP_CLIENTS",
                ",".join(server_file.get("clients", ["trainer"])),
            ).split(",")
        ],
        persona=os.getenv(
            "KUBEFLOW_MCP_PERSONA",
            server_file.get("persona", "readonly"),
        ),
        transport=os.getenv(
            "MCP_TRANSPORT",
            server_file.get("transport", "stdio"),
        ),
        instruction_tier=os.getenv(
            "KUBEFLOW_MCP_INSTRUCTION_TIER",
            server_file.get("instruction_tier", "full"),
        ),
    )

    # Build logging config with env overrides
    logging_file = file_config.get("logging", {})
    logging_config = LoggingConfig(
        level=os.getenv("LOG_LEVEL", logging_file.get("level", "INFO")),
        format=os.getenv("LOG_FORMAT", logging_file.get("format")),
    )

    observability_file = file_config.get("observability", {})
    observability = ObservabilityConfig(
        otel_endpoint=os.getenv(
            "KUBEFLOW_MCP_OTEL_ENDPOINT",
            observability_file.get("otel_endpoint"),
        )
    )

    # Build client-specific configs
    trainer_file = file_config.get("trainer", {})
    trainer = TrainerConfig(
        default_runtime=trainer_file.get("default_runtime"),
        default_namespace=trainer_file.get("default_namespace"),
        controller_namespace=os.getenv(
            "KUBEFLOW_MCP_CONTROLLER_NAMESPACE",
            trainer_file.get("controller_namespace"),
        ),
    )

    optimizer_file = file_config.get("optimizer", {})
    optimizer = OptimizerConfig(
        default_algorithm=optimizer_file.get("default_algorithm", "random"),
        max_trials=optimizer_file.get("max_trials", 10),
    )

    auth_file = file_config.get("auth", {})
    auth = AuthConfig(
        auth_token=os.getenv("KUBEFLOW_MCP_AUTH_TOKEN", auth_file.get("auth_token")),
        jwks_uri=os.getenv("KUBEFLOW_MCP_JWKS_URI", auth_file.get("jwks_uri")),
        issuer=os.getenv("KUBEFLOW_MCP_JWT_ISSUER", auth_file.get("issuer")),
        audience=os.getenv("KUBEFLOW_MCP_JWT_AUDIENCE", auth_file.get("audience")),
    )

    resilience_file = file_config.get("resilience", {})
    resilience = ResilienceConfig(
        rate_limit=float(
            os.getenv("KUBEFLOW_MCP_RATE_LIMIT", resilience_file.get("rate_limit", 10.0))
        ),
        rate_capacity=float(
            os.getenv("KUBEFLOW_MCP_RATE_CAPACITY", resilience_file.get("rate_capacity", 20.0))
        ),
        cb_failure_threshold=int(
            os.getenv(
                "KUBEFLOW_MCP_CB_FAILURE_THRESHOLD",
                resilience_file.get("cb_failure_threshold", 5),
            )
        ),
        cb_recovery_timeout=float(
            os.getenv(
                "KUBEFLOW_MCP_CB_RECOVERY_TIMEOUT",
                resilience_file.get("cb_recovery_timeout", 30.0),
            )
        ),
    )

    return Config(
        server=server,
        auth=auth,
        resilience=resilience,
        trainer=trainer,
        optimizer=optimizer,
        logging=logging_config,
        observability=observability,
    )


def get_config_path() -> Path | None:
    """Get the path to the active config file (if any)."""
    return _find_config_file()
