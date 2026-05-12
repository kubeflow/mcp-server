"""Tests for configuration loading."""

from kubeflow_mcp.core.config import Config, ServerConfig, load_config


def test_load_config_defaults():
    config = load_config()
    assert config.server.clients == ["trainer"]
    assert config.server.persona == "readonly"
    assert config.server.transport == "stdio"


def test_load_config_env_clients(monkeypatch):
    monkeypatch.setenv("KUBEFLOW_MCP_CLIENTS", "trainer,optimizer")
    config = load_config()
    assert config.server.clients == ["trainer", "optimizer"]


def test_load_config_env_persona(monkeypatch):
    monkeypatch.setenv("KUBEFLOW_MCP_PERSONA", "readonly")
    config = load_config()
    assert config.server.persona == "readonly"


def test_load_config_env_transport(monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "sse")
    config = load_config()
    assert config.server.transport == "sse"


def test_load_config_env_log_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    config = load_config()
    assert config.logging.level == "DEBUG"


def test_load_config_from_yaml_file(tmp_path):
    config_file = tmp_path / ".kubeflow-mcp.yaml"
    config_file.write_text(
        "server:\n"
        "  clients:\n"
        "    - trainer\n"
        "    - optimizer\n"
        "  persona: data-scientist\n"
        "  transport: sse\n"
        "logging:\n"
        "  level: DEBUG\n"
    )
    config = load_config(config_path=config_file)
    assert config.server.clients == ["trainer", "optimizer"]
    assert config.server.persona == "data-scientist"
    assert config.server.transport == "sse"
    assert config.logging.level == "DEBUG"


def test_load_config_yaml_not_found(tmp_path):
    missing = tmp_path / "missing.yaml"
    config = load_config(config_path=missing)
    assert config.server.clients == ["trainer"]


def test_load_config_yaml_invalid(tmp_path):
    bad_file = tmp_path / ".kubeflow-mcp.yaml"
    bad_file.write_text(": invalid: yaml: content: [\n")
    config = load_config(config_path=bad_file)
    assert config.server.clients == ["trainer"]


def test_load_config_trainer_section(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("trainer:\n  default_runtime: torch-distributed\n")
    config = load_config(config_path=config_file)
    assert config.trainer.default_runtime == "torch-distributed"


def test_server_config_defaults():
    cfg = ServerConfig()
    assert cfg.clients == ["trainer"]
    assert cfg.persona == "readonly"
    assert cfg.transport == "stdio"


def test_config_is_pydantic_model():
    config = load_config()
    assert isinstance(config, Config)
