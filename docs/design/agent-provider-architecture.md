# Agent provider architecture

## Summary

This document proposes a **pluggable agent provider** model for the `kubeflow-mcp agent` CLI. Providers register via Python `importlib.metadata` entry points (`kubeflow_mcp.providers`), implement a small `AgentProvider` protocol, and ship optional dependencies behind extras: **`agents-ollama`** (LlamaIndex + Ollama + Rich), **`agents-litellm`** (LiteLLM + Rich), or **`agents`** for both.

## Two execution planes

1. **MCP plane (vision-primary):** `kubeflow-mcp serve` → [`create_server`](../../kubeflow_mcp/core/server.py) → FastMCP tools with `_audit_wrap`, persona, policy, optional progressive/semantic meta-tools from [`core/dynamic_tools`](../../kubeflow_mcp/core/dynamic_tools.py).
2. **CLI agent plane (dev / convenience):** `kubeflow-mcp agent` → provider → LlamaIndex + same trainer callables; progressive/semantic use the **same** `core.dynamic_tools` implementation after `agents/dynamic_tools` calls `init_dynamic_tools`. Instructions and short tool descriptions align with `serve` via [`build_agent_instruction_text`](../../kubeflow_mcp/core/server.py) / [`get_merged_client_tool_descriptions`](../../kubeflow_mcp/core/server.py) (default trainer + health, persona `readonly`).

**`--mode`:** `serve` and `agent` both accept `full` | `progressive` | `semantic` with the **same** meaning; default is **`full`** for both (use `progressive` or `semantic` when you need smaller tool schemas). **Note:** `serve -m` is tool mode; `agent -m` is **`--model`**, not tool mode—use `agent --mode …` for tool mode.

**Registry note:** `init_dynamic_tools` mutates global state in `core.dynamic_tools`; avoid loading server and in-process agent meta-tools in one process without re-init expectations.

## Motivation

- Agent code previously lived under `src/kubeflow_mcp/agents/`, **outside** the wheel package, so imports only worked in editable installs.
- The CLI hard-coded `--backend ollama`, forcing code changes for every new backend.
- Community contributors need a clear pattern (protocol + entry point + example script).

## Goals

- Move agents into `kubeflow_mcp/agents/` as part of the published package.
- Dynamic discovery: `kubeflow-mcp agent --provider <name>`.
- Documented protocol and reference implementations: Ollama (full tools), LiteLLM (minimal REPL).

## Non-goals

- Changing MCP server transport or core tool registration.
- Mandating a single LLM stack (providers remain optional extras: `agents-ollama`, `agents-litellm`, or `agents`).

## Proposal

### Protocol

```text
kubeflow_mcp/agents/base.py → AgentProvider
  name: str
  default_model: str
  requires: list[str]   # pip package names for error messages
  run(self, model: str, mode: str, **kwargs) -> None
```

### Entry points

Registered in `pyproject.toml`:

```toml
[project.entry-points."kubeflow_mcp.providers"]
ollama = "kubeflow_mcp.agents.ollama:OllamaProvider"
litellm = "kubeflow_mcp.agents.litellm_provider:LiteLLMProvider"
```

### CLI

```bash
kubeflow-mcp agent --provider ollama --model qwen3:8b --mode full
kubeflow-mcp agent --provider ollama --model qwen3:8b --mode progressive
kubeflow-mcp agent --provider litellm --model gpt-4o-mini
```

Optional: `--url` for Ollama base URL; `--thinking` toggles reasoning-friendly models.

### Providers (status)

| Provider   | Status   | Notes                                      |
|-----------|----------|--------------------------------------------|
| `ollama`  | Shipped  | LlamaIndex + Kubeflow tools, full modes   |
| `litellm` | Minimal  | LiteLLM chat loop; tool parity TBD         |

## Implementation plan

1. Consolidate `src/kubeflow_mcp/agents/` into `kubeflow_mcp/agents/`.
2. Add `base.py`, entry points, CLI refactor.
3. Split Ollama REPL helpers to satisfy complexity limits.
4. Add `examples/agents/` and optional `examples/deployment/litellm-gateway/` notes.

## References

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Python importlib.metadata entry points](https://docs.python.org/3/library/importlib.metadata.html#entry-points)
- [LlamaIndex FunctionAgent](https://docs.llamaindex.ai/)
