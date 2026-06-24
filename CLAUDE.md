# Kubeflow MCP (this repository)

- **MCP server**: `uv run kubeflow-mcp serve` — exposes Kubeflow training tools over MCP (stdio by default).
- **Local agent** (optional LLM): `uv sync --extra agents-ollama` then `uv run kubeflow-mcp agent --provider ollama --model qwen3:8b` (default `--mode full`; add `--mode progressive` to match smaller tool schemas like `serve --mode progressive`). Use `--extra agents` for Ollama + LiteLLM together.
- **Docs**: `README.md`, `ROADMAP.md`, `docs/design/agent-provider-architecture.md`.

Use MCP tools instead of guessing kubectl; respect preview-before-submit (`confirmed` flags on mutating tools).
