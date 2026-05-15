# Example agent runners

Use the main CLI (recommended):

```bash
uv sync --extra agents-ollama
uv run kubeflow-mcp agent --provider ollama --model qwen3:8b
# Smaller tool schema (same as serve --mode progressive):
uv run kubeflow-mcp agent --provider ollama --model qwen3:8b --mode progressive
```

LiteLLM provider (separate extra):

```bash
uv sync --extra agents-litellm
uv run kubeflow-mcp agent --provider litellm --model gpt-4o-mini
```

All agent backends (Ollama + LiteLLM): `uv sync --extra agents`.

Or run the thin wrappers in this directory (same behavior, explicit `PYTHONPATH` not required when the package is installed):

```bash
uv run python examples/agents/ollama/run.py --model qwen3:8b
uv run python examples/agents/ollama/run.py --model qwen3:8b --mode progressive
```

See each subfolder `README.md` for provider-specific notes.
