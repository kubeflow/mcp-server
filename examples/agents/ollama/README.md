# Ollama example

Requires `uv sync --extra agents-ollama` (or `agents` for all backends) and a running `ollama serve`.

```bash
uv run python examples/agents/ollama/run.py --model qwen3:8b --mode full
```
