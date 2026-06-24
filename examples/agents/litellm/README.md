# LiteLLM example

Install `uv sync --extra agents-litellm` (or `agents`), set `OPENAI_API_KEY` (or other provider env vars), then:

```bash
uv run python examples/agents/litellm/run.py --model gpt-4o-mini
```

This is a minimal chat loop. For full Kubeflow tool calling, use `--provider ollama` today.
