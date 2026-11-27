# Benchmarks

This project has a pytest-based benchmark suite under `tests/benchmarks/`.

Run:

```bash
make benchmark
```

Benchmarks carry the `benchmark` marker and the default `addopts` deselects it, so `make test-python`, `make test`, and a bare `pytest` skip them. The marker is applied to every test under `tests/benchmarks/` automatically, so new benchmark files need no decorator.

To run them without make:

```bash
uv run pytest tests/benchmarks/ -m benchmark
```

This writes local output to:

```text
benchmark-results/latency.json
benchmark-results/token_usage.json
benchmark-results/index.html
```

`benchmark-results/` is generated output and should not be committed unless otherwise noted.
Failed runs and runs that measure only some of the three `tools/list` modes leave existing
output intact.

## Current Scope

Latency benchmarks use `pytest-benchmark` for warmup, repeated rounds, and timing collection.

Covered latency cases:

- server init in `full` mode
- server init in `progressive` mode
- server init in `semantic` mode
- dynamic tool registry init
- dynamic discovery tools:
  - `list_tools`
  - `describe_tools`
  - `find_tools`
- trainer schema and metadata scan
- preview tool paths:
  - `fine_tune`
  - `run_custom_training`
  - `run_container_training`
- security validation:
  - `validate_k8s_name`
  - `validate_resource_limits`
  - `is_safe_python_code`

Preview benchmarks use `confirmed=False`, so they do not submit jobs to Kubernetes.

`fine_tune` GPU pre-check is patched inside the benchmark so the benchmark can run without a cluster.

The `tools/list` benchmark measures the first MCP result returned to an in-memory
client in `full`, `progressive`, and `semantic` mode. It loads only Trainer and
uses the built-in `readonly` persona. Each mode reports the tool count and result
size in bytes after compact UTF-8 JSON serialization, including descriptions and
schemas. The measurement excludes the JSON-RPC envelope, server instructions,
and later discovery calls. It requires no cluster, model, or API key.

To run just this benchmark:

```bash
uv run pytest tests/benchmarks/test_token_usage.py -m benchmark
```

## Output Format

Latency results use this shape:

```json
{
  "suite": "latency",
  "iterations": 100,
  "warmup": 5,
  "results": [
    {
      "name": "server_init_full",
      "unit": "ms",
      "p50": 18.1,
      "p95": 24.5,
      "p99": 31.2,
      "min": 15.9,
      "max": 40.8
    }
  ]
}
```

`token_usage.json` records `client`, `persona`, `mode`, `tool_count`, and
`payload_bytes` for each result. The measured bytes describe the initial MCP
payload. Model token counts depend on how each client formats tools for a
particular model and tokenizer.

## Report

The HTML report combines the available benchmark JSON files. A standalone suite run
updates its own JSON file, so the report can include other suites' results from earlier runs.

Current report includes:

- P50
- P95
- P99
- min
- max
- P99/P50 tail ratio
- visual spread bars
- summary cards
- initial `tools/list` tool counts and payload sizes by loading mode

Open:

```text
benchmark-results/index.html
```

## Future Suites

Planned benchmark files:

```text
tests/benchmarks/test_cpu_profile.py
tests/benchmarks/test_memory.py
```

Planned output files:

```text
benchmark-results/cpu_profile.json
benchmark-results/memory.json
```

Recommended tools:

- CPU profile: Python `cProfile`
- memory profile: Python `tracemalloc`

Keep dependencies minimal. Add external profiling dependencies only if stdlib tools are not enough.
