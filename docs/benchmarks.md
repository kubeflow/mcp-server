# Benchmarks

This project has a pytest-based benchmark suite under `tests/benchmarks/`.

Run:

```bash
uv run pytest tests/benchmarks/
```

This writes local output to:

```text
benchmark-results/latency.json
benchmark-results/index.html
```

`benchmark-results/` is generated output. It should be overwritten on each run and should not be committed unless otherwise noted.

## Current Scope

The current benchmark suite measures latency only. Latency uses `pytest-benchmark` for warmup, repeated rounds, and timing collection. The project writes a small normalized JSON file so the HTML report can use the same format when future benchmark suites are added.

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

## Report

The HTML report is generated from benchmark JSON files.

Current report includes:

- P50
- P95
- P99
- min
- max
- P99/P50 tail ratio
- visual spread bars
- summary cards

Open:

```text
benchmark-results/index.html
```

## Future Suites

Planned benchmark files:

```text
tests/benchmarks/test_token_usage.py
tests/benchmarks/test_cpu_profile.py
tests/benchmarks/test_memory.py
```

Planned output files:

```text
benchmark-results/token_usage.json
benchmark-results/cpu_profile.json
benchmark-results/memory.json
```

Recommended tools:

- token usage: custom estimator
- CPU profile: Python `cProfile`
- memory profile: Python `tracemalloc`

Keep dependencies minimal. Add external profiling dependencies only if stdlib tools are not enough.
