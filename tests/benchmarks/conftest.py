"""Pytest wiring for benchmark result collection."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.benchmarks.report import generate_report

DEFAULT_ITERATIONS = 100
DEFAULT_WARMUP = 5
OUTPUT_DIR = Path(__file__).parent.parent.parent / "benchmark-results"
LATENCY_RESULTS: list[dict[str, object]] = []


def _percentile(sorted_samples: list[float], value: int) -> float:
    index = round((value / 100) * (len(sorted_samples) - 1))
    return sorted_samples[index]


def _latency_result(name: str, samples_ms: list[float]) -> dict[str, object]:
    sorted_samples = sorted(samples_ms)
    return {
        "name": name,
        "unit": "ms",
        "p50": _percentile(sorted_samples, 50),
        "p95": _percentile(sorted_samples, 95),
        "p99": _percentile(sorted_samples, 99),
        "min": sorted_samples[0],
        "max": sorted_samples[-1],
    }


@pytest.fixture
def record_latency_benchmark(
    benchmark: Any,
) -> Iterator[Callable[[str, Callable[[], object]], None]]:
    def run(name: str, func: Callable[[], object]) -> None:
        benchmark.pedantic(
            func,
            rounds=DEFAULT_ITERATIONS,
            warmup_rounds=DEFAULT_WARMUP,
            iterations=1,
        )
        samples_ms = [sample * 1_000 for sample in benchmark.stats["data"]]
        LATENCY_RESULTS.append(_latency_result(name, samples_ms))

    yield run


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    if not LATENCY_RESULTS:
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    latency_path = OUTPUT_DIR / "latency.json"
    latency_payload = {
        "suite": "latency",
        "iterations": DEFAULT_ITERATIONS,
        "warmup": DEFAULT_WARMUP,
        "results": LATENCY_RESULTS,
    }
    latency_path.write_text(json.dumps(latency_payload, indent=2) + "\n")
    report_path = generate_report(OUTPUT_DIR)

    terminalreporter.write_sep("-", f"benchmark results written to {latency_path}")
    terminalreporter.write_sep("-", f"benchmark report written to {report_path}")
