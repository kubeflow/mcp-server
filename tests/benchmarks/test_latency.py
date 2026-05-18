import time , json
from pathlib import Path
from collections.abc import Callable
import inspect
from kubeflow_mcp.core.server import create_server
from kubeflow_mcp.trainer import CLIENT_TOOL_ANNOTATIONS, CLIENT_TOOL_DESCRIPTIONS, TOOLS
from kubeflow_mcp.core.dynamic_tools import init_dynamic_tools
from kubeflow_mcp.core.security import (
    is_safe_python_code,
    validate_k8s_name,
    validate_resource_limits,
)

DEFAULT_ITERATIONS = 100
DEFAULT_WARMUP = 5
OUTPUT_DIR = Path("benchmark-results")

def measure_latency_ms(func: Callable[[], object] , iterations = DEFAULT_ITERATIONS , warmup = DEFAULT_WARMUP) -> dict[str,float]:
    for _ in range(warmup):
        func()
    
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        samples.append((time.perf_counter_ns() - start )/ 1_000_000)
    samples.sort()

    return {
        "p50": percentile(samples,50),
        "p95": percentile(samples,95),
        "p99": percentile(samples,99),
        "min": samples[0],
        "max": samples[-1],
    }


def latency_result(name:str,measurements:dict[str,float]) -> dict[str,object]:
    return {
        "name":name,
        "unit":"ms",
        **measurements,
    }

def percentile(sorted_samples: list[float], value: int) -> float:
    index = round((value / 100) * (len(sorted_samples) - 1))
    return sorted_samples[index]


def benchmark_server_init() -> list[dict[str,object]]:
    
    results = []
    for mode in ("full", "progressive", "semantic"):
        def create_trainer_server(mode: str = mode) -> object:
            return create_server(
                clients=["trainer"],
                persona="readonly",
                mode=mode,
            )

        measurements = measure_latency_ms(create_trainer_server)
        results.append(latency_result(f"server_init_{mode}", measurements))

    return results

def benchmark_dynamic_tool_registry_init() -> list[dict[str, object]]:
    def initialize_registry() -> None:
        init_dynamic_tools(TOOLS, CLIENT_TOOL_DESCRIPTIONS)

    return [
        latency_result(
            "dynamic_tool_registry_init",
            measure_latency_ms(initialize_registry),
        )
    ]


def benchmark_trainer_schema_metadata_scan() -> list[dict[str, object]]:
    def scan_metadata() -> None:
        for tool in TOOLS:
            tool_name = tool.__name__
            inspect.signature(tool)
            CLIENT_TOOL_DESCRIPTIONS[tool_name]
            CLIENT_TOOL_ANNOTATIONS[tool_name]

    return [
        latency_result(
            "trainer_schema_metadata_scan",
            measure_latency_ms(scan_metadata),
        )
    ]


def benchmark_security_validation() -> list[dict[str, object]]:
    cases: list[tuple[str, Callable[[], object]]] = [
        ("validate_k8s_name", lambda: validate_k8s_name("valid-training-job")),
        (
            "validate_resource_limits",
            lambda: validate_resource_limits(cpu="1", memory="2Gi", gpu=1),
        ),
        (
            "is_safe_python_code",
            lambda: is_safe_python_code("def train():\n    return 1\n"),
        ),
    ]

    return [
        latency_result(name, measure_latency_ms(func))
        for name, func in cases
    ]

def run_latency_benchmarks(
    output_dir: Path = OUTPUT_DIR,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
    ) -> Path:
    
    output_dir.mkdir(exist_ok=True)
    results = [] # list[dict[str,object]]
    results.extend(benchmark_server_init())
    results.extend(benchmark_dynamic_tool_registry_init())
    results.extend(benchmark_trainer_schema_metadata_scan())    
    results.extend(benchmark_security_validation())
    
    info = {
        "suite": "latency",
        "iterations": iterations,
        "warmup": warmup,
        "results": results,
    }
    
    output_path = output_dir / "latency.json"
    output_path.write_text(json.dumps(info, indent=2) + "\n")
    return output_path


if __name__ == "__main__":
    output = run_latency_benchmarks()
    print(f"Results written to: {output}")