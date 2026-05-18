"""Latency benchmarks for trainer MCP tools."""

import inspect
from collections.abc import Callable

import pytest

from kubeflow_mcp.core.dynamic_tools import (
    describe_tools,
    find_tools,
    init_dynamic_tools,
    list_tools,
)
from kubeflow_mcp.core.security import (
    is_safe_python_code,
    validate_k8s_name,
    validate_resource_limits,
)
from kubeflow_mcp.core.server import create_server
from kubeflow_mcp.trainer import CLIENT_TOOL_ANNOTATIONS, CLIENT_TOOL_DESCRIPTIONS, TOOLS
from kubeflow_mcp.trainer.api import training
from kubeflow_mcp.trainer.api.training import (
    fine_tune,
    run_container_training,
    run_custom_training,
)


@pytest.mark.parametrize("mode", ["full", "progressive", "semantic"])
def test_server_init_latency(
    record_latency_benchmark: Callable[[str, Callable[[], object]], None], mode: str
) -> None:
    def create_trainer_server() -> object:
        return create_server(
            clients=["trainer"],
            persona="readonly",
            mode=mode,
        )

    record_latency_benchmark(f"server_init_{mode}", create_trainer_server)


def test_dynamic_tool_registry_init_latency(
    record_latency_benchmark: Callable[[str, Callable[[], object]], None],
) -> None:
    def initialize_registry() -> None:
        init_dynamic_tools(TOOLS, CLIENT_TOOL_DESCRIPTIONS)

    record_latency_benchmark("dynamic_tool_registry_init", initialize_registry)


@pytest.mark.parametrize(
    ("name", "benchmark_func"),
    [
        ("dynamic_list_tools", lambda: list_tools()),
        (
            "dynamic_describe_tools",
            lambda: describe_tools(["fine_tune", "run_custom_training", "get_training_logs"]),
        ),
        ("dynamic_find_tools_keyword", lambda: find_tools("fine tune a model", top_k=5)),
    ],
)
def test_dynamic_discovery_tools_latency(
    record_latency_benchmark: Callable[[str, Callable[[], object]], None],
    name: str,
    benchmark_func: Callable[[], object],
) -> None:
    init_dynamic_tools(TOOLS, CLIENT_TOOL_DESCRIPTIONS)

    record_latency_benchmark(name, benchmark_func)


def test_trainer_schema_metadata_scan_latency(
    record_latency_benchmark: Callable[[str, Callable[[], object]], None],
) -> None:
    def scan_metadata() -> None:
        for tool in TOOLS:
            tool_name = tool.__name__
            inspect.signature(tool)
            CLIENT_TOOL_DESCRIPTIONS[tool_name]
            CLIENT_TOOL_ANNOTATIONS[tool_name]

    record_latency_benchmark("trainer_schema_metadata_scan", scan_metadata)


@pytest.mark.parametrize(
    ("name", "benchmark_func"),
    [
        ("validate_k8s_name", lambda: validate_k8s_name("valid-training-job")),
        (
            "validate_resource_limits",
            lambda: validate_resource_limits(cpu="1", memory="2Gi", gpu=1),
        ),
        (
            "is_safe_python_code",
            lambda: is_safe_python_code("def train():\n    return 1\n"),
        ),
    ],
)
def test_security_validation_latency(
    record_latency_benchmark: Callable[[str, Callable[[], object]], None],
    name: str,
    benchmark_func: Callable[[], object],
) -> None:
    record_latency_benchmark(name, benchmark_func)


def preview_fine_tune() -> dict[str, object]:
    return fine_tune(
        model="hf://google/gemma-2b",
        dataset="hf://tatsu-lab/alpaca",
        runtime="torchtune-llama3.2-1b",
        name="bench-fine-tune",
        confirmed=False,
    )


def preview_custom_training() -> dict[str, object]:
    return run_custom_training(
        script="print('training')",
        runtime="torch-distributed",
        name="bench-custom-training",
        gpu_per_node=0,
        confirmed=False,
    )


def preview_container_training() -> dict[str, object]:
    return run_container_training(
        image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
        command=["python", "-c"],
        args=["print('training')"],
        name="bench-container-training",
        runtime="torch-distributed",
        gpu_per_node=0,
        confirmed=False,
    )


@pytest.mark.parametrize(
    ("name", "benchmark_func"),
    [
        ("preview_fine_tune", preview_fine_tune),
        ("preview_custom_training", preview_custom_training),
        ("preview_container_training", preview_container_training),
    ],
)
def test_preview_tools_latency(
    record_latency_benchmark: Callable[[str, Callable[[], object]], None],
    name: str,
    benchmark_func: Callable[[], object],
) -> None:
    original_gpu_check = training._check_gpu_available
    training._check_gpu_available = lambda: None
    try:
        record_latency_benchmark(name, benchmark_func)
    finally:
        training._check_gpu_available = original_gpu_check
