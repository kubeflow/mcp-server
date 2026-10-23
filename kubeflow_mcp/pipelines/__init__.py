# Copyright The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pipelines client module — MCP tools for Kubeflow Pipelines.

Wraps ``kubeflow.pipelines.PipelinesClient`` (Kubeflow SDK 0.5.0, backed by kfp 2.17)
so agents can find registered pipelines, start runs and follow them. Structure
mirrors ``trainer/``:

├── api/       # Tool implementations (discovery, monitoring, runs)
├── types/     # SDK-object serialization helpers
└── resources/ # Agent-facing guides

``kfp`` ships behind the optional ``kubeflow[pipelines]`` extra and every SDK import
is lazy, so this module and its metadata load without it.
"""

from kubeflow_mcp.common.constants import KUBEFLOW_SDK_VERSION_SPEC
from kubeflow_mcp.pipelines.api.discovery import get_pipeline, list_pipelines
from kubeflow_mcp.pipelines.api.monitoring import (
    get_pipeline_logs,
    get_pipeline_run,
    list_pipeline_runs,
)
from kubeflow_mcp.pipelines.api.runs import (
    create_pipeline_run,
    retry_pipeline_run,
    stop_pipeline_run,
)

MODULE_INFO = {
    "name": "pipelines",
    "description": "Kubeflow Pipelines: registered pipelines and their runs",
    "sdk_client": "kubeflow.pipelines.PipelinesClient",
    "sdk_version": KUBEFLOW_SDK_VERSION_SPEC,
    "extra": "pipelines",
    "status": "implemented",
}

TOOLS = [
    list_pipelines,
    get_pipeline,
    list_pipeline_runs,
    get_pipeline_run,
    get_pipeline_logs,
    create_pipeline_run,
    stop_pipeline_run,
    retry_pipeline_run,
]

# ─── Tool metadata (owned by this client module) ───────────────────────────

CLIENT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "list_pipelines": "List pipelines registered on Kubeflow Pipelines. Paged with page_token.",
    "get_pipeline": "Get a registered pipeline by name, with its recent versions.",
    "list_pipeline_runs": (
        "List pipeline runs. Filter by pipeline, experiment or status (e.g. failed)."
    ),
    "get_pipeline_run": "Get a run's state, error, parameters and task states by run_id.",
    "get_pipeline_logs": (
        "Get logs of a run's task pods. Pass task= to read one task. Bounded output."
    ),
    "create_pipeline_run": (
        "Start a run of a registered pipeline by name. Preview first (confirmed=False), "
        "then set confirmed=True to start it."
    ),
    "stop_pipeline_run": (
        "[DESTRUCTIVE] Stop a running pipeline run. Preview first (confirmed=False), "
        "then set confirmed=True."
    ),
    "retry_pipeline_run": (
        "Retry a failed or canceled pipeline run. Preview first (confirmed=False), "
        "then set confirmed=True."
    ),
}

_READ = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
}

CLIENT_TOOL_ANNOTATIONS: dict[str, dict] = {
    "list_pipelines": {
        "title": "List Pipelines",
        **_READ,
        "tags": ["pipelines_discovery", "pipelines"],
    },
    "get_pipeline": {
        "title": "Get Pipeline",
        **_READ,
        "tags": ["pipelines_discovery", "pipelines"],
    },
    "list_pipeline_runs": {
        "title": "List Pipeline Runs",
        **_READ,
        "tags": ["pipelines_monitoring", "pipelines", "runs"],
    },
    "get_pipeline_run": {
        "title": "Get Pipeline Run",
        **_READ,
        "tags": ["pipelines_monitoring", "pipelines", "runs"],
    },
    "get_pipeline_logs": {
        "title": "Get Pipeline Run Logs",
        **_READ,
        "tags": ["pipelines_monitoring", "pipelines", "logs", "debug"],
    },
    "create_pipeline_run": {
        "title": "Create Pipeline Run",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["pipelines_runs", "pipelines", "runs", "create"],
    },
    "stop_pipeline_run": {
        "title": "Stop Pipeline Run",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["pipelines_lifecycle", "pipelines", "runs"],
    },
    "retry_pipeline_run": {
        "title": "Retry Pipeline Run",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["pipelines_lifecycle", "pipelines", "runs"],
    },
}

# ─── Resources (owned by this client module) ───────────────────────────────

CLIENT_RESOURCES: dict[str, tuple[str, str]] = {
    "pipelines://guides/workflow-patterns": (
        "resources/workflow-patterns.md",
        "Find a pipeline, start a run, follow it and debug a failed run.",
    ),
    "pipelines://guides/troubleshooting": (
        "resources/troubleshooting.md",
        "Error-to-fix table and known limitations of the pipelines tools.",
    ),
}

# ─── Instruction sections (full tier; compact/minimal auto-derived) ────────
# Phase names are prefixed so they never merge into the trainer's sections.

PHASE_TO_SECTION: dict[str, str | None] = {
    "pipelines_discovery": "pipelines_monitoring",
    "pipelines_monitoring": "pipelines_monitoring",
    "pipelines_runs": "pipelines_runs",
    "pipelines_lifecycle": "pipelines_runs",
}

SECTION_ORDER: list[str] = ["pipelines_monitoring", "pipelines_runs"]

INSTRUCTION_SECTIONS: dict[str, dict[str, str]] = {
    "pipelines_monitoring": {
        "full": """\
KUBEFLOW PIPELINES:
- list_pipelines() -> registered pipelines; get_pipeline(name) -> one pipeline + its versions
- list_pipeline_runs(pipeline=..., status="failed") -> find runs; pass next_page_token to page
- get_pipeline_run(run_id) -> state, error, parameters and task states
- get_pipeline_logs(run_id, task=...) -> output of a run's task pods (bounded)
- Run IDs are UUIDs; take them from list_pipeline_runs() or create_pipeline_run()
- Read pipelines://guides/troubleshooting for error-to-fix tables""",
    },
    "pipelines_runs": {
        "full": """\
PIPELINE RUNS:
- create_pipeline_run(pipeline, params=...) -> ALWAYS preview first (confirmed=False), show it, then confirmed=True
- Only registered pipelines run, by name. File paths are refused; register pipelines outside MCP
- stop_pipeline_run(run_id) / retry_pipeline_run(run_id) -> preview first, then confirmed=True
- KFP runs have no MCP ownership label: stop/retry act on any run in the namespace, so confirm the run with the user
- Read pipelines://guides/workflow-patterns for the end-to-end flow""",
    },
}


__all__ = [
    "MODULE_INFO",
    "TOOLS",
    "CLIENT_TOOL_DESCRIPTIONS",
    "CLIENT_TOOL_ANNOTATIONS",
    "CLIENT_RESOURCES",
    "INSTRUCTION_SECTIONS",
    "SECTION_ORDER",
    "PHASE_TO_SECTION",
    *[t.__name__ for t in TOOLS],
]
