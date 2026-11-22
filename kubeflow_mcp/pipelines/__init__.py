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
so agents can register pipelines from KFP code, manage experiments, start runs and
follow them. Structure mirrors ``trainer/``:

├── api/       # Tool implementations (discovery, monitoring, runs, experiments, authoring)
├── types/     # SDK-object serialization helpers
└── resources/ # Agent-facing guides

``kfp`` ships behind the optional ``kubeflow[pipelines]`` extra and every SDK import
is lazy, so this module and its metadata load without it.
"""

from kubeflow_mcp.common.constants import KUBEFLOW_SDK_VERSION_SPEC
from kubeflow_mcp.pipelines.api.authoring import create_pipeline, delete_pipeline
from kubeflow_mcp.pipelines.api.discovery import get_pipeline, list_pipelines
from kubeflow_mcp.pipelines.api.experiments import (
    create_experiment,
    delete_experiment,
    get_experiment,
    list_experiments,
)
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
    "description": "Kubeflow Pipelines: pipelines, experiments and runs",
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
    list_experiments,
    get_experiment,
    create_experiment,
    delete_experiment,
    create_pipeline,
    delete_pipeline,
]

# ─── Tool metadata (owned by this client module) ───────────────────────────

CLIENT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "list_pipelines": (
        "List registered pipelines (the blueprints, not their runs). Paged with page_token."
    ),
    "get_pipeline": (
        "Get a registered pipeline (the blueprint) by name, with its recent versions. "
        "For a run of it, use get_pipeline_run(run_id)."
    ),
    "list_pipeline_runs": (
        "List pipeline runs (executions). Filter by pipeline, experiment or status (e.g. failed)."
    ),
    "get_pipeline_run": (
        "Get one run (an execution of a pipeline) by its run_id: state, error, "
        "parameters, task states, created_at and finished_at."
    ),
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
    "list_experiments": "List experiments (groups of pipeline runs). Paged with page_token.",
    "get_experiment": "Get an experiment by name.",
    "create_experiment": (
        "Create an experiment to group runs; returns the existing one if the name is taken. "
        "Preview first (confirmed=False), then set confirmed=True."
    ),
    "delete_experiment": (
        "[DESTRUCTIVE] Delete an experiment (its runs are kept). Preview first "
        "(confirmed=False), then set confirmed=True."
    ),
    "create_pipeline": (
        "Register a pipeline from KFP v2 Python code (@dsl.component functions and one "
        "@dsl.pipeline). Optional dataset= is checked like fine_tune's. Preview first "
        "(confirmed=False) to check the code, then confirmed=True to compile and upload it. "
        "Read pipelines://guides/authoring before writing the code."
    ),
    "delete_pipeline": (
        "[DESTRUCTIVE] Delete a registered pipeline with all its versions, or one "
        "version (version=). Runs are kept. Preview first (confirmed=False), then set "
        "confirmed=True."
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
    "list_experiments": {
        "title": "List Experiments",
        **_READ,
        "tags": ["pipelines_discovery", "pipelines", "experiments"],
    },
    "get_experiment": {
        "title": "Get Experiment",
        **_READ,
        "tags": ["pipelines_discovery", "pipelines", "experiments"],
    },
    "create_experiment": {
        "title": "Create Experiment",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["pipelines_runs", "pipelines", "experiments", "create"],
    },
    "delete_experiment": {
        "title": "Delete Experiment",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["pipelines_lifecycle", "pipelines", "experiments"],
    },
    "create_pipeline": {
        "title": "Create Pipeline",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["pipelines_authoring", "pipelines", "create"],
    },
    "delete_pipeline": {
        "title": "Delete Pipeline",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["pipelines_lifecycle", "pipelines"],
    },
}

# ─── Resources (owned by this client module) ───────────────────────────────

CLIENT_RESOURCES: dict[str, tuple[str, str]] = {
    "pipelines://guides/workflow-patterns": (
        "resources/workflow-patterns.md",
        "Find a pipeline, start a run, follow it and debug a failed run.",
    ),
    "pipelines://guides/authoring": (
        "resources/authoring.md",
        "How to write pipeline code create_pipeline accepts, with a train-on-a-dataset example.",
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
    "pipelines_authoring": "pipelines_authoring",
}

SECTION_ORDER: list[str] = ["pipelines_monitoring", "pipelines_runs", "pipelines_authoring"]

INSTRUCTION_SECTIONS: dict[str, dict[str, str]] = {
    "pipelines_monitoring": {
        "full": """\
KUBEFLOW PIPELINES:
- list_pipelines() -> registered pipelines; get_pipeline(name) -> one pipeline + its versions
- list_pipeline_runs(pipeline=..., status="failed") -> find runs; pass next_page_token to page
- get_pipeline_run(run_id) -> state, error, parameters and task states
- get_pipeline_logs(run_id, task=...) -> output of a run's task pods (bounded)
- list_experiments() / get_experiment(name) -> experiments that group runs
- Run IDs are UUIDs; take them from list_pipeline_runs() or create_pipeline_run()
- Read pipelines://guides/troubleshooting for error-to-fix tables""",
    },
    "pipelines_runs": {
        "full": """\
PIPELINE RUNS:
- create_pipeline_run(pipeline, params=...) -> ALWAYS preview first (confirmed=False), show it, then confirmed=True
- Only registered pipelines run, by name. File paths are refused
- create_experiment(name) -> preview first; returns the existing experiment if the name is taken
- stop_pipeline_run(run_id) / retry_pipeline_run(run_id) -> preview first, then confirmed=True
- delete_experiment(name) -> preview first; the experiment's runs are kept
- delete_pipeline(name, version=...) -> preview first; platform-admin only; runs are kept
- KFP runs have no MCP ownership label: stop/retry act on any run in the namespace, so confirm the run with the user
- Read pipelines://guides/workflow-patterns for the end-to-end flow""",
    },
    "pipelines_authoring": {
        "full": """PIPELINE AUTHORING (prompt -> pipeline -> run):
- Read pipelines://guides/authoring before writing pipeline code
- Write KFP v2 code: 'from kfp import dsl', @dsl.component functions, one @dsl.pipeline
- Put imports of other packages inside components; they run in the cluster, not on the server
- create_pipeline(name, pipeline_code, dataset=...) -> preview checks the code, confirmed=True compiles and uploads
- Fix the code and preview again when the check or the compile reports an error
- Then create_experiment(name) and create_pipeline_run(pipeline=name, experiment=..., params=...)""",
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
