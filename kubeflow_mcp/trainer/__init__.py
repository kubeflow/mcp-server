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

"""Trainer client module — MCP tools for distributed training and LLM fine-tuning.

Structure mirrors kubeflow/trainer/:
├── api/           # Tool implementations
├── types/         # Type definitions
└── constants/     # Constants
"""

from kubeflow_mcp.core.health import get_server_logs, health_check
from kubeflow_mcp.trainer.api.discovery import (
    get_runtime,
    get_training_job,
    list_runtimes,
    list_training_jobs,
)
from kubeflow_mcp.trainer.api.lifecycle import (
    delete_training_job,
    update_training_job,
)
from kubeflow_mcp.trainer.api.monitoring import (
    get_training_events,
    get_training_logs,
    wait_for_training,
)
from kubeflow_mcp.trainer.api.planning import (
    check_compatibility,
    estimate_resources,
    get_cluster_resources,
    pre_flight,
)
from kubeflow_mcp.trainer.api.platform import (
    create_runtime,
    delete_runtime,
    inspect_controller,
    inspect_crd,
    patch_runtime,
)
from kubeflow_mcp.trainer.api.training import (
    fine_tune,
    run_container_training,
    run_custom_training,
)

MODULE_INFO = {
    "name": "trainer",
    "description": "Distributed training and LLM fine-tuning on Kubernetes",
    "status": "implemented",
}

TOOLS = [
    pre_flight,
    check_compatibility,
    get_cluster_resources,
    estimate_resources,
    fine_tune,
    run_custom_training,
    run_container_training,
    list_training_jobs,
    get_training_job,
    list_runtimes,
    get_runtime,
    get_training_logs,
    get_training_events,
    wait_for_training,
    delete_training_job,
    update_training_job,
    inspect_crd,
    inspect_controller,
    patch_runtime,
    create_runtime,
    delete_runtime,
    health_check,
    get_server_logs,
]

# ─── Tool metadata (owned by this client module) ───────────────────────────

CLIENT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "pre_flight": (
        "One-shot: compatibility + cluster resources + model estimate + runtimes. "
        "Call FIRST. Pass model= for GPU sizing."
    ),
    "check_compatibility": "Verify K8s version, Trainer CRD, installed packages, and platform. Use pre_flight() for full check.",
    "get_cluster_resources": "Check cluster GPU/CPU availability. Use pre_flight() instead for full check.",
    "estimate_resources": "Estimate GPU memory needed for a HuggingFace model. Use pre_flight(model=...) instead.",
    "list_training_jobs": "List training jobs. Filter by runtime, status, or namespace.",
    "get_training_job": "Get details of a specific training job. Supports optional namespace.",
    "list_runtimes": "List available ClusterTrainingRuntimes.",
    "get_runtime": "Get runtime config. Pass include_packages=True to fetch pip list (slow: creates a Pod).",
    "fine_tune": (
        "Fine-tune HuggingFace model with LoRA. Run list_runtimes() first to find the "
        "correct runtime name. Optional name= for custom job name. Set confirmed=True to submit."
    ),
    "run_custom_training": (
        "Run Python training script on the cluster. Pass runtime= for runtime selection. "
        "Set confirmed=True to submit."
    ),
    "run_container_training": (
        "Run training with custom container image. Pass runtime= and command= to "
        "override runtime and entrypoint. Set confirmed=True to submit."
    ),
    "get_training_logs": "Get pod logs from a training job. Supports optional namespace.",
    "get_training_events": "Get K8s events for debugging pending/failed jobs. Supports optional namespace.",
    "wait_for_training": "Block until job reaches target status (Complete/Failed). Supports optional namespace.",
    "delete_training_job": "[DESTRUCTIVE] Delete a training job permanently. Set confirmed=True to execute.",
    "update_training_job": "Suspend or resume a training job. Pass action='suspend' or 'resume'.",
    "inspect_crd": "List Trainer CRDs or get details for a specific one. Pass name= for details.",
    "inspect_controller": "Inspect controller pod. Pass view='logs' or 'events'. Auto-discovers namespace.",
    "patch_runtime": "Strategic merge patch on a ClusterTrainingRuntime. Set confirmed=True to apply.",
    "create_runtime": "Create a new ClusterTrainingRuntime. Set confirmed=True to create.",
    "delete_runtime": "[DESTRUCTIVE] Delete a ClusterTrainingRuntime. Lists dependent jobs first. Set confirmed=True.",
    "health_check": "Check server health and K8s connectivity.",
    "get_server_logs": "Get recent server logs for debugging. Filter by level.",
}

CLIENT_TOOL_ANNOTATIONS: dict[str, dict] = {
    "pre_flight": {
        "title": "Pre-flight Environment Check",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["planning", "preflight", "compound"],
    },
    "check_compatibility": {
        "title": "Check Environment Compatibility",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["planning", "preflight", "compatibility"],
    },
    "get_cluster_resources": {
        "title": "Get Cluster Resources",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["planning", "cluster", "gpu"],
    },
    "estimate_resources": {
        "title": "Estimate Training Resources",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["planning", "resources", "estimation"],
    },
    "list_training_jobs": {
        "title": "List Training Jobs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "jobs"],
    },
    "get_training_job": {
        "title": "Get Training Job Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "monitoring", "jobs"],
    },
    "list_runtimes": {
        "title": "List Training Runtimes",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "runtimes"],
    },
    "get_runtime": {
        "title": "Get Runtime Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "runtimes"],
    },
    "fine_tune": {
        "title": "Fine-tune Model",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["training", "fine-tuning", "llm"],
    },
    "run_custom_training": {
        "title": "Run Custom Training Script",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["training", "custom", "script"],
    },
    "run_container_training": {
        "title": "Run Container Training",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["training", "container"],
    },
    "get_training_logs": {
        "title": "Get Training Logs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "logs", "debug"],
    },
    "get_training_events": {
        "title": "Get Training Events",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "events", "debug"],
    },
    "wait_for_training": {
        "title": "Wait for Training Completion",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "blocking"],
    },
    "delete_training_job": {
        "title": "Delete Training Job",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["lifecycle", "cleanup"],
    },
    "update_training_job": {
        "title": "Update Training Job",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["lifecycle", "suspend", "resume"],
    },
    "inspect_crd": {
        "title": "Inspect Trainer CRDs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["platform", "crd"],
    },
    "inspect_controller": {
        "title": "Inspect Controller",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["platform", "logs", "events", "debug"],
    },
    "patch_runtime": {
        "title": "Patch Training Runtime",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["platform", "runtime", "admin"],
    },
    "create_runtime": {
        "title": "Create Training Runtime",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["platform", "runtime", "admin"],
    },
    "delete_runtime": {
        "title": "Delete Training Runtime",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["platform", "runtime", "admin"],
    },
    "health_check": {
        "title": "Health Check",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
        "tags": ["health", "monitoring"],
    },
    "get_server_logs": {
        "title": "Get Server Logs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
        "tags": ["health", "monitoring", "debug"],
    },
}

# ─── Resources (owned by this client module) ───────────────────────────────

CLIENT_RESOURCES: dict[str, tuple[str, str]] = {
    "trainer://guides/training-patterns": (
        "resources/training-patterns.md",
        "Distributed training and LoRA code patterns.",
    ),
    "trainer://guides/platform-fixes": (
        "resources/platform-fixes.md",
        "Platform-specific volume and toleration guidance.",
    ),
    "trainer://guides/troubleshooting": (
        "resources/troubleshooting.md",
        "Error-to-fix tables, diagnostics, known limitations.",
    ),
}

# ─── Instruction sections (full tier; compact/minimal auto-derived) ────────
# Phase-to-section mapping for auto-deriving which sections a persona needs.
# TOOL_PHASES keys -> instruction section names.

PHASE_TO_SECTION: dict[str, str | None] = {
    "planning": "planning",
    "discovery": None,
    "training": "training",
    "monitoring": "monitoring",
    "lifecycle": "monitoring",
    "platform": "platform",
    "health": None,
}

INSTRUCTION_SECTIONS: dict[str, dict[str, str]] = {
    "planning": {
        "full": """\
PLANNING (always do first):
- pre_flight(model="<model>") -> One call: compatibility + cluster + estimate + runtimes
- If blockers returned, STOP and inform user
- If gpu_total=0 -> fine_tune() will NOT work (torchtune needs GPUs). Use run_custom_training() with gloo backend instead
- Individual tools (check_compatibility, get_cluster_resources, estimate_resources) available for targeted re-checks
DISCOVERY (before training):
- list_runtimes() -> find available runtimes; ALWAYS pass the exact runtime name to training tools
- get_runtime(name) -> inspect runtime spec, container images, default config
- get_runtime(name, include_packages=True) -> list installed packages (slow: ~30-60s, creates temporary Pod). Call BEFORE choosing packages to avoid version conflicts""",
    },
    "monitoring": {
        "full": """\
MONITORING AND LIFECYCLE:
- get_training_job(name) -> check status (Created/Running/Complete/Failed/Suspended)
- get_training_logs(name) -> view output/errors. Failure patterns auto-detected with hints
- get_training_events(name) -> debug scheduling issues, pending pods, image pull errors
- wait_for_training(name) -> block until Complete/Failed (caution: blocks MCP connection)
- All monitoring tools accept optional namespace= to query jobs in a different namespace
- Suspended jobs show status "Created" in the API — this is a known controller behavior
- delete_training_job(name, confirmed=True) -> remove job permanently (preview first, non-admin personas can only delete MCP-created jobs)
- update_training_job(name, action="suspend"|"resume") -> pause/resume without deleting (non-admin personas can only modify MCP-created jobs)""",
    },
    "training": {
        "full": """\
TOOL SELECTION:
- HuggingFace model fine-tuning -> fine_tune(runtime=..., name=...) — requires a torchtune runtime AND GPUs (NCCL backend)
- No torchtune runtime or no GPUs? -> use run_custom_training() with gloo backend. Read trainer://guides/training-patterns for LoRA patterns
- Custom Python training script -> run_custom_training(runtime=...) — defaults to torch-distributed, works on CPU or GPU
- Pre-built container image -> run_container_training(runtime=..., command=...) — supports command and args override
- Runtime names encode the model family (e.g. llama runtime will fail on a qwen model)

TRAINING RULES:
- ALWAYS preview before submitting (confirmed=False first), then show preview to user and wait for approval
- fine_tune() does NOT support env parameter — if env vars are needed, use run_custom_training() instead
- run_custom_training() and run_container_training() support env as a direct parameter: env={"KEY": "VALUE"}
- URI formats: fine_tune() requires hf:// prefix for model/dataset; estimate_resources() uses bare model IDs
- Volume scope: fine_tune() volumes apply to ALL replicated jobs (node, dataset-initializer, model-initializer). Do NOT add a workspace emptyDir — /workspace is provided by the runtime PVC
- run_custom_training() auto-injects a workspace emptyDir when volumes are provided
- All training tools accept tolerations, node_selector, volumes as direct parameters
- Read trainer://guides/training-patterns for distributed training code patterns

PLATFORM:
- If pre_flight() returns platform=openshift, ALWAYS pass emptyDir volumes for /.local, /.cache, /tmp — without these, jobs fail on read-only filesystem. Read trainer://guides/platform-fixes for copy-paste JSON
- Gated HuggingFace models require hf_token parameter
- Training jobs consume GPU resources — be conservative with num_nodes
- Use get_training_events() to debug stuck/failed jobs. Read trainer://guides/troubleshooting for error-to-fix tables""",
    },
    "platform": {
        "full": """\
PLATFORM ADMINISTRATION:
- inspect_crd() -> list all Trainer CRDs; inspect_crd(name) -> get schema, versions, conditions
- inspect_controller(view="logs") -> read trainer-controller-manager pod logs (auto-discovers namespace; override via KUBEFLOW_MCP_CONTROLLER_NAMESPACE)
- inspect_controller(view="events") -> K8s events for controller pod
- patch_runtime(name, patch, confirmed=True) -> strategic merge patch on ClusterTrainingRuntime (update images, add volumes, change defaults)
- create_runtime(name, spec, confirmed=True) -> create new ClusterTrainingRuntime
- delete_runtime(name, confirmed=True) -> delete runtime; lists dependent TrainJobs as warning before deletion""",
    },
}


__all__ = [
    "MODULE_INFO",
    "TOOLS",
    "CLIENT_TOOL_DESCRIPTIONS",
    "CLIENT_TOOL_ANNOTATIONS",
    "CLIENT_RESOURCES",
    "INSTRUCTION_SECTIONS",
    "PHASE_TO_SECTION",
    *[t.__name__ for t in TOOLS],
]
