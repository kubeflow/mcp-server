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

"""Optimizer client module — MCP tools for Katib hyperparameter optimization.

Structure mirrors kubeflow/optimizer/ and the trainer client module:
├── api/           # Tool implementations
├── types/         # Type definitions
├── constants/     # Constants (re-exported from SDK)
└── resources/     # MCP resources (guides, troubleshooting)
"""

from kubeflow_mcp.common.constants import KUBEFLOW_SDK_VERSION_SPEC
from kubeflow_mcp.common.types import ToolPermission
from kubeflow_mcp.optimizer.api.discovery import (
    get_hpo_experiment,
    get_hpo_experiment_status,
    get_successful_trials,
    get_trial,
    list_hpo_experiments,
    list_suggestions,
)
from kubeflow_mcp.optimizer.api.lifecycle import (
    delete_hpo_experiment,
    update_hpo_experiment,
)
from kubeflow_mcp.optimizer.api.monitoring import (
    get_best_trial,
    get_hpo_experiment_events,
    get_hpo_experiment_trial_logs,
    get_hpo_experiment_trials,
    get_suggestion,
    wait_for_hpo_experiment,
)
from kubeflow_mcp.optimizer.api.optimization import (
    create_hpo_experiment,
    create_hpo_experiment_from_spec,
)
from kubeflow_mcp.optimizer.api.planning import (
    katib_pre_flight,
)

MODULE_INFO = {
    "name": "optimizer",
    "description": "Hyperparameter optimization with Katib",
    "sdk_client": "kubeflow.optimizer.OptimizerClient",
    "sdk_version": KUBEFLOW_SDK_VERSION_SPEC,
    "status": "implemented",
    "planned_tools": 17,
}

TOOLS = [
    # Planning
    katib_pre_flight,
    # Discovery
    list_hpo_experiments,
    get_hpo_experiment,
    get_hpo_experiment_status,
    get_trial,
    get_successful_trials,
    list_suggestions,
    # Monitoring
    get_hpo_experiment_trials,
    get_best_trial,
    get_suggestion,
    wait_for_hpo_experiment,
    get_hpo_experiment_trial_logs,
    get_hpo_experiment_events,
    # Optimization
    create_hpo_experiment,
    create_hpo_experiment_from_spec,
    # Lifecycle
    delete_hpo_experiment,
    update_hpo_experiment,
]

# ─── Tool metadata (owned by this client module) ───────────────────────────

CLIENT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "katib_pre_flight": (
        "One-shot: Katib CRD + controller health + trainer availability. "
        "Call FIRST before any optimizer operations."
    ),
    "list_hpo_experiments": "List Katib experiments. Filter by status or namespace.",
    "get_hpo_experiment": (
        "Get full experiment details: spec, status, trials, search space, algorithm, best trial."
    ),
    "get_hpo_experiment_status": (
        "Lightweight status check — status string + trial counts only. "
        "Faster than get_hpo_experiment() for polling."
    ),
    "get_trial": (
        "Get one trial's parameters, metrics and status. "
        "Pass the experiment as `name` and the trial as `trial`."
    ),
    "get_successful_trials": (
        "Successful trials with hyperparameters and metrics for comparison. "
        "Returns up to `limit` (default 50); `total` reports the full count."
    ),
    "list_suggestions": "List suggestion resources for algorithm debugging.",
    "get_hpo_experiment_trials": (
        "List trials for an experiment with optional status filter. "
        "Returns up to `limit` (default 50); `total` reports the full count."
    ),
    "get_best_trial": "Best trial with optimal hyperparameters and metrics.",
    "get_suggestion": "Suggestion algorithm status and request/reply counts.",
    "wait_for_hpo_experiment": (
        "Block until experiment reaches terminal status (Complete/Failed). Timeout capped at 3600s."
    ),
    "get_hpo_experiment_trial_logs": (
        "Pod logs from a trial. If no trial specified, uses the best trial. "
        "Failure patterns auto-detected with hints."
    ),
    "get_hpo_experiment_events": ("K8s events for experiment. Debug scheduling/image pull issues."),
    "create_hpo_experiment": (
        "Create HPO experiment from flat params. Search types: uniform, "
        "loguniform, int, choice. Algorithms: random, grid, bayesianoptimization, "
        "tpe, multivariate-tpe, cmaes, sobol. Trial template: a TrainJob, batch/v1 Job "
        "or Kubeflow training job. Set confirmed=True to submit."
    ),
    "create_hpo_experiment_from_spec": (
        "Create experiment from raw V1beta1Experiment JSON spec. "
        "Escape hatch for advanced configs. Set confirmed=True to submit."
    ),
    "delete_hpo_experiment": (
        "[DESTRUCTIVE] Delete experiment permanently. Set confirmed=True to execute."
    ),
    "update_hpo_experiment": (
        "[DESTRUCTIVE] Change a running experiment's parallel_trial_count, max_trial_count "
        "or max_failed_trial_count. Katib cannot pause: lowering parallel_trial_count "
        "makes Katib delete trials, newest first and whatever their state, until the "
        "active ones fit (the preview lists them). Set confirmed=True to apply."
    ),
}

CLIENT_TOOL_ANNOTATIONS: dict[str, dict] = {
    "katib_pre_flight": {
        "title": "Katib Pre-flight Check",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["planning", "preflight", "katib", "compound"],
    },
    "list_hpo_experiments": {
        "title": "List Experiments",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "experiments"],
    },
    "get_hpo_experiment": {
        "title": "Get Experiment Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "monitoring", "experiments"],
    },
    "get_hpo_experiment_status": {
        "title": "Get Experiment Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "monitoring", "experiments"],
    },
    "get_trial": {
        "title": "Get Trial Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "trials"],
    },
    "get_successful_trials": {
        "title": "Get Successful Trials",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "trials", "results"],
    },
    "list_suggestions": {
        "title": "List Suggestions",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "suggestions", "debug"],
    },
    "get_hpo_experiment_trials": {
        "title": "Get Experiment Trials",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "trials"],
    },
    "get_best_trial": {
        "title": "Get Best Trial",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "results", "optimization"],
    },
    "get_suggestion": {
        "title": "Get Suggestion Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "suggestions", "debug"],
    },
    "wait_for_hpo_experiment": {
        "title": "Wait for Experiment",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "blocking"],
    },
    "get_hpo_experiment_trial_logs": {
        "title": "Get Trial Logs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "logs", "debug"],
    },
    "get_hpo_experiment_events": {
        "title": "Get Experiment Events",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "events", "debug"],
    },
    "create_hpo_experiment": {
        "title": "Create HPO Experiment",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["optimization", "hpo", "create"],
    },
    "create_hpo_experiment_from_spec": {
        "title": "Create Experiment from Spec",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["optimization", "advanced", "create"],
    },
    "delete_hpo_experiment": {
        "title": "Delete Experiment",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["lifecycle", "cleanup"],
    },
    "update_hpo_experiment": {
        "title": "Update Experiment",
        "readOnlyHint": False,
        # Lowering parallel_trial_count makes Katib delete trials, completed
        # ones included, so policies denying risk:destructive must remove it.
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["lifecycle", "scale", "budget"],
    },
}

# ─── Resources (owned by this client module) ───────────────────────────────

CLIENT_RESOURCES: dict[str, tuple[str, str]] = {
    "optimizer://guides/hpo-patterns": (
        "resources/hpo-patterns.md",
        "HPO workflow patterns and algorithm selection guide.",
    ),
    "optimizer://guides/troubleshooting": (
        "resources/troubleshooting.md",
        "Common Katib errors and fixes.",
    ),
}

# ─── Instruction sections (full tier; compact/minimal auto-derived) ────────

PHASE_TO_SECTION: dict[str, str | None] = {
    "optimizer_planning": "optimizer_planning",
    "optimizer_discovery": None,
    "optimizer_optimization": "optimizer_optimization",
    "optimizer_monitoring": "optimizer_monitoring",
    "optimizer_lifecycle": "optimizer_lifecycle",
}

# Canonical ordering of this client's instruction sections.  server.py
# concatenates SECTION_ORDER from every loaded client to build a stable
# global preferred order, with no hardcoded list in the core. Names are
# namespaced so they cannot collide with the trainer's sections.
SECTION_ORDER: list[str] = [
    "optimizer_planning",
    "optimizer_optimization",
    "optimizer_monitoring",
    "optimizer_lifecycle",
]

INSTRUCTION_SECTIONS: dict[str, dict[str, str]] = {
    "optimizer_planning": {
        "full": """\
OPTIMIZER PLANNING (always do first):
- katib_pre_flight() → One call: Katib CRD + controller health + trainer availability
- If blockers returned, STOP and inform user
- If trainer_sdk_available=false → the Trainer SDK cannot start, so cross-client workflows are unavailable
OPTIMIZER DISCOVERY (before creating experiments):
- list_hpo_experiments() → find existing experiments in namespace
- get_hpo_experiment(name) → inspect spec, status, trials, best trial
- get_hpo_experiment_status(name) → lightweight status poll
- get_successful_trials(name) → compare successful trials' hyperparameters. Returns up to
  limit (default 50, at most 500); if total is larger, the list is not complete""",
    },
    "optimizer_monitoring": {
        "full": """\
OPTIMIZER MONITORING:
- get_hpo_experiment_trials(name) → trials with status and parameters, up to limit
  (default 50, at most 500); if total is larger, the list is not complete
- get_best_trial(name) → optimal hyperparameters and metrics from best trial
- get_hpo_experiment_trial_logs(name) → pod logs with failure pattern detection
- get_hpo_experiment_events(name) → K8s events for debugging scheduling/image issues
- get_suggestion(name) → algorithm status and request/reply counts""",
    },
    # Separate from monitoring so personas without these tools (readonly) are
    # not told to call them.
    "optimizer_lifecycle": {
        "full": """OPTIMIZER LIFECYCLE:
- wait_for_hpo_experiment(name) → block until Complete/Failed (timeout capped at 3600s)
- delete_hpo_experiment(name) → preview, then confirmed=True to remove it permanently
- update_hpo_experiment(name, max_trial_count=N) → preview a budget or
  parallel_trial_count change, show it, then repeat with confirmed=True after approval.
  Katib cannot pause; lowering parallelism deletes trials, newest first and whatever
  their state, so show trials_to_delete from the preview before confirming""",
    },
    "optimizer_optimization": {
        "full": """\
OPTIMIZER TOOL SELECTION:
- HPO with flat params → create_hpo_experiment() — specify objective, search space, algorithm
  - Search types: uniform (continuous), loguniform (log-scale), int (discrete range),
    choice (categorical)
  - Prefer int over choice for ordered whole numbers: categorical values carry no
    ordering, so the algorithm cannot tell that 4 lies between 2 and 8
  - Algorithms: random (default), grid, bayesianoptimization, tpe, multivariate-tpe,
    cmaes, sobol (hyperband needs algorithm settings: use the spec tool)
  - trial_template is the Katib trialSpec (a TrainJob, batch/v1 Job or Kubeflow training
    job); reference every searched param as ${trialParameters.<name>}
  - primary_container_name is derived from trial_template and must name a container
    every replica has; set it only when the container Katib should collect metrics
    from is not the one it picks
  - ALWAYS preview first (confirmed=False), then show preview and wait for approval
- Raw V1beta1Experiment spec → create_hpo_experiment_from_spec() — escape hatch for
  early stopping, custom metrics collectors, resume policies and NAS

OPTIMIZATION RULES:
- ALWAYS call katib_pre_flight() before creating experiments
- ALWAYS preview before submitting (confirmed=False first)
- max_trial_count is capped at 1000 and parallel_trial_count at 100
- Use get_hpo_experiment_events() to debug stuck/failed experiments
- Read optimizer://guides/hpo-patterns for algorithm selection guide""",
    },
}


# Kubernetes permissions each tool's requests need (#565). Katib's resources are in
# the kubeflow.org group. Not declared: katib_pre_flight's controller Pod lookup, its
# dry-run create that probes the admission webhooks, and its Trainer client check;
# each only adds a warning when it is denied.
_KATIB = "kubeflow.org"
_EXPERIMENT_GET = ToolPermission("get", _KATIB, "experiments", name="name")
_TRIALS_LIST = ToolPermission("list", _KATIB, "trials")
_EXPERIMENT_CREATE = ToolPermission("create", _KATIB, "experiments")

CLIENT_TOOL_PERMISSIONS: dict[str, tuple[ToolPermission, ...]] = {
    "katib_pre_flight": (
        ToolPermission(
            "get", "apiextensions.k8s.io", "customresourcedefinitions", cluster_wide=True
        ),
    ),
    "list_hpo_experiments": (ToolPermission("list", _KATIB, "experiments"),),
    "get_hpo_experiment": (_EXPERIMENT_GET, _TRIALS_LIST),
    "get_hpo_experiment_status": (_EXPERIMENT_GET,),
    "get_trial": (_EXPERIMENT_GET, _TRIALS_LIST),
    "get_successful_trials": (_EXPERIMENT_GET, _TRIALS_LIST),
    "list_suggestions": (ToolPermission("list", _KATIB, "suggestions"),),
    "get_hpo_experiment_trials": (_EXPERIMENT_GET, _TRIALS_LIST),
    "get_best_trial": (_EXPERIMENT_GET,),
    # The Suggestion shares the experiment's name.
    "get_suggestion": (ToolPermission("get", _KATIB, "suggestions", name="name"),),
    "wait_for_hpo_experiment": (_EXPERIMENT_GET,),
    "get_hpo_experiment_trial_logs": (
        _EXPERIMENT_GET,
        _TRIALS_LIST,
        ToolPermission("list", "", "pods"),
        ToolPermission("get", "", "pods/log"),
    ),
    "get_hpo_experiment_events": (
        _EXPERIMENT_GET,
        _TRIALS_LIST,
        ToolPermission("list", "", "pods"),
        ToolPermission("list", "", "events"),
    ),
    "create_hpo_experiment": (_EXPERIMENT_CREATE,),
    "create_hpo_experiment_from_spec": (_EXPERIMENT_CREATE,),
    "delete_hpo_experiment": (
        _EXPERIMENT_GET,  # ownership check
        ToolPermission("delete", _KATIB, "experiments", name="name"),
    ),
    # Trials are listed only when parallel_trial_count is lowered, to warn which
    # running trials Katib will delete; it is checked whenever the argument is set.
    "update_hpo_experiment": (
        _EXPERIMENT_GET,
        ToolPermission("list", _KATIB, "trials", when="parallel_trial_count"),
        ToolPermission("patch", _KATIB, "experiments", name="name"),
    ),
}

__all__ = [
    "MODULE_INFO",
    "TOOLS",
    "CLIENT_TOOL_DESCRIPTIONS",
    "CLIENT_TOOL_ANNOTATIONS",
    "CLIENT_TOOL_PERMISSIONS",
    "CLIENT_RESOURCES",
    "INSTRUCTION_SECTIONS",
    "PHASE_TO_SECTION",
    "SECTION_ORDER",
    *[t.__name__ for t in TOOLS],
]
