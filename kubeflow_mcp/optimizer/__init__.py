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

from kubeflow_mcp.optimizer.api.discovery import (
    get_experiment,
    get_experiment_status,
    get_successful_trials,
    get_trial,
    list_experiments,
    list_suggestions,
)
from kubeflow_mcp.optimizer.api.lifecycle import (
    delete_experiment,
    update_experiment,
)
from kubeflow_mcp.optimizer.api.monitoring import (
    get_best_trial,
    get_experiment_events,
    get_experiment_trial_logs,
    get_experiment_trials,
    get_suggestion,
    wait_for_experiment,
)
from kubeflow_mcp.optimizer.api.optimization import (
    create_experiment_from_spec,
    create_hpo_experiment,
)
from kubeflow_mcp.optimizer.api.planning import (
    katib_pre_flight,
)

MODULE_INFO = {
    "name": "optimizer",
    "description": "Hyperparameter optimization with Katib",
    "sdk_client": "kubeflow.optimizer.OptimizerClient",
    "sdk_version": ">=0.4.0",
    "status": "implemented",
    "planned_tools": 17,
}

TOOLS = [
    # Planning
    katib_pre_flight,
    # Discovery
    list_experiments,
    get_experiment,
    get_experiment_status,
    get_trial,
    get_successful_trials,
    list_suggestions,
    # Monitoring
    get_experiment_trials,
    get_best_trial,
    get_suggestion,
    wait_for_experiment,
    get_experiment_trial_logs,
    get_experiment_events,
    # Optimization
    create_hpo_experiment,
    create_experiment_from_spec,
    # Lifecycle
    delete_experiment,
    update_experiment,
]

# ─── Tool metadata (owned by this client module) ───────────────────────────

CLIENT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "katib_pre_flight": (
        "One-shot: Katib CRD + controller health + trainer availability. "
        "Call FIRST before any optimizer operations."
    ),
    "list_experiments": "List Katib experiments. Filter by status or namespace.",
    "get_experiment": (
        "Get full experiment details: spec, status, trials, search space, algorithm, best trial."
    ),
    "get_experiment_status": (
        "Lightweight status check — status string + trial counts only. "
        "Faster than get_experiment() for polling."
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
    "get_experiment_trials": ("List trials for an experiment with optional status filter."),
    "get_best_trial": "Best trial with optimal hyperparameters and metrics.",
    "get_suggestion": "Suggestion algorithm status and request/reply counts.",
    "wait_for_experiment": (
        "Block until experiment reaches terminal status (Complete/Failed). Timeout capped at 3600s."
    ),
    "get_experiment_trial_logs": (
        "Pod logs from a trial. If no trial specified, uses the best trial. "
        "Failure patterns auto-detected with hints."
    ),
    "get_experiment_events": ("K8s events for experiment. Debug scheduling/image pull issues."),
    "create_hpo_experiment": (
        "Create HPO experiment from flat params. Search types: uniform, "
        "loguniform, int, choice. Algorithms: random, grid, bayesianoptimization, "
        "tpe, multivariate-tpe, cmaes, sobol, hyperband. "
        "Set confirmed=True to submit."
    ),
    "create_experiment_from_spec": (
        "Create experiment from raw V1beta1Experiment JSON spec. "
        "Escape hatch for advanced configs. Set confirmed=True to submit."
    ),
    "delete_experiment": (
        "[DESTRUCTIVE] Delete experiment permanently. Set confirmed=True to execute."
    ),
    "update_experiment": ("Suspend or resume experiment. Pass action='suspend' or 'resume'."),
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
    "list_experiments": {
        "title": "List Experiments",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "experiments"],
    },
    "get_experiment": {
        "title": "Get Experiment Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "monitoring", "experiments"],
    },
    "get_experiment_status": {
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
    "get_experiment_trials": {
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
    "wait_for_experiment": {
        "title": "Wait for Experiment",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "blocking"],
    },
    "get_experiment_trial_logs": {
        "title": "Get Trial Logs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "logs", "debug"],
    },
    "get_experiment_events": {
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
    "create_experiment_from_spec": {
        "title": "Create Experiment from Spec",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["optimization", "advanced", "create"],
    },
    "delete_experiment": {
        "title": "Delete Experiment",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["lifecycle", "cleanup"],
    },
    "update_experiment": {
        "title": "Update Experiment",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["lifecycle", "suspend", "resume"],
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
    "optimizer_lifecycle": "optimizer_monitoring",
}

# Canonical ordering of this client's instruction sections.  server.py
# concatenates SECTION_ORDER from every loaded client to build a stable
# global preferred order, with no hardcoded list in the core. Names are
# namespaced so they cannot collide with the trainer's sections.
SECTION_ORDER: list[str] = [
    "optimizer_planning",
    "optimizer_optimization",
    "optimizer_monitoring",
]

INSTRUCTION_SECTIONS: dict[str, dict[str, str]] = {
    "optimizer_planning": {
        "full": """\
OPTIMIZER PLANNING (always do first):
- katib_pre_flight() → One call: Katib CRD + controller health + trainer availability
- If blockers returned, STOP and inform user
- If trainer_available=false → cross-client workflows require --clients trainer,optimizer
OPTIMIZER DISCOVERY (before creating experiments):
- list_experiments() → find existing experiments in namespace
- get_experiment(name) → inspect spec, status, trials, best trial
- get_experiment_status(name) → lightweight status poll
- get_successful_trials(name) → compare all successful trial hyperparameters""",
    },
    "optimizer_monitoring": {
        "full": """\
OPTIMIZER MONITORING AND LIFECYCLE:
- get_experiment_trials(name) → list all trials with status and parameters
- get_best_trial(name) → optimal hyperparameters and metrics from best trial
- get_experiment_trial_logs(name) → pod logs with failure pattern detection
- get_experiment_events(name) → K8s events for debugging scheduling/image issues
- wait_for_experiment(name) → block until Complete/Failed (timeout capped at 3600s)
- get_suggestion(name) → algorithm status and request/reply counts
- delete_experiment(name, confirmed=True) → remove permanently (preview first)
- update_experiment(name, action="suspend"|"resume") → pause/resume experiment""",
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
    cmaes, sobol, hyperband
  - trial_template is the Katib trialSpec; reference params as ${trialParameters.<name>}
  - primary_container_name is derived from trial_template; set it only when the
    container Katib should collect metrics from is not the one it picks
  - ALWAYS preview first (confirmed=False), then show preview and wait for approval
- Raw V1beta1Experiment spec → create_experiment_from_spec() — escape hatch for
  early stopping, custom metrics collectors, resume policies and NAS

OPTIMIZATION RULES:
- ALWAYS call katib_pre_flight() before creating experiments
- ALWAYS preview before submitting (confirmed=False first)
- max_trial_count is capped at 1000 and parallel_trial_count at 100
- Use get_experiment_events() to debug stuck/failed experiments
- Read optimizer://guides/hpo-patterns for algorithm selection guide""",
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
    "SECTION_ORDER",
    *[t.__name__ for t in TOOLS],
]
