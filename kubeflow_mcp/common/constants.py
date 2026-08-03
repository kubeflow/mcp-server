# Copyright The Kubeflow Authors
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

"""Centralized constants for the kubeflow-mcp server.

This module is the single source of truth for:
- Error codes and job statuses
- Tool phase categorization

Import from here to ensure consistency across the codebase.
"""

from typing import Any


class ErrorCode:
    """Standard error codes for tool responses."""

    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    KUBERNETES_ERROR = "KUBERNETES_ERROR"
    SDK_ERROR = "SDK_ERROR"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    TIMEOUT = "TIMEOUT"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    RATE_LIMITED = "RATE_LIMITED"


def is_infrastructure_error(result: Any) -> bool:
    """Return True if a tool result indicates a K8s/SDK infrastructure failure.

    Only these should trip the circuit breaker — user validation errors
    (VALIDATION_ERROR, RESOURCE_NOT_FOUND) are not infrastructure problems.
    """
    if not isinstance(result, dict):
        return False
    code = result.get("error_code", "")
    return code in (ErrorCode.KUBERNETES_ERROR, ErrorCode.SDK_ERROR, ErrorCode.TIMEOUT)


class JobStatus:
    """TrainJob status strings (Kubeflow Trainer SDK / CR conditions).

    Pod phase ``Succeeded`` is not the same as TrainJob ``Complete``; use
    :attr:`POD_SUCCEEDED` only when referring to pod-level state.
    """

    CREATED = "Created"
    RUNNING = "Running"
    COMPLETE = "Complete"
    FAILED = "Failed"
    SUSPENDED = "Suspended"
    POD_SUCCEEDED = "Succeeded"


# =============================================================================
# Tool Phase Categories
# Maps tools to their workflow phase for tagging and discovery.
# Used by: server.py (TOOL_ANNOTATIONS), dynamic_tools.py (TOOL_HIERARCHY)
# =============================================================================

TOOL_PHASES: dict[str, list[str]] = {
    # ── Trainer phases ──
    "planning": [
        "pre_flight",
        "check_compatibility",
        "get_cluster_resources",
        "estimate_resources",
    ],
    "discovery": [
        "list_training_jobs",
        "get_training_job",
        "list_runtimes",
        "get_runtime",
    ],
    "training": ["fine_tune", "run_custom_training", "run_container_training"],
    "monitoring": ["get_training_logs", "get_training_events", "wait_for_training"],
    "lifecycle": ["delete_training_job", "update_training_job"],
    "platform": [
        "inspect_crd",
        "inspect_controller",
        "patch_runtime",
        "create_runtime",
        "delete_runtime",
    ],
    "health": ["health_check", "get_server_logs"],
    # ── Optimizer (Katib) phases ──
    "optimizer_planning": ["katib_pre_flight"],
    "optimizer_discovery": [
        "list_experiments",
        "get_experiment",
        "get_experiment_status",
        "get_trial",
        "get_successful_trials",
        "list_suggestions",
    ],
    "optimizer_optimization": ["create_hpo_experiment", "create_experiment_from_spec"],
    "optimizer_monitoring": [
        "get_experiment_trials",
        "get_best_trial",
        "get_suggestion",
        "wait_for_experiment",
        "get_experiment_trial_logs",
        "get_experiment_events",
    ],
    "optimizer_lifecycle": ["delete_experiment", "update_experiment"],
}

# Reverse mapping: tool name -> phase
TOOL_TO_PHASE: dict[str, str] = {
    tool: phase for phase, tools in TOOL_PHASES.items() for tool in tools
}


# =============================================================================
# SDK Compatibility
# Machine-readable version constraints and API coverage per client.
# Mirrors the compatibility table in README.md.
# =============================================================================

MIN_K8S_VERSION = (1, 27)
MIN_TRAINER_CRD_VERSION = "v1alpha1"
TRAINER_CRD_GROUP = "trainer.kubeflow.org"
TRAINER_CRD_NAME = "trainjobs.trainer.kubeflow.org"

# =============================================================================
# Tool Next-Step Hints
# Injected into tool responses as _meta.next for clients that don't
# consume server instructions or resources (e.g. Ollama, custom agents).
# =============================================================================

TOOL_NEXT_HINTS: dict[str, str] = {
    # ── Trainer hints ──
    "pre_flight": "Proceed to list_runtimes() to find available runtimes",
    "check_compatibility": "Call get_cluster_resources() or use pre_flight() for full check",
    "get_cluster_resources": "Call estimate_resources(model=...) or list_runtimes()",
    "estimate_resources": "Call list_runtimes() to find a runtime, then submit training",
    "list_runtimes": "Call get_runtime(name) to inspect before training",
    "get_runtime": "Ready to submit training — use fine_tune() or run_custom_training()",
    "list_training_jobs": "Call get_training_job(name) for details on a specific job",
    "fine_tune": "Monitor with get_training_job(name) and get_training_logs(name)",
    "run_custom_training": "Monitor with get_training_job(name) and get_training_logs(name)",
    "run_container_training": "Monitor with get_training_job(name) and get_training_logs(name)",
    "get_training_job": (
        "Use get_training_logs(name) for output or get_training_events(name) for scheduling issues"
    ),
    "get_training_logs": "If errors found, check get_training_events(name) for K8s-level issues",
    "get_training_events": "Fix the issue and retry, or delete_training_job(name) and resubmit",
    "wait_for_training": (
        "Check get_training_logs(name) for final output. "
        "Use create_hpo_experiment() to optimize hyperparameters"
    ),
    "delete_training_job": "Resubmit with fine_tune() or run_custom_training() if needed",
    "update_training_job": "Check get_training_job(name) to verify new status",
    "inspect_crd": "Use inspect_controller(view='logs') to check controller health",
    "inspect_controller": "Check get_training_events(name) if a specific job is failing",
    "health_check": "If degraded, check get_server_logs() for errors",
    "get_server_logs": "Filter by level='ERROR' to find issues",
    "patch_runtime": "Verify changes with get_runtime(name)",
    "create_runtime": "Verify with list_runtimes() or get_runtime(name)",
    "delete_runtime": "Confirm removal with list_runtimes()",
    # ── Optimizer (Katib) hints ──
    "katib_pre_flight": "Proceed to list_experiments() or create_hpo_experiment()",
    "list_experiments": "Call get_experiment(name) for details on a specific experiment",
    "get_experiment": "Use get_experiment_trials(name) or get_best_trial(name) for results",
    "get_experiment_status": "Use get_experiment(name) for full details or get_experiment_trial_logs(name)",
    "get_trial": "Check get_experiment_trial_logs(name, trial=trial) for trial output",
    "get_successful_trials": "Use get_best_trial(name) for the optimal result",
    "list_suggestions": "Check get_suggestion(name) for details on a specific suggestion",
    "get_experiment_trials": "Use get_best_trial(name) or get_experiment_trial_logs(name, trial=trial)",
    "get_best_trial": "Use fine_tune() to retrain with these hyperparameters",
    "get_suggestion": "Check get_experiment_events(name) if suggestion is stuck",
    "wait_for_experiment": "Check get_best_trial(name) for optimal results",
    "get_experiment_trial_logs": "If errors found, check get_experiment_events(name)",
    "get_experiment_events": "Fix the issue and retry, or delete_experiment(name) and resubmit",
    "create_hpo_experiment": "Monitor with get_experiment_status() or wait_for_experiment()",
    "create_experiment_from_spec": "Monitor with get_experiment_status() or wait_for_experiment()",
    "delete_experiment": "Resubmit with create_hpo_experiment() if needed",
    "update_experiment": "Check get_experiment_status(name) to verify new status",
}


SDK_COMPATIBILITY: dict[str, object] = {
    "sdk_package": "kubeflow",
    "sdk_version_min": "0.4.0",
    "trainer_version_min": "v2.2.0",
    "python_requires": ">=3.10",
    "kubernetes_requires": ">=1.27",
    "clients": {
        "trainer": {
            "status": "implemented",
            "sdk_client": "kubeflow.trainer.TrainerClient",
            "covered_methods": [
                "train",
                "get_job",
                "list_jobs",
                "delete_job",
                "get_job_logs",
                "get_job_events",
                "wait_for_job_status",
                "list_runtimes",
                "get_runtime",
            ],
            "uncovered_methods": [
                "get_job_logs(follow=True)",  # streaming not exposed
            ],
            "k8s_api_operations": [
                "suspend (CustomObjectsApi patch)",
                "resume (CustomObjectsApi patch)",
            ],
        },
        "optimizer": {
            "status": "implemented",
            "sdk_client": "kubeflow.optimizer.OptimizerClient",
            "covered_methods": [
                "optimize",
                "get_job",
                "list_jobs",
                "delete_job",
                "get_best_results",
                "get_job_logs",
                "get_job_events",
                "wait_for_job_status",
            ],
            "uncovered_methods": [],  # all 8 OptimizerClient methods are covered
            "k8s_api_operations": [
                "create_experiment_from_spec (CustomObjectsApi create)",
                "create_hpo_experiment/early_stopping (CustomObjectsApi create)",
                "list_suggestions (CustomObjectsApi list)",
                "get_suggestion (CustomObjectsApi get)",
                "update_experiment (CustomObjectsApi patch)",
                "katib_pre_flight (ApiextensionsV1Api + CoreV1Api)",
            ],
        },
        "hub": {
            "status": "stub",
            "sdk_client": "model_registry.ModelRegistry",
            "covered_methods": [],
        },
        "pipelines": {
            "status": "planned",
            "sdk_client": "kubeflow.pipelines.Client",
            "covered_methods": [],
        },
        "spark": {
            "status": "planned",
            "sdk_client": "kubeflow.spark.SparkClient",
            "covered_methods": [],
        },
        "feast": {
            "status": "planned",
            "sdk_client": "kubeflow.feast.FeastClient",
            "covered_methods": [],
        },
    },
}
