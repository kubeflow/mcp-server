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
- Version constraints (SDK, Trainer, Kubernetes)

Import from here to ensure consistency across the codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kubeflow_mcp.core.resilience import CircuitBreaker


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


def record_tool_result(
    breaker: CircuitBreaker,
    result: Any,
    generation: int | None = None,
) -> bool:
    """Record an outcome against the breaker generation that executed the tool.

    Successes reset that generation. Infrastructure errors record a failure;
    neutral errors (validation, not found) release its reserved slot. Returns
    True for success and False otherwise.
    """
    is_success = (
        ("error_code" not in result and "error" not in result) if isinstance(result, dict) else True
    )
    if is_success:
        breaker.record_success(generation)
    elif is_infrastructure_error(result):
        breaker.record_failure(generation)
    else:
        breaker.release(generation)
    return is_success


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
    # Spark client (kubeflow.spark.SparkClient) — SparkConnect session management.
    "spark_discovery": ["list_spark_sessions", "get_spark_session"],
    "spark_sessions": ["create_spark_session", "delete_spark_session"],
    "spark_monitoring": ["get_spark_session_logs"],
    # Pipelines client (kubeflow.pipelines.PipelinesClient). Prefixed so the phases
    # never merge into the trainer's instruction sections.
    "pipelines_discovery": ["list_pipelines", "get_pipeline", "list_experiments", "get_experiment"],
    "pipelines_authoring": ["create_pipeline"],
    "pipelines_monitoring": ["list_pipeline_runs", "get_pipeline_run", "get_pipeline_logs"],
    "pipelines_runs": ["create_pipeline_run", "create_experiment"],
    "pipelines_lifecycle": [
        "stop_pipeline_run",
        "retry_pipeline_run",
        "delete_experiment",
        "delete_pipeline",
    ],
    # ── Optimizer (Katib) phases ──
    "optimizer_planning": ["katib_pre_flight"],
    "optimizer_discovery": [
        "list_hpo_experiments",
        "get_hpo_experiment",
        "get_hpo_experiment_status",
        "get_trial",
        "get_successful_trials",
        "list_suggestions",
    ],
    "optimizer_optimization": ["create_hpo_experiment", "create_hpo_experiment_from_spec"],
    "optimizer_monitoring": [
        "get_hpo_experiment_trials",
        "get_best_trial",
        "get_suggestion",
        "wait_for_hpo_experiment",
        "get_hpo_experiment_trial_logs",
        "get_hpo_experiment_events",
    ],
    "optimizer_lifecycle": ["delete_hpo_experiment", "update_hpo_experiment"],
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

KUBEFLOW_SDK_VERSION = "0.5.0"
KUBEFLOW_SDK_VERSION_MIN = "0.5.0"
KUBEFLOW_SDK_VERSION_SPEC = ">=0.5.0,<0.6.0"
KUBEFLOW_TRAINER_VERSION_MIN = "2.3.0"
KUBEFLOW_TRAINER_VERSION_SPEC = ">=2.3.0,<2.4.0"
KUBERNETES_VERSION_LABEL = f"{MIN_K8S_VERSION[0]}.{MIN_K8S_VERSION[1]}+"

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
    # Spark client
    "list_spark_sessions": "Call get_spark_session(name) for details on a specific session",
    "get_spark_session": (
        "Use get_spark_session_logs(name) for driver output, or delete_spark_session(name) to tear down"
    ),
    "get_spark_session_logs": "If errors found, check the session state with get_spark_session(name)",
    "create_spark_session": (
        "Monitor with get_spark_session(name); attach with pyspark using the returned connect info"
    ),
    "delete_spark_session": "Confirm removal with list_spark_sessions()",
    # Pipelines client
    "list_pipelines": "Call get_pipeline(name) for its versions, or list_pipeline_runs()",
    "get_pipeline": "Preview a run with create_pipeline_run(pipeline=name)",
    "list_pipeline_runs": "Call get_pipeline_run(run_id) for a run's tasks and error",
    "get_pipeline_run": "If it failed, read get_pipeline_logs(run_id) for the failing task",
    "get_pipeline_logs": "Fix the cause, then preview retry_pipeline_run(run_id)",
    "create_pipeline_run": "Follow the run with get_pipeline_run(run_id)",
    "stop_pipeline_run": (
        "Confirm with get_pipeline_run(run_id) that it stopped ('canceled' or 'failed')"
    ),
    "retry_pipeline_run": "Follow the retried run with get_pipeline_run(run_id)",
    "list_experiments": "Call get_experiment(name), or preview create_experiment(name) for a new one",
    "get_experiment": "Preview a run in it with create_pipeline_run(pipeline, experiment=name)",
    "create_experiment": "Preview a run in it with create_pipeline_run(pipeline, experiment=name)",
    "delete_experiment": "Confirm removal with list_experiments()",
    "create_pipeline": "Preview a run with create_pipeline_run(pipeline=name)",
    "delete_pipeline": "Confirm the result with list_pipelines()",
    # ── Optimizer (Katib) hints ──
    "katib_pre_flight": "Proceed to list_hpo_experiments() or create_hpo_experiment()",
    "list_hpo_experiments": "Call get_hpo_experiment(name) for details on a specific experiment",
    "get_hpo_experiment": "Use get_hpo_experiment_trials(name) or get_best_trial(name) for results",
    "get_hpo_experiment_status": "Use get_hpo_experiment(name) for full details or get_hpo_experiment_trial_logs(name)",
    "get_trial": "Check get_hpo_experiment_trial_logs(name, trial=trial) for trial output",
    "get_successful_trials": "Use get_best_trial(name) for the optimal result",
    "list_suggestions": "Check get_suggestion(name) for details on a specific suggestion",
    "get_hpo_experiment_trials": "Use get_best_trial(name) or get_hpo_experiment_trial_logs(name, trial=trial)",
    "get_best_trial": "Use fine_tune() to retrain with these hyperparameters",
    "get_suggestion": "Check get_hpo_experiment_events(name) if suggestion is stuck",
    "wait_for_hpo_experiment": "Check get_best_trial(name) for optimal results",
    "get_hpo_experiment_trial_logs": "If errors found, check get_hpo_experiment_events(name)",
    "get_hpo_experiment_events": "Fix the issue and retry, or delete_hpo_experiment(name) and resubmit",
    "create_hpo_experiment": "Monitor with get_hpo_experiment_status() or wait_for_hpo_experiment()",
    "create_hpo_experiment_from_spec": "Monitor with get_hpo_experiment_status() or wait_for_hpo_experiment()",
    "delete_hpo_experiment": "Resubmit with create_hpo_experiment() if needed",
    "update_hpo_experiment": "Check get_hpo_experiment(name) to confirm the new trial settings",
}


SDK_COMPATIBILITY: dict[str, object] = {
    "sdk_package": "kubeflow",
    "sdk_version_min": KUBEFLOW_SDK_VERSION_MIN,
    "trainer_version_min": KUBEFLOW_TRAINER_VERSION_MIN,
    "sdk_version": KUBEFLOW_SDK_VERSION_SPEC,
    "trainer_version": KUBEFLOW_TRAINER_VERSION_SPEC,
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
            # Only the client's resolved default namespace is used. Every tool
            # reads and writes Katib's resources directly; see uncovered_methods.
            "covered_methods": [],
            "uncovered_methods": [
                # optimize() requires TrainJobTemplate.trainer to be a CustomTrainer
                # holding a live Python callable (func), which cannot cross the
                # MCP/JSON boundary. It also generates its own Experiment name and
                # sets no labels, so a caller-supplied name and the MCP ownership
                # label are both unreachable through it. create_hpo_experiment
                # builds the V1beta1Experiment CR directly instead.
                "optimize",
                # The readers below convert the algorithm through a registry that
                # holds only random and grid, and resolve every trial as a TrainJob
                # named after it. Experiments using any other algorithm, or a
                # batch/v1 Job (or other) trial template, fail, so the tools read
                # the Experiment and Trial CRs and the trial pods instead.
                "list_jobs",
                "get_job",
                "get_best_results",
                "get_job_logs",
                "get_job_events",
                "wait_for_job_status",
                # Sends no request timeout, so a confirmed delete could hang.
                "delete_job",
            ],
            "k8s_api_operations": [
                "create_hpo_experiment (CustomObjectsApi create)",
                "create_hpo_experiment_from_spec (CustomObjectsApi create)",
                "list_hpo_experiments (CustomObjectsApi list)",
                "get_hpo_experiment (CustomObjectsApi get + Trial list)",
                "get_hpo_experiment_status (CustomObjectsApi get)",
                "get_trial (CustomObjectsApi get + Trial list)",
                "get_successful_trials (CustomObjectsApi get + Trial list)",
                "get_hpo_experiment_trials (CustomObjectsApi get + Trial list)",
                "get_best_trial (CustomObjectsApi get)",
                "wait_for_hpo_experiment (CustomObjectsApi get, polled)",
                "get_hpo_experiment_trial_logs (CoreV1Api pod list + bounded log read)",
                "get_hpo_experiment_events (CoreV1Api event + pod list)",
                "list_suggestions (CustomObjectsApi list)",
                "get_suggestion (CustomObjectsApi get)",
                "update_hpo_experiment (CustomObjectsApi get + Trial list + patch)",
                "delete_hpo_experiment (CustomObjectsApi delete)",
                "katib_pre_flight (ApiextensionsV1Api + CoreV1Api)",
            ],
        },
        "hub": {
            "status": "stub",
            "sdk_client": "model_registry.ModelRegistry",
            "covered_methods": [],
        },
        "pipelines": {
            "status": "implemented",
            "sdk_client": "kubeflow.pipelines.PipelinesClient",
            "covered_methods": [
                "list_pipelines",
                "get_pipeline",
                "get_pipeline_version",
                "list_pipeline_versions",
                "list_runs",
                "get_run",
                "run",
                "get_experiment",
                "list_experiments",
                "create_experiment",
                "delete_experiment",
                "upload_pipeline",  # compiled YAML only; create_pipeline compiles the code itself
                "delete_pipeline",
                "_backend.run_api.run_service_terminate_run",
                "_backend.run_api.run_service_retry_run",
                "_backend.experiment_api.experiment_service_get_experiment",
            ],
            "uncovered_methods": [
                "upload_pipeline(callable)",  # would compile in-process; see create_pipeline
                "wait_for_run_status",  # blocks; agents poll get_pipeline_run instead
                "list_pipeline_versions(page_token)",  # get_pipeline returns the first page
            ],
            "k8s_api_operations": [
                "compile pipeline code (kfp.compiler.Compiler in a child interpreter)",
                "retry run (kfp.Client run API, run_service_retry_run)",
                "task pod logs (CoreV1Api read_namespaced_pod_log)",
            ],
        },
        "spark": {
            "status": "implemented",
            "sdk_client": "kubeflow.spark.SparkClient",
            "covered_methods": [
                "connect",  # create mode, via create_spark_session
                "list_sessions",
                "get_session",
                "delete_session",
                "get_session_logs",
            ],
            "uncovered_methods": [
                "connect(base_url=...)",  # attaching to an existing server returns a live
                # pyspark SparkSession, which is not serializable over MCP; the data plane
                # attaches directly using the connect info returned by create_spark_session.
                "get_session_logs(follow=True)",  # streaming not exposed
                # SparkApplication batch lifecycle is deferred to issue #331.
                "submit_job",
                "get_job",
                "list_jobs",
                "delete_job",
                "wait_for_job_status",
                "get_job_logs",
            ],
        },
        "feast": {
            "status": "planned",
            "sdk_client": "kubeflow.feast.FeastClient",
            "covered_methods": [],
        },
    },
}
