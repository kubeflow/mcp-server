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

"""Shared data and helpers for the Tier 1 eval judges."""

from typing import Any

# Minimal valid arguments that get each mutating tool as far as its preview.
# A new mutating tool fails test_every_mutating_tool_has_gate_arguments until
# it is added here, so the gate check cannot silently skip it.
# Pipeline runs the KFP fake in conftest.py knows: one still running, so it can be
# stopped, and one failed, so it can be retried. Run IDs must be UUIDs.
RUNNING_RUN_ID = "00000000-0000-4000-8000-000000000001"
FAILED_RUN_ID = "00000000-0000-4000-8000-000000000002"
# The one experiment the KFP fake knows, so it can be deleted.
EXPERIMENT = "exp-a"

PIPELINE_CODE = """
from kfp import dsl


@dsl.component
def hello() -> str:
    return "hi"


@dsl.pipeline
def pipe():
    hello()
"""

# A TrainJob trial template and a complete Experiment manifest for the optimizer.
TRIAL_TEMPLATE: dict[str, Any] = {
    "apiVersion": "trainer.kubeflow.org/v1alpha1",
    "kind": "TrainJob",
    "spec": {
        "runtimeRef": {"name": "torch-distributed"},
        "trainer": {"image": "busybox:1.36", "args": ["--lr=${trialParameters.lr}"]},
    },
}
EXPERIMENT_SPEC: dict[str, Any] = {
    "apiVersion": "kubeflow.org/v1beta1",
    "kind": "Experiment",
    "metadata": {"name": "hpo-b"},
    "spec": {
        "objective": {"type": "maximize", "objectiveMetricName": "accuracy"},
        "algorithm": {"algorithmName": "random"},
        "maxTrialCount": 4,
        "parallelTrialCount": 2,
        "parameters": [
            {
                "name": "lr",
                "parameterType": "double",
                "feasibleSpace": {"min": "0.001", "max": "0.1"},
            }
        ],
        "trialTemplate": {
            "primaryContainerName": "node",
            "trialParameters": [{"name": "lr", "reference": "lr"}],
            "trialSpec": TRIAL_TEMPLATE,
        },
    },
}

GATE_ARGS: dict[str, dict[str, Any]] = {
    "create_experiment": {"name": "exp-new", "namespace": "default"},
    "create_hpo_experiment": {
        "name": "hpo-a",
        "objective_metric": "accuracy",
        "search_space": {"lr": {"min": 0.001, "max": 0.1, "type": "loguniform"}},
        "trial_template": TRIAL_TEMPLATE,
        "namespace": "default",
    },
    "create_hpo_experiment_from_spec": {"spec": EXPERIMENT_SPEC, "namespace": "default"},
    "create_pipeline": {"name": "pipe-a", "pipeline_code": PIPELINE_CODE, "namespace": "default"},
    "create_pipeline_run": {"pipeline": "pipe-a", "namespace": "default"},
    "create_runtime": {"name": "rt-a", "spec": {"template": {}}},
    "create_spark_session": {"name": "spark-a"},
    "delete_experiment": {"name": EXPERIMENT, "namespace": "default"},
    "delete_hpo_experiment": {"name": "hpo-a", "namespace": "default"},
    "delete_pipeline": {"name": "pipe-a", "namespace": "default"},
    "delete_runtime": {"name": "rt-a"},
    "delete_spark_session": {"name": "spark-a"},
    "delete_training_job": {"name": "job-a"},
    "fine_tune": {"model": "hf://google/gemma-2b", "dataset": "hf://tatsu-lab/alpaca"},
    "patch_runtime": {"name": "rt-a", "patch": {"metadata": {"labels": {"a": "b"}}}},
    "retry_pipeline_run": {"run_id": FAILED_RUN_ID, "namespace": "default"},
    "run_container_training": {"image": "busybox:1.36"},
    "run_custom_training": {"script": "print('hi')"},
    "stop_pipeline_run": {"run_id": RUNNING_RUN_ID, "namespace": "default"},
    "update_hpo_experiment": {"name": "hpo-a", "parallel_trial_count": 1, "namespace": "default"},
    "update_training_job": {"name": "job-a", "action": "suspend"},
}

# Valid arguments for each read-only tool that some persona hides, so the persona
# judges also try hidden tools that only read. A new one fails
# test_every_hidden_read_only_tool_has_arguments until it is added here.
READ_ARGS: dict[str, dict[str, Any]] = {
    "inspect_controller": {},
    "inspect_crd": {},
    "wait_for_hpo_experiment": {
        "name": "hpo-a",
        "namespace": "default",
        "timeout_seconds": 10,
        "polling_interval": 5,
    },
    "wait_for_training": {"name": "job-a", "timeout_seconds": 1, "polling_interval": 1},
}

# Arguments for every tool the readonly persona exposes, with optional flags that
# reach extra work switched on (include_packages starts a Pod on some paths), so
# test_readonly_persona_never_writes covers the deepest path of each tool. A new
# readonly tool fails test_every_readonly_tool_has_arguments until it is added here.
READONLY_ARGS: dict[str, dict[str, Any]] = {
    "check_compatibility": {},
    "estimate_resources": {"model": "hf://google/gemma-2b"},
    "get_best_trial": {"name": "hpo-a", "namespace": "default"},
    "get_cluster_resources": {},
    "get_experiment": {"name": EXPERIMENT, "namespace": "default"},
    "get_hpo_experiment": {"name": "hpo-a", "namespace": "default"},
    "get_hpo_experiment_events": {"name": "hpo-a", "namespace": "default"},
    "get_hpo_experiment_status": {"name": "hpo-a", "namespace": "default"},
    "get_hpo_experiment_trial_logs": {"name": "hpo-a", "namespace": "default"},
    "get_hpo_experiment_trials": {"name": "hpo-a", "namespace": "default"},
    "get_pipeline": {"name": "pipe-a", "namespace": "default", "include_versions": True},
    "get_pipeline_logs": {"run_id": RUNNING_RUN_ID, "namespace": "default"},
    "get_pipeline_run": {"run_id": RUNNING_RUN_ID, "namespace": "default"},
    "get_runtime": {"name": "rt-a", "include_packages": True},
    "get_server_logs": {},
    "get_spark_session": {"name": "spark-a"},
    "get_spark_session_logs": {"name": "spark-a"},
    "get_successful_trials": {"name": "hpo-a", "namespace": "default"},
    "get_suggestion": {"name": "hpo-a", "namespace": "default"},
    "get_training_events": {"name": "job-a"},
    "get_training_job": {"name": "job-a"},
    "get_training_logs": {"name": "job-a"},
    "get_trial": {"name": "hpo-a", "trial": "hpo-a-trial-1", "namespace": "default"},
    "health_check": {},
    "katib_pre_flight": {},
    "list_experiments": {"namespace": "default"},
    "list_hpo_experiments": {"namespace": "default"},
    "list_pipeline_runs": {"namespace": "default"},
    "list_pipelines": {"namespace": "default"},
    "list_runtimes": {},
    "list_spark_sessions": {},
    "list_suggestions": {"namespace": "default"},
    "list_training_jobs": {},
    "pre_flight": {},
}

# Readonly tools the fakes cannot serve yet: the call fails before the tool's main
# work, so the judge cannot see a write that comes after that point. Every other
# readonly tool must succeed. Drop an entry once the fakes can serve the tool.
SHALLOW_TOOLS: dict[str, str] = {
    "get_suggestion": "the fake returns a mock Suggestion that cannot be serialized",
    "get_trial": "the fake lists no Katib trials, so the trial is not found",
    "get_spark_session": "the SDK cannot parse a mock SparkConnect",
    "get_spark_session_logs": "the SDK cannot parse a mock SparkConnect",
    "get_training_events": "the SDK cannot parse a mock TrainJob",
    "get_training_job": "the SDK cannot parse a mock TrainJob",
    "get_training_logs": "the SDK cannot parse a mock TrainJob",
    "list_runtimes": "the SDK cannot parse a mock runtime list",
    "list_spark_sessions": "the SDK cannot parse a mock SparkConnect list",
    "list_training_jobs": "the SDK cannot parse a mock TrainJob list",
}


def payload(result: Any) -> dict[str, Any]:
    """Return the tool's response dict from an MCP CallToolResult."""
    body = result.structured_content or {}
    if set(body) == {"result"}:
        body = body["result"]
    return body
