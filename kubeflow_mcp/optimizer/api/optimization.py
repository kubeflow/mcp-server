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

"""Optimization tools for creating Katib experiments.

Both tools build a ``V1beta1Experiment`` and create it through
``CustomObjectsApi``.

``OptimizerClient.optimize()`` is deliberately **not** used: it requires
``TrainJobTemplate.trainer`` to be a ``CustomTrainer`` holding a live Python
callable (``func``), which cannot cross the MCP/JSON boundary. It also
generates its own Experiment name and sets no metadata labels, so MCP could
neither honour a caller-supplied name nor stamp the ownership label that
``delete_experiment``/``update_experiment`` rely on. Building the CR directly
avoids all three limits, and additionally reaches Katib algorithms the SDK's
typed classes cannot express (tpe, cmaes, hyperband, ...).
"""

import logging
from typing import Any

from kubeflow.optimizer import Search
from kubeflow.trainer.constants import constants as trainer_constants
from kubeflow_katib_api import models

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import PreviewResponse, ToolError, ToolResponse
from kubeflow_mcp.core.security import validate_k8s_name
from kubeflow_mcp.optimizer.api._common import check_optimizer_namespace, experiment_error
from kubeflow_mcp.optimizer.constants import (
    EXPERIMENT_KIND,
    EXPERIMENT_PLURAL,
    KATIB_API_GROUP,
    KATIB_API_VERSION,
)

logger = logging.getLogger(__name__)

# Katib suggestion algorithms. Superset of what the SDK's typed algorithm
# classes expose (RandomSearch/GridSearch), reachable because we build the CR.
ALGORITHMS = frozenset(
    {
        "random",
        "grid",
        "bayesianoptimization",
        "tpe",
        "multivariate-tpe",
        "cmaes",
        "sobol",
        "hyperband",
    }
)
OBJECTIVE_TYPES = frozenset({"maximize", "minimize"})

# Guard rails against an agent requesting a runaway experiment.
MAX_TRIAL_COUNT_LIMIT = 1000
MAX_PARALLEL_TRIAL_LIMIT = 100


def _build_parameter(name: str, spec: dict[str, Any]) -> models.V1beta1ParameterSpec:
    """Convert one search-space entry into a Katib parameter spec.

    Uses the SDK's ``Search`` helpers so feasible-space encoding (distribution
    names, string coercion of numbers) stays identical to the SDK's own output.

    Raises:
        ValueError: The entry is malformed; the message is surfaced to the caller.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"search_space['{name}'] must be an object, got {type(spec).__name__}")

    if "choices" in spec:
        choices = spec["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"search_space['{name}'].choices must be a non-empty list")
        param = Search.choice(choices)
    else:
        missing = {"min", "max"} - spec.keys()
        if missing:
            raise ValueError(
                f"search_space['{name}'] needs 'min' and 'max' (or 'choices'); "
                f"missing {sorted(missing)}"
            )
        distribution = spec.get("type", "uniform")
        if distribution == "uniform":
            param = Search.uniform(spec["min"], spec["max"])
        elif distribution == "loguniform":
            param = Search.loguniform(spec["min"], spec["max"])
        else:
            raise ValueError(
                f"search_space['{name}'].type must be 'uniform' or 'loguniform', "
                f"got '{distribution}'"
            )

    param.name = name
    return param


def _build_experiment(
    *,
    name: str,
    namespace: str,
    objective_metric: str,
    objective_type: str,
    search_space: dict[str, Any],
    trial_template: dict[str, Any],
    algorithm: str,
    max_trial_count: int,
    parallel_trial_count: int,
    max_failed_trials: int | None,
) -> models.V1beta1Experiment:
    """Assemble the Experiment CR, stamped with the MCP ownership label."""
    return models.V1beta1Experiment(
        apiVersion=f"{KATIB_API_GROUP}/{KATIB_API_VERSION}",
        kind=EXPERIMENT_KIND,
        metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
            name=name,
            namespace=namespace,
            labels={mcp_utils.MCP_MANAGED_LABEL: mcp_utils.MCP_MANAGED_VALUE},
        ),
        spec=models.V1beta1ExperimentSpec(
            parameters=[_build_parameter(key, spec) for key, spec in search_space.items()],
            maxTrialCount=max_trial_count,
            parallelTrialCount=parallel_trial_count,
            maxFailedTrialCount=max_failed_trials,
            objective=models.V1beta1ObjectiveSpec(
                objectiveMetricName=objective_metric,
                type=objective_type,
            ),
            algorithm=models.V1beta1AlgorithmSpec(algorithmName=algorithm),
            trialTemplate=models.V1beta1TrialTemplate(
                retain=True,
                primaryContainerName=trainer_constants.NODE,
                trialParameters=[
                    models.V1beta1TrialParameterSpec(name=key, reference=key)
                    for key in search_space
                ],
                trialSpec=trial_template,
            ),
        ),
    )


def _create(experiment: dict[str, Any], namespace: str, name: str) -> dict[str, Any]:
    """Create the Experiment CR and build the success response."""
    api = mcp_utils.get_custom_objects_api()
    api.create_namespaced_custom_object(
        group=KATIB_API_GROUP,
        version=KATIB_API_VERSION,
        namespace=namespace,
        plural=EXPERIMENT_PLURAL,
        body=experiment,
        _request_timeout=mcp_utils.K8S_TIMEOUT,
    )
    return ToolResponse(
        data={
            "experiment": name,
            "namespace": namespace,
            "created": True,
            "message": f"Experiment '{name}' created in namespace '{namespace}'",
        }
    ).model_dump()


def create_hpo_experiment(
    name: str,
    objective_metric: str,
    search_space: dict[str, Any],
    trial_template: dict[str, Any],
    objective_type: str = "maximize",
    algorithm: str = "random",
    max_trial_count: int = 10,
    parallel_trial_count: int = 2,
    max_failed_trials: int | None = None,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Create a hyperparameter optimization experiment from flat parameters.

    ``search_space`` maps each parameter name to a continuous range or a
    categorical list::

        {
            "lr": {"min": 0.001, "max": 0.1, "type": "loguniform"},
            "momentum": {"min": 0.5, "max": 0.99},
            "batch_size": {"choices": [16, 32, 64]},
        }

    ``trial_template`` is the Katib ``trialSpec`` — the workload cloned per
    trial. Reference tuned parameters with ``${trialParameters.<name>}``::

        {
            "apiVersion": "trainer.kubeflow.org/v1alpha1",
            "kind": "TrainJob",
            "spec": {
                "runtimeRef": {"name": "torch-distributed"},
                "trainer": {
                    "image": "my/trainer:latest",
                    "args": ["--lr=${trialParameters.lr}"],
                },
            },
        }

    Args:
        name: Experiment name (also the Kubernetes resource name).
        objective_metric: Metric to optimize (e.g. ``"accuracy"``).
        search_space: Parameter ranges to explore (see format above).
        trial_template: Katib ``trialSpec`` for the workload to run per trial.
        objective_type: ``"maximize"`` or ``"minimize"``. Default ``"maximize"``.
        algorithm: Suggestion algorithm; one of :data:`ALGORITHMS`.
        max_trial_count: Total trials to run. Default 10.
        parallel_trial_count: Trials to run concurrently. Default 2.
        max_failed_trials: Failed trials tolerated before the experiment fails.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        confirmed: Set ``True`` to submit. ``False`` returns a preview.

    Returns:
        dict: Preview of the Experiment CR, or the creation result.
    """
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    if objective_type not in OBJECTIVE_TYPES:
        return ToolError(
            error=f"objective_type must be one of {sorted(OBJECTIVE_TYPES)}, got '{objective_type}'",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    if algorithm not in ALGORITHMS:
        return ToolError(
            error=f"algorithm must be one of {sorted(ALGORITHMS)}, got '{algorithm}'",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    if not search_space:
        return ToolError(
            error="search_space must define at least one parameter",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    if not trial_template:
        return ToolError(
            error="trial_template must define the workload to run per trial",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    if not 1 <= max_trial_count <= MAX_TRIAL_COUNT_LIMIT:
        return ToolError(
            error=(
                f"max_trial_count must be between 1 and {MAX_TRIAL_COUNT_LIMIT}, "
                f"got {max_trial_count}"
            ),
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    if not 1 <= parallel_trial_count <= MAX_PARALLEL_TRIAL_LIMIT:
        return ToolError(
            error=(
                f"parallel_trial_count must be between 1 and {MAX_PARALLEL_TRIAL_LIMIT}, "
                f"got {parallel_trial_count}"
            ),
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    try:
        ns = mcp_utils.get_optimizer_effective_namespace(namespace)
        experiment = _build_experiment(
            name=name,
            namespace=ns,
            objective_metric=objective_metric,
            objective_type=objective_type,
            search_space=search_space,
            trial_template=trial_template,
            algorithm=algorithm,
            max_trial_count=max_trial_count,
            parallel_trial_count=parallel_trial_count,
            max_failed_trials=max_failed_trials,
        ).to_dict()
    except ValueError as e:
        return ToolError(error=str(e), error_code=ErrorCode.VALIDATION_ERROR).model_dump()

    if not confirmed:
        return PreviewResponse(
            message=(
                f"Will create experiment '{name}' in '{ns}': {max_trial_count} trials "
                f"({parallel_trial_count} in parallel) optimizing '{objective_metric}' "
                f"via '{algorithm}'. Set confirmed=True to submit."
            ),
            config=experiment,
        ).model_dump()

    try:
        return _create(experiment, ns, name)
    except Exception as e:
        logger.warning("create_hpo_experiment(%s) failed: %s", name, e, exc_info=True)
        return experiment_error(e, name).model_dump()


def create_experiment_from_spec(
    spec: dict[str, Any],
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Create an experiment from a complete V1beta1Experiment manifest.

    Escape hatch for configurations ``create_hpo_experiment()`` does not model —
    early stopping, custom metrics collectors, resume policies or NAS. The
    manifest is validated against the Katib schema before submission and is
    stamped with the MCP ownership label.

    Args:
        spec: Full ``V1beta1Experiment`` manifest (``apiVersion``, ``kind``,
            ``metadata.name``, ``spec``).
        namespace: K8s namespace. Overrides ``metadata.namespace`` when given.
        confirmed: Set ``True`` to submit. ``False`` returns a preview.

    Returns:
        dict: Preview of the Experiment CR, or the creation result.
    """
    if not isinstance(spec, dict) or not spec:
        return ToolError(
            error="spec must be a non-empty V1beta1Experiment manifest",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    name = (spec.get("metadata") or {}).get("name")
    if not name:
        return ToolError(
            error="spec.metadata.name is required",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    name_err = validate_k8s_name(name)
    if name_err is not None:
        return name_err.model_dump()

    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        experiment = models.V1beta1Experiment.from_dict(spec)
    except Exception as e:
        return ToolError(
            error=f"Invalid V1beta1Experiment spec: {e}",
            error_code=ErrorCode.VALIDATION_ERROR,
            hint="Compare against the Katib v1beta1 Experiment schema.",
        ).model_dump()
    if experiment is None:
        return ToolError(
            error="Invalid V1beta1Experiment spec: parsed to nothing",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    ns = mcp_utils.get_optimizer_effective_namespace(
        namespace or (spec.get("metadata") or {}).get("namespace")
    )

    body = experiment.to_dict()
    metadata = body.setdefault("metadata", {})
    metadata["namespace"] = ns
    metadata.setdefault("labels", {})[mcp_utils.MCP_MANAGED_LABEL] = mcp_utils.MCP_MANAGED_VALUE

    if not confirmed:
        return PreviewResponse(
            message=f"Will create experiment '{name}' in '{ns}'. Set confirmed=True to submit.",
            config=body,
        ).model_dump()

    try:
        return _create(body, ns, name)
    except Exception as e:
        logger.warning("create_experiment_from_spec(%s) failed: %s", name, e, exc_info=True)
        return experiment_error(e, name).model_dump()
