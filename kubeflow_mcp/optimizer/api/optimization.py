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
    INT_PARAMETER,
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
        TypeError: The entry is not an object.
        ValueError: The entry is an object but malformed.

    Either message is surfaced to the caller as a validation error.
    """
    if not isinstance(spec, dict):
        raise TypeError(f"search_space['{name}'] must be an object, got {type(spec).__name__}")

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
        elif distribution == "int":
            param = _build_int_parameter(name, spec)
        else:
            raise ValueError(
                f"search_space['{name}'].type must be 'uniform', 'loguniform' or 'int', "
                f"got '{distribution}'"
            )

    param.name = name
    return param


def _build_int_parameter(name: str, spec: dict[str, Any]) -> models.V1beta1ParameterSpec:
    """Build an integer parameter, which the SDK's ``Search`` cannot express.

    ``Search`` only emits ``double`` and ``categorical``, so an ordered discrete
    range could otherwise only be written as ``choices``, which every algorithm
    then treats as unordered categories. Reuses ``Search.uniform`` for the
    feasible-space encoding and retypes it, so the string coercion still matches
    the SDK's own output exactly.
    """
    for bound in ("min", "max"):
        value = spec[bound]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"search_space['{name}'].{bound} must be an integer when type is "
                f"'int', got {value!r}"
            )

    param = Search.uniform(spec["min"], spec["max"])
    param.parameter_type = INT_PARAMETER

    if "step" in spec:
        step = spec["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError(
                f"search_space['{name}'].step must be a positive integer, got {step!r}"
            )
        param.feasible_space.step = str(step)

    return param


def _resolve_primary_container(trial_template: dict[str, Any], override: str | None) -> str:
    """Name the container Katib should collect metrics from.

    Katib injects its metrics collector into the container named by
    ``primaryContainerName``; if that name matches nothing in the trial's pod,
    no metrics are ever collected and the experiment stalls with no error.
    Defaulting to Trainer's ``node`` is only right for TrainJob templates, so
    the name is read off the template when it carries an inline pod spec.
    """
    if override:
        return override

    spec = trial_template.get("spec")
    if not isinstance(spec, dict):
        return trainer_constants.NODE

    # batch/v1 Job and friends nest the pod under spec.template; a bare Pod
    # trialSpec carries containers directly.
    pod_spec = (
        spec.get("template", {}).get("spec") if isinstance(spec.get("template"), dict) else None
    )
    containers = (pod_spec or spec).get("containers")
    if not isinstance(containers, list):
        return trainer_constants.NODE

    names = [c["name"] for c in containers if isinstance(c, dict) and c.get("name")]
    if not names:
        # TrainJob and other runtime-backed templates have no inline containers;
        # Trainer names the workload container "node".
        return trainer_constants.NODE
    # A sidecar can sort first, so prefer the conventional name when present.
    return trainer_constants.NODE if trainer_constants.NODE in names else names[0]


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
    primary_container_name: str | None = None,
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
                primaryContainerName=_resolve_primary_container(
                    trial_template, primary_container_name
                ),
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
    primary_container_name: str | None = None,
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Create a hyperparameter optimization experiment from flat parameters.

    ``search_space`` maps each parameter name to a continuous range, an integer
    range, or a categorical list::

        {
            "lr": {"min": 0.001, "max": 0.1, "type": "loguniform"},
            "momentum": {"min": 0.5, "max": 0.99},
            "num_layers": {"min": 2, "max": 8, "type": "int"},
            "batch_size": {"choices": [16, 32, 64]},
        }

    Prefer ``type: "int"`` over ``choices`` for an ordered discrete range:
    categorical values carry no ordering, so the algorithm cannot tell that 4
    lies between 2 and 8.

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
        primary_container_name: Container Katib collects metrics from. Defaults
            to the container named in ``trial_template``, or Trainer's ``node``
            when the template carries no inline pod spec (a TrainJob).
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

    # (is_invalid, message) pairs, evaluated in order; the first failure wins.
    # Kept as a table so adding a rule does not add another return path.
    problems: list[tuple[bool, str]] = [
        (
            objective_type not in OBJECTIVE_TYPES,
            f"objective_type must be one of {sorted(OBJECTIVE_TYPES)}, got '{objective_type}'",
        ),
        (
            algorithm not in ALGORITHMS,
            f"algorithm must be one of {sorted(ALGORITHMS)}, got '{algorithm}'",
        ),
        (not search_space, "search_space must define at least one parameter"),
        (not trial_template, "trial_template must define the workload to run per trial"),
        (
            not 1 <= max_trial_count <= MAX_TRIAL_COUNT_LIMIT,
            f"max_trial_count must be between 1 and {MAX_TRIAL_COUNT_LIMIT}, got {max_trial_count}",
        ),
        (
            not 1 <= parallel_trial_count <= MAX_PARALLEL_TRIAL_LIMIT,
            f"parallel_trial_count must be between 1 and {MAX_PARALLEL_TRIAL_LIMIT}, "
            f"got {parallel_trial_count}",
        ),
    ]
    for is_invalid, message in problems:
        if is_invalid:
            return ToolError(error=message, error_code=ErrorCode.VALIDATION_ERROR).model_dump()

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
            primary_container_name=primary_container_name,
        ).to_dict()
    except (TypeError, ValueError) as e:
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
