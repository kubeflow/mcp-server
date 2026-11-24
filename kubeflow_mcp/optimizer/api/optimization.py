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
``delete_hpo_experiment``/``update_hpo_experiment`` rely on. Building the CR directly
avoids all three limits, and additionally reaches Katib algorithms the SDK's
typed classes cannot express (tpe, cmaes, sobol, ...).
"""

import json
import math
import re
from typing import Any

from kubeflow.optimizer import Search
from kubeflow.trainer.constants import constants as trainer_constants
from kubeflow_katib_api import models

from kubeflow_mcp.common import utils as mcp_utils
from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import PreviewResponse, ToolError, ToolResponse
from kubeflow_mcp.core.security import validate_image_reference, validate_k8s_name
from kubeflow_mcp.optimizer.api._common import check_optimizer_namespace, experiment_error
from kubeflow_mcp.optimizer.constants import (
    EXPERIMENT_KIND,
    EXPERIMENT_PLURAL,
    INT_PARAMETER,
    KATIB_API_GROUP,
    KATIB_API_VERSION,
)

# Katib suggestion algorithms. Superset of what the SDK's typed algorithm
# classes expose (RandomSearch/GridSearch), reachable because we build the CR.
# hyperband is left out: Katib's Hyperband service requires the r_l and
# resource_name algorithm settings, which create_hpo_experiment cannot pass,
# so it would always fail. create_hpo_experiment_from_spec can set them.
ALGORITHMS = frozenset(
    {
        "random",
        "grid",
        "bayesianoptimization",
        "tpe",
        "multivariate-tpe",
        "cmaes",
        "sobol",
    }
)
OBJECTIVE_TYPES = frozenset({"maximize", "minimize"})

# Guard rails against an agent requesting a runaway experiment.
MAX_TRIAL_COUNT_LIMIT = 1000
MAX_PARALLEL_TRIAL_LIMIT = 100

# Katib's validator is stricter than Kubernetes about Experiment names: it
# requires a leading letter and at most 40 characters.
MAX_EXPERIMENT_NAME_LENGTH = 40

# Trial workload kinds Katib fills successCondition and failureCondition in for
# (setDefaultTrialTemplate). Its validator requires both, and
# create_hpo_experiment has no way to pass them, so it accepts only these kinds;
# anything else, a bare Pod say, needs create_hpo_experiment_from_spec.
DEFAULTED_TRIAL_KINDS = frozenset(
    {"Job", "TFJob", "PyTorchJob", "XGBoostJob", "MPIJob", "TrainJob"}
)


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
            _check_range(name, spec, distribution)
            param = Search.uniform(spec["min"], spec["max"])
        elif distribution == "loguniform":
            _check_range(name, spec, distribution)
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


def _check_range(name: str, spec: dict[str, Any], distribution: str) -> None:
    """Require a range's bounds to be numbers, in order.

    The SDK's ``Search`` helpers only stringify the bounds and Katib's webhook
    only checks that they are set, so a non-numeric or reversed range would
    otherwise be admitted and handed to the suggestion service as is.

    Raises:
        ValueError: A bound is not a finite number (an integer for ``int``), or
            ``min`` exceeds ``max``.
    """
    integer = distribution == "int"
    for bound in ("min", "max"):
        value = spec[bound]
        is_number = isinstance(value, int) or (
            not integer and isinstance(value, float) and math.isfinite(value)
        )
        if isinstance(value, bool) or not is_number:
            raise ValueError(
                f"search_space['{name}'].{bound} must be "
                f"{'an integer' if integer else 'a number'} when type is '{distribution}', "
                f"got {value!r}"
            )
    if spec["min"] > spec["max"]:
        raise ValueError(
            f"search_space['{name}'].min ({spec['min']}) must not exceed max ({spec['max']})"
        )


def _build_int_parameter(name: str, spec: dict[str, Any]) -> models.V1beta1ParameterSpec:
    """Build an integer parameter, which the SDK's ``Search`` cannot express.

    ``Search`` only emits ``double`` and ``categorical``, so an ordered discrete
    range could otherwise only be written as ``choices``, which every algorithm
    then treats as unordered categories. Reuses ``Search.uniform`` for the
    feasible-space encoding and retypes it, so the string coercion still matches
    the SDK's own output exactly.
    """
    _check_range(name, spec, "int")
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


# Replicas a training-operator job labels training.kubeflow.org/job-role=master,
# the label Katib defaults primaryPodLabels to for those kinds.
_PRIMARY_REPLICAS = ("Master", "Chief", "Launcher")


def _pod_specs(trial_spec: Any, field: str) -> list[tuple[str, dict[str, Any]]]:
    """``(field path, pod spec)`` for every pod spec written out in a trial spec.

    A bare Pod is its own pod spec and a batch/v1 Job nests one under
    ``spec.template``. A training-operator job (PyTorchJob, TFJob, ...) nests one
    per replica under ``spec.<x>ReplicaSpecs.<replica>.template``, listed here
    primary replica first. A TrainJob has none: its pods come from the runtime.
    """
    spec = trial_spec.get("spec") if isinstance(trial_spec, dict) else None
    if not isinstance(spec, dict):
        return []
    if "containers" in spec:
        return [(f"{field}.spec", spec)]

    templates: list[tuple[str, Any]] = [(f"{field}.spec.template", spec.get("template"))]
    for key, replicas in spec.items():
        if not (key.endswith("ReplicaSpecs") and isinstance(replicas, dict)):
            continue
        for replica in sorted(replicas, key=lambda r: r not in _PRIMARY_REPLICAS):
            entry = replicas[replica]
            template = entry.get("template") if isinstance(entry, dict) else None
            templates.append((f"{field}.spec.{key}.{replica}.template", template))
    return [
        (f"{path}.spec", template["spec"])
        for path, template in templates
        if isinstance(template, dict) and isinstance(template.get("spec"), dict)
    ]


def _pod_container_names(trial_spec: Any, field: str) -> list[tuple[str, list[str]]]:
    """``(pod spec path, container names)`` for every pod spec with a container list."""
    found: list[tuple[str, list[str]]] = []
    for path, pod_spec in _pod_specs(trial_spec, field):
        containers = pod_spec.get("containers")
        if isinstance(containers, list):
            names = [c["name"] for c in containers if isinstance(c, dict) and c.get("name")]
            found.append((path, names))
    return found


def _resolve_primary_container(trial_template: dict[str, Any], override: str | None) -> str:
    """Name the container Katib should collect metrics from.

    Katib needs that name in every trial pod (see :func:`_missing_primary_container`),
    so the default is a container every pod spec in the template has, in the order
    of the first one: for a training-operator job, the primary replica's (its
    framework container, e.g. ``pytorch``). When no name is shared, the primary
    pod's container is returned and the check rejects it.
    """
    if override:
        return override

    named = [names for _, names in _pod_container_names(trial_template, "trial_template") if names]
    if not named:
        # TrainJob and other runtime-backed templates have no inline containers;
        # Trainer names the workload container "node".
        return trainer_constants.NODE
    shared = [n for n in named[0] if all(n in names for names in named[1:])] or named[0]
    # A sidecar can sort first, so prefer the conventional name when present.
    return trainer_constants.NODE if trainer_constants.NODE in shared else shared[0]


def _missing_primary_container(trial_spec: Any, container: str, field: str) -> str | None:
    """Describe the first pod spec Katib would reject for lacking ``container``, if any.

    Katib's pod webhook (``mutatePodEnv`` in v0.19.0) sets KATIB_TRIAL_NAME on the
    primary container of every trial pod, worker replicas included, before it
    looks at primaryPodLabels, and rejects a pod that has no container of that
    name. One replica naming its container differently stops every trial.
    """
    for path, names in _pod_container_names(trial_spec, field):
        if container not in names:
            return (
                f"Container '{container}' is missing from {path}.containers (has {names}). "
                "Katib requires the primary container in every trial pod, so no trial "
                "could start"
            )
    return None


# Katib substitutes ${trialParameters.<name>} throughout the trial spec, so an
# image may legitimately carry one (a tuned tag, say). Each is swapped for "1",
# which is valid anywhere in a reference, a registry port included, before the
# rest is checked.
_TRIAL_PARAMETER_REF = re.compile(r"\$\{trialParameters\.[^}]*\}")
# A reference used as a digest (img@sha256:${trialParameters.digest}) needs a
# placeholder that is itself a valid digest, or every such image is rejected.
_DIGEST_PARAMETER_REF = re.compile(r"(@[a-z0-9]+(?:[+._-][a-z0-9]+)*:)\$\{trialParameters\.[^}]*\}")


def _trial_spec_images(trial_spec: Any, field: str) -> list[tuple[str, Any]]:
    """List ``(field path, image)`` for every container image in a trial spec.

    Covers a TrainJob's ``spec.trainer.image`` and the containers of every pod
    spec :func:`_pod_specs` finds.
    """
    spec = trial_spec.get("spec") if isinstance(trial_spec, dict) else None
    trainer = spec.get("trainer") if isinstance(spec, dict) else None

    images: list[tuple[str, Any]] = []
    if isinstance(trainer, dict) and "image" in trainer:
        images.append((f"{field}.spec.trainer.image", trainer["image"]))
    for pod_path, pod_spec in _pod_specs(trial_spec, field):
        for key in ("initContainers", "containers"):
            containers = pod_spec.get(key)
            if not isinstance(containers, list):
                continue
            for i, container in enumerate(containers):
                if isinstance(container, dict) and "image" in container:
                    images.append((f"{pod_path}.{key}[{i}].image", container["image"]))
    return images


def _validate_trial_images(trial_spec: Any, field: str) -> ToolError | None:
    """Reject trial images Kubernetes could never pull, before anything is created.

    Without this a malformed image only surfaces once every trial pod fails to
    pull it, long after the experiment was accepted.
    """
    for path, image in _trial_spec_images(trial_spec, field):
        candidate = image
        if isinstance(image, str):
            candidate = _DIGEST_PARAMETER_REF.sub(lambda m: m.group(1) + "0" * 64, image)
            candidate = _TRIAL_PARAMETER_REF.sub("1", candidate)
        err = validate_image_reference(candidate, path)
        if err is not None:
            err.details = {"value": image}
            return err
    return None


def _experiment_name_error(name: str, field: str) -> ToolError | None:
    """Reject a name Kubernetes or Katib would refuse, before any preview."""
    err = validate_k8s_name(name, field)
    if err is not None:
        return err
    if len(name) > MAX_EXPERIMENT_NAME_LENGTH or not name[0].isalpha():
        return ToolError(
            error=(
                f"{field} must start with a letter and be at most "
                f"{MAX_EXPERIMENT_NAME_LENGTH} characters, as Katib requires, got '{name}'"
            ),
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    return None


def _template_problem(trial_template: Any) -> str | None:
    """Describe why create_hpo_experiment cannot use a trial template, if it cannot."""
    if not isinstance(trial_template, dict) or not trial_template:
        return "trial_template must be a Kubernetes resource manifest (an object)"
    kind = trial_template.get("kind")
    if kind not in DEFAULTED_TRIAL_KINDS:
        return (
            f"trial_template.kind must be one of {sorted(DEFAULTED_TRIAL_KINDS)}, got {kind!r}. "
            "Katib needs successCondition and failureCondition for any other kind; set them "
            "with create_hpo_experiment_from_spec()"
        )
    # A Job nests its pod under spec.template.spec; each level present must be
    # an object, or container resolution fails on it later.
    node: dict[str, Any] = trial_template
    path: list[str] = []
    for key in ("spec", "template", "spec"):
        path.append(key)
        child = node.get(key)
        if child is None:
            break
        if not isinstance(child, dict):
            return f"trial_template.{'.'.join(path)} must be an object, got {type(child).__name__}"
        node = child
    return None


def _unreferenced_parameters(
    search_space: dict[str, Any], trial_template: dict[str, Any]
) -> list[str]:
    """Search parameters the template never references.

    Katib rejects an Experiment whose trialParameters are not all used in the
    trial template, which is every search_space key here.
    """
    text = json.dumps(trial_template)
    return [name for name in search_space if f"${{trialParameters.{name}}}" not in text]


def _build_experiment(
    *,
    name: str,
    namespace: str,
    objective_metric: str,
    objective_type: str,
    parameters: list[models.V1beta1ParameterSpec],
    trial_template: dict[str, Any],
    algorithm: str,
    max_trial_count: int,
    parallel_trial_count: int,
    max_failed_trials: int | None,
    primary_container_name: str,
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
            parameters=parameters,
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
                primaryContainerName=primary_container_name,
                trialParameters=[
                    models.V1beta1TrialParameterSpec(name=p.name, reference=p.name)
                    for p in parameters
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
        # Creating an Experiment goes through Katib's admission webhooks, so it
        # needs the longer write budget rather than the read timeout.
        _request_timeout=mcp_utils.K8S_WRITE_TIMEOUT,
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
        primary_container_name: Container Katib collects metrics from; every pod
            spec in ``trial_template`` must have one of that name. Defaults to a
            container they all share, in the primary replica's order, or
            Trainer's ``node`` when the template has no inline pod spec (a TrainJob).
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        confirmed: Set ``True`` to submit. ``False`` returns a preview.

    Returns:
        dict: Preview of the Experiment CR, or the creation result.
    """
    name_err = _experiment_name_error(name, "name")
    if name_err is not None:
        return name_err.model_dump()

    image_err = _validate_trial_images(trial_template, "trial_template")
    if image_err is not None:
        return image_err.model_dump()

    template_problem = _template_problem(trial_template)
    if template_problem is not None:
        return ToolError(error=template_problem, error_code=ErrorCode.VALIDATION_ERROR).model_dump()

    primary_container = _resolve_primary_container(trial_template, primary_container_name)
    container_problem = _missing_primary_container(
        trial_template, primary_container, "trial_template"
    )
    if container_problem is not None:
        return ToolError(
            error=container_problem,
            error_code=ErrorCode.VALIDATION_ERROR,
            hint=(
                "Give the metrics container the same name in every replica, or pass "
                "primary_container_name naming a container they all have."
            ),
        ).model_dump()

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
        (
            not isinstance(search_space, dict) or not search_space,
            "search_space must define at least one parameter",
        ),
        (
            not 1 <= max_trial_count <= MAX_TRIAL_COUNT_LIMIT,
            f"max_trial_count must be between 1 and {MAX_TRIAL_COUNT_LIMIT}, got {max_trial_count}",
        ),
        (
            not 1 <= parallel_trial_count <= MAX_PARALLEL_TRIAL_LIMIT,
            f"parallel_trial_count must be between 1 and {MAX_PARALLEL_TRIAL_LIMIT}, "
            f"got {parallel_trial_count}",
        ),
        # Katib's webhook rejects both of these; catching them here keeps the
        # preview honest about what will be admitted.
        (
            parallel_trial_count > max_trial_count,
            f"parallel_trial_count ({parallel_trial_count}) must not exceed "
            f"max_trial_count ({max_trial_count})",
        ),
        (
            max_failed_trials is not None and not 0 <= max_failed_trials <= max_trial_count,
            f"max_failed_trials must be between 0 and max_trial_count ({max_trial_count}), "
            f"got {max_failed_trials}",
        ),
    ]
    for is_invalid, message in problems:
        if is_invalid:
            return ToolError(error=message, error_code=ErrorCode.VALIDATION_ERROR).model_dump()

    try:
        parameters = [_build_parameter(key, spec) for key, spec in search_space.items()]
    except (TypeError, ValueError) as e:
        return ToolError(error=str(e), error_code=ErrorCode.VALIDATION_ERROR).model_dump()

    unreferenced = _unreferenced_parameters(search_space, trial_template)
    if unreferenced:
        return ToolError(
            error=(
                f"search_space parameters not referenced in trial_template: {unreferenced}. "
                "Katib rejects an experiment with a parameter the template never uses"
            ),
            error_code=ErrorCode.VALIDATION_ERROR,
            hint="Reference each as ${trialParameters.<name>} in trial_template, or remove it.",
        ).model_dump()

    # Input checks above run first, as in create_hpo_experiment_from_spec, so a bad
    # request gets the same error whatever namespace it names.
    ns_err = check_optimizer_namespace(namespace)
    if ns_err is not None:
        return ns_err.model_dump()

    try:
        ns = mcp_utils.get_optimizer_effective_namespace(namespace)
    except Exception as e:
        return experiment_error(e, name).model_dump()

    try:
        experiment = _build_experiment(
            name=name,
            namespace=ns,
            objective_metric=objective_metric,
            objective_type=objective_type,
            parameters=parameters,
            trial_template=trial_template,
            algorithm=algorithm,
            max_trial_count=max_trial_count,
            parallel_trial_count=parallel_trial_count,
            max_failed_trials=max_failed_trials,
            primary_container_name=primary_container,
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
        return experiment_error(e, name).model_dump()


# Fields Katib needs in every Experiment spec, as paths under ``spec``.
# ``from_dict`` accepts a manifest missing any of them, so they are checked
# here, where the message can name the field.
_REQUIRED_SPEC_FIELDS: tuple[tuple[str, ...], ...] = (
    ("objective", "type"),
    ("objective", "objectiveMetricName"),
    ("algorithm", "algorithmName"),
    ("trialTemplate",),
)


def _manifest_problem(manifest: dict[str, Any]) -> str | None:
    """Describe the first way a raw Experiment manifest is incomplete, if any."""
    expected_api_version = f"{KATIB_API_GROUP}/{KATIB_API_VERSION}"
    if manifest.get("apiVersion") != expected_api_version:
        return (
            f"spec.apiVersion must be '{expected_api_version}', got {manifest.get('apiVersion')!r}"
        )
    if manifest.get("kind") != EXPERIMENT_KIND:
        return f"spec.kind must be '{EXPERIMENT_KIND}', got {manifest.get('kind')!r}"
    if not isinstance(manifest.get("metadata"), dict):
        return "spec.metadata must be an object with a name"
    body = manifest.get("spec")
    if not isinstance(body, dict):
        return "spec.spec must be an object"
    for path in _REQUIRED_SPEC_FIELDS:
        node: Any = body
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if not node:
            return f"spec.spec.{'.'.join(path)} is required"
    if not body.get("parameters") and not body.get("nasConfig"):
        return "spec.spec.parameters (or spec.spec.nasConfig) is required"
    if body.get("maxTrialCount") is None:
        return (
            "spec.spec.maxTrialCount is required: without it Katib keeps starting trials "
            "with no limit"
        )
    for field, limit in (
        ("maxTrialCount", MAX_TRIAL_COUNT_LIMIT),
        ("parallelTrialCount", MAX_PARALLEL_TRIAL_LIMIT),
    ):
        value = body.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > limit:
            return f"spec.spec.{field} must be at most {limit}, got {value}"
    template = body["trialTemplate"]
    trial_spec = template.get("trialSpec") if isinstance(template, dict) else None
    kind = trial_spec.get("kind") if isinstance(trial_spec, dict) else None
    if (
        kind is not None
        and kind not in DEFAULTED_TRIAL_KINDS
        and not (template.get("successCondition") and template.get("failureCondition"))
    ):
        return (
            f"spec.spec.trialTemplate needs successCondition and failureCondition for a "
            f"{kind} trial; Katib only fills them in for {sorted(DEFAULTED_TRIAL_KINDS)}"
        )
    return None


def create_hpo_experiment_from_spec(
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
            ``metadata.name``, ``spec``). ``spec.maxTrialCount`` is required, at
            most 1000, so the experiment cannot run without limit.
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

    manifest_problem = _manifest_problem(spec)
    if manifest_problem is not None:
        return ToolError(
            error=manifest_problem,
            error_code=ErrorCode.VALIDATION_ERROR,
            hint="Pass a complete Katib v1beta1 Experiment manifest.",
        ).model_dump()

    name = spec["metadata"].get("name")
    if not isinstance(name, str) or not name:
        return ToolError(
            error="spec.metadata.name is required and must be a string",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    name_err = _experiment_name_error(name, "spec.metadata.name")
    if name_err is not None:
        return name_err.model_dump()

    experiment_spec = spec.get("spec")
    trial_template = (
        experiment_spec.get("trialTemplate") if isinstance(experiment_spec, dict) else None
    )
    if isinstance(trial_template, dict):
        trial_spec = trial_template.get("trialSpec")
        image_err = _validate_trial_images(trial_spec, "spec.spec.trialTemplate.trialSpec")
        if image_err is not None:
            return image_err.model_dump()
        container = trial_template.get("primaryContainerName")
        container_problem = (
            _missing_primary_container(trial_spec, container, "spec.spec.trialTemplate.trialSpec")
            if isinstance(container, str) and container
            else None
        )
        if container_problem is not None:
            return ToolError(
                error=container_problem,
                error_code=ErrorCode.VALIDATION_ERROR,
                hint=(
                    "Set spec.spec.trialTemplate.primaryContainerName to a container "
                    "every replica has."
                ),
            ).model_dump()

    # Check the namespace the experiment will actually be created in. The spec
    # can name its own, so checking only the argument would let
    # spec.metadata.namespace bypass the allowlist.
    target_namespace = namespace or spec["metadata"].get("namespace")
    if target_namespace is not None and not isinstance(target_namespace, str):
        return ToolError(
            error="spec.metadata.namespace must be a string",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()
    ns_err = check_optimizer_namespace(target_namespace)
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

    try:
        ns = mcp_utils.get_optimizer_effective_namespace(target_namespace)
    except Exception as e:
        return experiment_error(e, name).model_dump()

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
        return experiment_error(e, name).model_dump()
