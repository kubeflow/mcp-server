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

SDK methods used:
    - OptimizerClient.optimize()                          → create_hpo_experiment
    - CustomObjectsApi.create_namespaced_custom_object()  → create_experiment_from_spec
"""

import logging
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError

logger = logging.getLogger(__name__)

_NOT_IMPLEMENTED = ToolError(
    error="Not yet implemented — planned for Phase 3",
    error_code=ErrorCode.SDK_ERROR,
    hint="This tool is registered but not yet implemented. See KEP-34.",
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
    """Create a hyperparameter optimization experiment.

    Constructs typed SDK objects (``Objective``, ``TrialConfig``, ``Search.*``,
    ``TrainJobTemplate``) from flat MCP parameters, then calls
    ``OptimizerClient.optimize()``.

    The ``search_space`` dict maps parameter names to search specs::

        {
            "lr": {"min": 0.001, "max": 0.1, "type": "uniform"},
            "batch_size": {"choices": [16, 32, 64]},
            "weight_decay": {"min": 0.0001, "max": 0.01, "type": "loguniform"},
        }

    Supported search types: ``uniform``, ``loguniform``, ``choice``.
    Supported algorithms: ``random``, ``grid``.

    Args:
        name: Experiment name (used for tracking, SDK generates the K8s name).
        objective_metric: Metric name to optimize (e.g., "accuracy", "loss").
        search_space: Parameter ranges to explore (see format above).
        trial_template: TrainJob template defining the training script.
        objective_type: ``"maximize"`` or ``"minimize"``. Default: ``"maximize"``.
        algorithm: Optimization algorithm. Default: ``"random"``.
        max_trial_count: Maximum number of trials. Default: 10.
        parallel_trial_count: Trials to run in parallel. Default: 2.
        max_failed_trials: Maximum allowed failed trials. Optional.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        confirmed: Set ``True`` to submit. ``False`` returns a preview.

    Returns:
        dict: Preview config or experiment creation result with generated name.
    """
    return _NOT_IMPLEMENTED


def create_experiment_from_spec(
    spec: dict[str, Any],
    namespace: str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Create an experiment from raw V1beta1Experiment JSON spec.

    Escape hatch for advanced users who need full control over the
    experiment configuration (e.g., advanced algorithms like tpe, cmaes).

    Args:
        spec: Complete V1beta1Experiment spec as a dict.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
        confirmed: Set ``True`` to submit. ``False`` returns a preview.

    Returns:
        dict: Preview config or experiment creation result.
    """
    return _NOT_IMPLEMENTED
