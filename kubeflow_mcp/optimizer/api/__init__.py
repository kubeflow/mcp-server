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

"""Optimizer API tools."""

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

__all__ = [
    "katib_pre_flight",
    "list_hpo_experiments",
    "get_hpo_experiment",
    "get_hpo_experiment_status",
    "get_trial",
    "get_successful_trials",
    "list_suggestions",
    "get_hpo_experiment_trials",
    "get_best_trial",
    "get_suggestion",
    "wait_for_hpo_experiment",
    "get_hpo_experiment_trial_logs",
    "get_hpo_experiment_events",
    "create_hpo_experiment",
    "create_hpo_experiment_from_spec",
    "delete_hpo_experiment",
    "update_hpo_experiment",
]
