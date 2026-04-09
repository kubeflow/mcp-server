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

"""Trainer API tools."""

from kubeflow_mcp.trainer.api.discovery import (
    get_runtime,
    get_training_job,
    list_runtimes,
    list_training_jobs,
)
from kubeflow_mcp.trainer.api.lifecycle import (
    delete_training_job,
    update_training_job,
)
from kubeflow_mcp.trainer.api.monitoring import (
    get_training_events,
    get_training_logs,
    wait_for_training,
)
from kubeflow_mcp.trainer.api.planning import (
    check_compatibility,
    estimate_resources,
    get_cluster_resources,
    pre_flight,
)
from kubeflow_mcp.trainer.api.platform import (
    create_runtime,
    delete_runtime,
    inspect_controller,
    inspect_crd,
    patch_runtime,
)
from kubeflow_mcp.trainer.api.training import (
    fine_tune,
    run_container_training,
    run_custom_training,
)

__all__ = [
    "pre_flight",
    "check_compatibility",
    "get_cluster_resources",
    "estimate_resources",
    "fine_tune",
    "run_custom_training",
    "run_container_training",
    "list_training_jobs",
    "get_training_job",
    "list_runtimes",
    "get_runtime",
    "get_training_logs",
    "get_training_events",
    "wait_for_training",
    "delete_training_job",
    "update_training_job",
    "inspect_crd",
    "inspect_controller",
    "patch_runtime",
    "create_runtime",
    "delete_runtime",
]
