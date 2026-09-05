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

"""OptimizerClient integration for Katib hyperparameter tuning.

STATUS: Stub module - Ready for contributors (Phase 2)

This module will wrap kubeflow.optimizer.OptimizerClient to provide
hyperparameter optimization tools via MCP.

Planned Tools (8 total):
    - create_optimization_job() → OptimizerClient.optimize()
    - list_optimization_jobs() → OptimizerClient.list_jobs()
    - get_optimization_job() → OptimizerClient.get_job()
    - get_optimization_logs() → OptimizerClient.get_job_logs()
    - get_optimization_events() → OptimizerClient.get_job_events()
    - get_best_hyperparameters() → OptimizerClient.get_best_results()
    - wait_for_optimization() → OptimizerClient.wait_for_job_status()
    - delete_optimization_job() → OptimizerClient.delete_job()

To implement:
    1. Copy trainer/ structure as reference
    2. Add client factory to common/utils.py:
       ```python
       @lru_cache(maxsize=1)
       def get_optimizer_client() -> "OptimizerClient":
           from kubeflow.optimizer import OptimizerClient
           return OptimizerClient()
       ```
    3. Implement tools in api/ subdirectory
    4. Update TOOLS list below
    5. Add MCP prompts in core/prompts.py

See: docs/CONVENTIONS.md and CONTRIBUTING.md
"""

from collections.abc import Callable
from typing import Any

from kubeflow_mcp.common.constants import KUBEFLOW_SDK_VERSION_SPEC

MODULE_INFO = {
    "name": "optimizer",
    "description": "Hyperparameter tuning with Katib",
    "sdk_client": "kubeflow.optimizer.OptimizerClient",
    "sdk_version": KUBEFLOW_SDK_VERSION_SPEC,
    "status": "stub",
    "planned_tools": 8,
}

# Tool categories for persona filtering (when implemented)
TOOL_CATEGORIES: dict[str, list[str]] = {
    "discovery": [
        "list_optimization_jobs",
        "get_optimization_job",
    ],
    "optimization": [
        "create_optimization_job",
        "get_best_hyperparameters",
    ],
    "monitoring": [
        "get_optimization_logs",
        "get_optimization_events",
        "wait_for_optimization",
    ],
    "lifecycle": [
        "delete_optimization_job",
    ],
}

# TODO: Implement tools (see CONTRIBUTING.md)
TOOLS: list[Callable[..., dict[str, Any]]] = []

__all__ = ["MODULE_INFO", "TOOLS", "TOOL_CATEGORIES"]
