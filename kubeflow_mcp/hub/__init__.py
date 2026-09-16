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

"""ModelRegistryClient integration for model versioning.

STATUS: Stub module - Ready for contributors (Phase 3)

This module will wrap kubeflow.hub.ModelRegistryClient to provide
model registration and versioning tools via MCP.

Planned Tools (7 total):
    - register_model() → ModelRegistryClient.register_model()
    - list_models() → ModelRegistryClient.list_models()
    - get_model() → ModelRegistryClient.get_model()
    - list_model_versions() → ModelRegistryClient.list_model_versions()
    - get_model_version() → ModelRegistryClient.get_model_version()
    - get_model_artifact() → ModelRegistryClient.get_model_artifact()
    - update_model() → ModelRegistryClient.update_model()

Integration with Trainer:
    After a training job completes, the agent can use register_model()
    to save the trained model to the registry with metadata.

To implement:
    1. Copy trainer/ structure as reference
    2. Add client factory to common/utils.py:
       ```python
       @lru_cache(maxsize=1)
       def get_hub_client() -> "ModelRegistryClient":
           from kubeflow.hub import ModelRegistryClient
           return ModelRegistryClient()
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
    "name": "hub",
    "description": "Model Registry for artifact versioning",
    "sdk_client": "kubeflow.hub.ModelRegistryClient",
    "sdk_version": KUBEFLOW_SDK_VERSION_SPEC,
    "status": "stub",
    "planned_tools": 7,
}

# Tool categories for persona filtering (when implemented)
TOOL_CATEGORIES: dict[str, list[str]] = {
    "discovery": [
        "list_models",
        "get_model",
        "list_model_versions",
        "get_model_version",
        "get_model_artifact",
    ],
    "registration": [
        "register_model",
        "update_model",
    ],
}

# TODO: Implement tools (see CONTRIBUTING.md)
TOOLS: list[Callable[..., dict[str, Any]]] = []

__all__ = ["MODULE_INFO", "TOOLS", "TOOL_CATEGORIES"]
