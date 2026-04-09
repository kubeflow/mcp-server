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

"""Hub API tools - Ready for contributors.

This package will contain MCP tools for Model Registry operations.

Suggested file structure:
    api/
    ├── __init__.py         # This file - exports all tools
    ├── registration.py     # register_model, update_model
    └── discovery.py        # list_models, get_model, list_model_versions, etc.

Example tool implementation:

    def register_model(
        name: str,
        version: str,
        artifact_uri: str,
        description: str | None = None,
        labels: dict | None = None,
    ) -> dict:
        '''Register a trained model in the Model Registry.

        Args:
            name: Model name (e.g., "llama-finetuned")
            version: Version string (e.g., "v1.0.0")
            artifact_uri: Storage location (e.g., "s3://bucket/model")
            description: Human-readable description
            labels: Key-value metadata labels

        Returns:
            Registration result with model ID
        '''
        client = get_hub_client()
        model_id = client.register_model(
            name=name,
            version=version,
            artifact_uri=artifact_uri,
            description=description,
            labels=labels,
        )
        return ToolResponse(
            data={"model_id": model_id, "name": name, "version": version}
        ).model_dump()
"""

__all__: list[str] = []
