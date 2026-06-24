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

"""Agent provider entry-points (avoid importing heavy ollama module for coverage)."""

from importlib.metadata import entry_points


def test_entry_points_registered():
    names = {ep.name for ep in entry_points().select(group="kubeflow_mcp.providers")}
    assert names >= {"ollama", "litellm"}
