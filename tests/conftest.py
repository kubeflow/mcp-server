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

"""Pytest configuration."""

import pytest


@pytest.fixture(autouse=True)
def _reset_trainer_client_cache():
    """Clear :func:`get_trainer_client` LRU cache so tests never share a real cluster client.

    Patches replace symbols on modules, but unpatching restores the original cached
    wrapper; without clearing, a prior test can leave a real ``TrainerClient`` in the
    cache and cause flakes or hangs (e.g. follow-on tests that expect mocks).

    No-ops gracefully on the skeleton branch where utils doesn't exist yet.
    """
    try:
        from kubeflow_mcp.common.utils import reset_clients
    except ImportError:
        yield
        return

    reset_clients()
    yield
    reset_clients()


@pytest.fixture
def mock_k8s_client():
    """Mock Kubernetes client for testing."""
    return None
