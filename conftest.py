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

import os

import pytest

_MCP_ENV_PREFIX = "KUBEFLOW_MCP_"


@pytest.fixture(autouse=True)
def _isolate_mcp_env():
    """Clear KUBEFLOW_MCP_* env vars so config default tests are deterministic."""
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith(_MCP_ENV_PREFIX)}
    yield
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def _reset_policy_cache():
    """Clear the policy lru_cache between tests to avoid cross-test leakage."""
    from kubeflow_mcp.core.policy import _get_cached_policy

    _get_cached_policy.cache_clear()
    yield
    _get_cached_policy.cache_clear()


@pytest.fixture(autouse=True)
def _reset_circuit_breakers():
    """Clear all circuit breaker state between tests."""
    from kubeflow_mcp.core.resilience import reset_breakers

    reset_breakers()
    yield
    reset_breakers()


@pytest.fixture
def mock_env_vars():
    """Fixture to set and clean up environment variables."""
    original_env = dict(os.environ)

    def _set_env_vars(**kwargs):
        for key, value in kwargs.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        return os.environ

    yield _set_env_vars

    os.environ.clear()
    os.environ.update(original_env)
