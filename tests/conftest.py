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

"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _reset_trainer_client_cache():
    """Clear all LRU-cached SDK/K8s clients before and after every test.

    Without this, @lru_cache on get_trainer_client() and _get_api_client()
    preserves mock objects from one test into the next, causing order-dependent
    failures (e.g. a test that primes the cache with a mock client pointing at
    localhost:80 breaks later tests that expect load_config() to be called).
    """
    from kubeflow_mcp.common.utils import reset_clients

    reset_clients()
    yield
    reset_clients()
