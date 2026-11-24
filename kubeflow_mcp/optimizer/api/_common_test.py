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

"""Unit tests for the optimizer's shared error mapping."""

import logging

import pytest
from kubernetes.client.exceptions import ApiException

from kubeflow_mcp.common.constants import is_infrastructure_error
from kubeflow_mcp.optimizer.api._common import collection_error, is_conflict, resource_error

_LOGGER = "kubeflow_mcp.optimizer.api._common"


@pytest.mark.parametrize("status", [403, 404, 409])
def test_caller_mistakes_are_not_logged_as_failures(caplog, status):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        resource_error(ApiException(status=status), "exp")
    assert caplog.records == []


def test_infrastructure_failures_are_logged_once_with_the_trace(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        err = resource_error(RuntimeError("connection reset"), "exp")
        collection_error(RuntimeError("connection reset"))
    assert err.error_code == "SDK_ERROR"
    assert len(caplog.records) == 2
    assert all(r.exc_info for r in caplog.records)


def test_conflict_is_found_through_the_cause_chain():
    wrapped = RuntimeError("patch failed")
    wrapped.__cause__ = ApiException(status=409)
    assert is_conflict(wrapped)
    assert not is_conflict(ApiException(status=404))


def test_a_forbidden_resource_does_not_count_as_an_outage():
    """RBAC covering only some allowed namespaces must not open the circuit
    breaker for the namespaces it does cover."""
    wrapped = RuntimeError("get failed")
    wrapped.__cause__ = ApiException(status=403, reason="Forbidden")
    err = resource_error(wrapped, "exp")
    assert err.error_code == "PERMISSION_DENIED"
    assert err.hint == collection_error(ApiException(status=403)).hint
    assert not is_infrastructure_error(err.model_dump())
