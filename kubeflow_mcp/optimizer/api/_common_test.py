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


@pytest.mark.parametrize("status", [401, 403, 500])
@pytest.mark.parametrize("wrapping", ["direct", "cause", "context", "nested"])
def test_denials_do_not_count_as_outages(status: int, wrapping: str) -> None:
    error: Exception = ApiException(status=status)
    if wrapping != "direct":
        cause = error
        error = RuntimeError("Katib request failed")
        if wrapping == "context":
            error.__context__ = cause
        else:
            error.__cause__ = cause
        if wrapping == "nested":
            cause = error
            error = RuntimeError("Experiment request failed")
            error.__cause__ = cause
    resource = resource_error(error, "exp")
    collection = collection_error(error)
    for result in (resource, collection):
        assert result.error_code == ("PERMISSION_DENIED" if status in (401, 403) else "SDK_ERROR")
        assert is_infrastructure_error(result.model_dump()) is (status == 500)
    if status in (401, 403):
        assert resource.hint == collection.hint
