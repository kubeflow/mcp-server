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

"""Unit tests for E2E helpers. These run without a cluster, unlike the E2E suite."""

import json
from types import SimpleNamespace

import pytest

from tests.e2e.test_kubernetes_e2e import _cleanup_trainjob


class _FakeSession:
    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc

    async def call_tool(self, name, arguments=None):
        if self._exc is not None:
            raise self._exc
        return self._result


def _result(payload: dict, is_error: bool) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))], is_error=is_error)


@pytest.mark.parametrize(
    ("session", "expected"),
    [
        (_FakeSession(_result({"success": True}, is_error=False)), None),
        (
            _FakeSession(_result({"error_code": "RESOURCE_NOT_FOUND"}, is_error=True)),
            None,
        ),
        (
            _FakeSession(_result({"error_code": "KUBERNETES_ERROR"}, is_error=True)),
            "failed",
        ),
        (_FakeSession(exc=RuntimeError("connection lost")), "raised"),
    ],
    ids=["deleted", "already-gone", "error-result", "exception"],
)
async def test_cleanup_trainjob_reports_failures(session, expected):
    error = await _cleanup_trainjob(session, "job-a", "default")

    if expected is None:
        assert error is None
    else:
        assert expected in error
        assert "job-a" in error
