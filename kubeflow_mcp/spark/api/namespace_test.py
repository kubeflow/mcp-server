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

"""Namespace sweep for every spark tool.

``docs/CONVENTIONS.md`` requires a namespace sweep before a module may claim
``status: implemented``: every tool must run ``check_namespace_allowed()`` and
must fail closed when the namespace is denied — including when ``namespace`` is
omitted, which is the case the "Do not" list calls out explicitly.
"""

from unittest.mock import patch

import pytest

from kubeflow_mcp.common.types import ToolError as ToolErrorModel
from kubeflow_mcp.spark import TOOLS
from kubeflow_mcp.spark.api import discovery, monitoring, sessions

# (tool callable, module it lives in, kwargs sufficient to reach the ns check)
SPARK_TOOL_CASES = [
    (discovery.list_spark_sessions, discovery, {}),
    (discovery.get_spark_session, discovery, {"name": "x"}),
    (monitoring.get_spark_session_logs, monitoring, {"name": "x"}),
    (sessions.create_spark_session, sessions, {"name": "x", "confirmed": True}),
    (sessions.delete_spark_session, sessions, {"name": "x", "confirmed": True}),
]


def _denied(*_args, **_kwargs):
    return ToolErrorModel(error="namespace blocked", error_code="PERMISSION_DENIED")


def test_sweep_covers_every_registered_tool():
    """The sweep must not silently drift out of date as tools are added."""
    swept = {tool.__name__ for tool, _mod, _kwargs in SPARK_TOOL_CASES}
    assert swept == {t.__name__ for t in TOOLS}


@pytest.mark.parametrize(
    ("tool", "module", "kwargs"),
    SPARK_TOOL_CASES,
    ids=[t.__name__ for t, _m, _k in SPARK_TOOL_CASES],
)
class TestNamespaceEnforcement:
    def test_denied_namespace_is_refused(self, tool, module, kwargs):
        with patch.object(module, "check_namespace_allowed", _denied):
            out = tool(namespace="forbidden-ns", **kwargs)
        assert out["success"] is False
        assert out["error_code"] == "PERMISSION_DENIED"

    def test_omitted_namespace_still_checked(self, tool, module, kwargs):
        """Fail closed on an implicit default namespace."""
        with patch.object(module, "check_namespace_allowed") as ns_check:
            ns_check.return_value = _denied()
            out = tool(**kwargs)
        ns_check.assert_called_once()
        assert out["success"] is False
        assert out["error_code"] == "PERMISSION_DENIED"

    def test_denial_happens_before_any_sdk_call(self, tool, module, kwargs):
        """A denied namespace must never reach the SparkClient."""
        with (
            patch.object(module, "check_namespace_allowed", _denied),
            patch.object(module, "get_spark_client_for_namespace") as client_fn,
        ):
            tool(namespace="forbidden-ns", **kwargs)
        client_fn.assert_not_called()
