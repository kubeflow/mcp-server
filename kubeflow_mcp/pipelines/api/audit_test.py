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

"""Audit trail: every pipelines tool call is logged, with secrets masked."""

from unittest.mock import MagicMock

import pytest

from kubeflow_mcp.core import server
from kubeflow_mcp.pipelines.api.monitoring import get_pipeline_run
from kubeflow_mcp.pipelines.api.runs import create_pipeline_run, stop_pipeline_run
from kubeflow_mcp.pipelines.conftest import RUN_ID, make_pipeline, make_run


@pytest.fixture
def audit_log(monkeypatch):
    log = MagicMock()
    monkeypatch.setattr(server, "logger", log)
    return log


def _audit_entries(log: MagicMock) -> list[dict]:
    return [call.kwargs["extra"] for call in log.info.call_args_list if call.args == ("tool_call",)]


def test_create_run_is_audited_with_secrets_masked(kfp_client, audit_log):
    kfp_client.get_pipeline.return_value = make_pipeline()
    kfp_client.run.return_value = make_run("PENDING")

    server._audit_wrap(create_pipeline_run)(
        pipeline="churn-prediction",
        params={"learning_rate": 0.1, "api_token": "super-secret-value"},
        confirmed=True,
    )

    [entry] = _audit_entries(audit_log)
    assert entry["audit"] is True
    assert entry["tool"] == "create_pipeline_run"
    assert entry["success"] is True
    assert entry["parameters"]["pipeline"] == "churn-prediction"
    assert entry["parameters"]["params"]["learning_rate"] == 0.1
    assert "super-secret-value" not in str(entry["parameters"])


@pytest.mark.parametrize(
    ("tool", "kwargs", "expected_success"),
    [
        (get_pipeline_run, {"run_id": RUN_ID}, True),
        (stop_pipeline_run, {"run_id": RUN_ID, "confirmed": True}, True),
        (get_pipeline_run, {"run_id": "not-a-uuid"}, False),
    ],
)
def test_reads_writes_and_refusals_are_all_audited(
    kfp_client, audit_log, tool, kwargs, expected_success
):
    kfp_client.get_run.return_value = make_run("RUNNING")

    server._audit_wrap(tool)(**kwargs)

    [entry] = _audit_entries(audit_log)
    assert entry["tool"] == tool.__name__
    assert entry["success"] is expected_success
