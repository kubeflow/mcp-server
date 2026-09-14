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

from unittest import mock

from fastmcp import FastMCP
from starlette.testclient import TestClient

from kubeflow_mcp.core.a2a import register_a2a_routes
from kubeflow_mcp.core.auth import APIKeyVerifier


def _a2a_client(
    *, authenticated: bool = False, clients=None, persona="data-scientist"
) -> TestClient:
    if clients is None:
        clients = ["trainer"]
    auth = APIKeyVerifier(expected_token="secret") if authenticated else None
    mcp = FastMCP("test-server", auth=auth)
    register_a2a_routes(mcp, clients=clients, persona=persona, auth_provider=auth)
    return TestClient(mcp.http_app(transport="streamable-http"))


def test_agent_card_unauthenticated():
    with _a2a_client() as client:
        response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "kubeflow-mcp-server"
    assert "fine-tune" in data["skills"]
    assert data["endpoints"]["a2a"] == "/a2a"


def test_a2a_endpoint_requires_auth():
    with _a2a_client(authenticated=True) as client:
        response = client.post(
            "/a2a", json={"method": "tasks/send", "id": 1, "params": {"task": "test"}}
        )
    assert response.status_code == 401


@mock.patch("kubeflow_mcp.core.a2a.fine_tune")
def test_a2a_tasks_send_finetune(mock_fine_tune):
    mock_fine_tune.return_value = {"status": "success"}

    with _a2a_client(authenticated=True) as client:
        response = client.post(
            "/a2a",
            headers={"Authorization": "Bearer secret"},
            json={
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "id": 1,
                "params": {"task": "fine-tune model llama3 on dataset my-data"},
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["result"] == {"status": "success"}
    assert data["id"] == 1

    mock_fine_tune.assert_called_once_with(model="llama3", dataset="my-data", confirmed=True)


def test_a2a_tasks_send_invalid_task():
    with _a2a_client(authenticated=True) as client:
        response = client.post(
            "/a2a",
            headers={"Authorization": "Bearer secret"},
            json={
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "id": 2,
                "params": {"task": "do something unknown"},
            },
        )

    assert response.status_code == 400
    data = response.json()
    assert data["error"]["code"] == -32602
