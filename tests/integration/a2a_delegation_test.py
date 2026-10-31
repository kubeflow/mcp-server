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

"""End-to-end A2A delegation against a real server, with the SDK mocked.

Covers the acceptance criterion in #183: "fine-tune model X on dataset Y"
delegated end to end. The server, routes, auth, persona filtering, audit
wrapping and tool dispatch are all real; only the Kubeflow SDK call at the
very bottom is replaced, so the test stays fast and needs no cluster.

The unit tests in ``kubeflow_mcp/core/a2a_test.py`` exercise the endpoint
against stub tools. This file is the one that proves the whole path holds
together against the actual trainer tools.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from kubeflow_mcp.core.a2a import A2A_PATH, AGENT_CARD_PATH
from kubeflow_mcp.core.auth import APIKeyVerifier
from kubeflow_mcp.core.server import create_server

TOKEN = "integration-secret"

PATCH_CLIENT = "kubeflow_mcp.trainer.api.training._get_client"
PATCH_NS_CHECK = "kubeflow_mcp.trainer.api.training.check_namespace_allowed"
PATCH_GPU_CHECK = "kubeflow_mcp.trainer.api.training._check_gpu_available"


def _delegate(tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """A standard A2A message/send request carrying a structured DataPart."""
    return {
        "jsonrpc": "2.0",
        "id": "itest-1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "kind": "message",
                "messageId": "m-itest-1",
                "parts": [{"kind": "data", "data": {"tool": tool, "arguments": arguments or {}}}],
            }
        },
    }


def _result_payload(body: dict[str, Any]) -> dict[str, Any]:
    assert "error" not in body, body["error"]
    return body["result"]["parts"][0]["data"]


@pytest.fixture
def trainer_sdk() -> Iterator[MagicMock]:
    """Replace the Kubeflow SDK boundary, leaving everything above it real."""
    with (
        patch(PATCH_GPU_CHECK, return_value=None),
        patch(PATCH_NS_CHECK, return_value=None),
        patch(PATCH_CLIENT) as client_factory,
    ):
        client = MagicMock()
        client.train.return_value = "train-gemma-itest"
        client_factory.return_value = client
        yield client


@pytest.fixture
def authenticated_client() -> Iterator[TestClient]:
    mcp = create_server(persona="data-scientist", auth_provider=APIKeyVerifier(TOKEN))
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        yield client


def _post(client: TestClient, payload: dict[str, Any]):
    return client.post(A2A_PATH, json=payload, headers={"Authorization": f"Bearer {TOKEN}"})


def test_fine_tune_delegation_previews_then_submits(
    authenticated_client: TestClient, trainer_sdk: MagicMock
) -> None:
    """The documented minimum viable delegation, end to end."""
    preview = _result_payload(
        _post(
            authenticated_client,
            _delegate("fine_tune", {"model": "hf://org/model", "dataset": "hf://org/ds"}),
        ).json()
    )

    assert preview["status"] == "preview"
    assert trainer_sdk.train.call_count == 0, "a preview must not reach the SDK"

    submitted = _result_payload(
        _post(
            authenticated_client,
            _delegate(
                "fine_tune",
                {
                    "model": "hf://org/model",
                    "dataset": "hf://org/ds",
                    "runtime": "torchtune-llama",
                    "confirmed": True,
                },
            ),
        ).json()
    )

    assert submitted["success"] is True
    assert submitted["data"]["job_name"] == "train-gemma-itest"
    assert submitted["data"]["status"] == "Created"
    trainer_sdk.train.assert_called_once()


def test_delegation_requires_a_credential(trainer_sdk: MagicMock) -> None:
    mcp = create_server(persona="data-scientist", auth_provider=APIKeyVerifier(TOKEN))
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        response = client.post(
            A2A_PATH,
            json=_delegate("fine_tune", {"model": "hf://org/model", "dataset": "hf://org/ds"}),
        )

    assert response.status_code == 401
    assert trainer_sdk.train.call_count == 0


def test_persona_refuses_a_tool_it_does_not_allow(trainer_sdk: MagicMock) -> None:
    """readonly cannot submit training, and the card does not offer it."""
    mcp = create_server(persona="readonly", auth_provider=APIKeyVerifier(TOKEN))
    with TestClient(mcp.http_app(transport="streamable-http")) as client:
        card = client.get(AGENT_CARD_PATH).json()
        refused = _post(
            client,
            _delegate(
                "fine_tune",
                {"model": "hf://org/model", "dataset": "hf://org/ds", "confirmed": True},
            ),
        ).json()

    assert "fine-tuning" not in {skill["id"] for skill in card["skills"]}
    assert refused["error"]["code"] == -32004
    assert trainer_sdk.train.call_count == 0, "a refused delegation must not reach the SDK"


def test_agent_card_advertises_the_endpoint_that_serves_it(
    authenticated_client: TestClient,
) -> None:
    card = authenticated_client.get(AGENT_CARD_PATH).json()

    assert card["url"].endswith(A2A_PATH)
    assert card["securitySchemes"]["bearerAuth"]["scheme"] == "bearer"
    assert "fine-tuning" in {skill["id"] for skill in card["skills"]}


def test_malformed_delegation_never_reaches_the_sdk(
    authenticated_client: TestClient, trainer_sdk: MagicMock
) -> None:
    request = _delegate(
        "fine_tune", {"model": "hf://org/model", "dataset": "hf://org/ds", "confirmed": True}
    )
    request["params"]["message"]["role"] = "agent"

    body = _post(authenticated_client, request).json()

    assert body["error"]["code"] == -32602
    assert trainer_sdk.train.call_count == 0
