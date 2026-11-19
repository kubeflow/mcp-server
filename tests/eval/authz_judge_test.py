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

"""Judge per-caller authorization (#565) through the real server.

With ``authz.mode: sar``, a call the caller's RBAC does not allow must be refused
before it sends any request to the cluster, in every tool mode, while the same call
by a caller who is allowed goes through.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastmcp import Client
from fastmcp.server.auth.providers.jwt import JWTVerifier

from kubeflow_mcp.core.auth import APIKeyVerifier
from kubeflow_mcp.core.config import AuthzConfig
from kubeflow_mcp.core.server import CLIENT_MODULES, create_server
from tests.eval.helpers import payload

CLIENTS = list(CLIENT_MODULES)
SAR = AuthzConfig(mode="sar", cache_ttl=0)
DELETE = {"name": "job-a", "namespace": "team-a", "confirmed": True}


class Rbac:
    """Answers SubjectAccessReviews: alice may do anything in team-a, nobody else anything."""

    def __init__(self) -> None:
        self.reviews: list[Any] = []

    def create_subject_access_review(self, review: Any, **kwargs: Any) -> Any:
        self.reviews.append(review)
        attrs = review.spec.resource_attributes
        allowed = review.spec.user == "alice" and attrs.namespace == "team-a"
        return SimpleNamespace(status=SimpleNamespace(allowed=allowed, denied=False))


@pytest.fixture
def rbac():
    fake = Rbac()
    # The review is a POST, so it is faked apart from the cluster fake that records writes.
    with patch("kubeflow_mcp.common.utils.get_authorization_api", return_value=fake):
        yield fake


def _server(user: str | None, mode: str = "full"):
    claims = {"sub": user} if user else None
    with patch("kubeflow_mcp.core.authz._verified_claims", lambda: claims):
        return create_server(
            clients=CLIENTS,
            persona="platform-admin",
            mode=mode,
            auth_provider=JWTVerifier(jwks_uri="https://issuer.invalid/jwks.json"),
            authz=SAR,
        )


async def _call(server, tool: str, arguments: dict[str, Any], mode: str):
    async with Client(server) as client:
        if mode == "full":
            return await client.call_tool(tool, arguments, raise_on_error=False)
        return await client.call_tool(
            "execute_tool", {"tool_name": tool, "arguments": arguments}, raise_on_error=False
        )


@pytest.mark.parametrize("mode", ["full", "progressive", "semantic"])
async def test_denied_caller_sends_nothing_to_the_cluster(mode, k8s, rbac):
    result = await _call(_server("bob", mode), "delete_training_job", DELETE, mode)
    body = payload(result)
    assert body.get("error_code") == "PERMISSION_DENIED", body
    assert (
        "delete trainjobs.trainer.kubeflow.org in namespace 'team-a'"
        in (body["details"]["missing_permissions"])
    )
    assert k8s.calls == [], f"{mode}: a refused call still reached the cluster"
    assert {r.spec.user for r in rbac.reviews} == {"bob"}


async def test_allowed_caller_goes_through(k8s, rbac):
    result = await _call(_server("alice"), "delete_training_job", DELETE, "full")
    assert payload(result).get("error_code") != "PERMISSION_DENIED"
    assert any(method == "DELETE" for method, _ in k8s.writes)


async def test_allowed_caller_is_refused_in_another_namespace(k8s, rbac):
    result = await _call(
        _server("alice"), "delete_training_job", {**DELETE, "namespace": "team-b"}, "full"
    )
    assert payload(result).get("error_code") == "PERMISSION_DENIED"
    assert k8s.calls == []


async def test_call_without_a_verified_user_is_refused(k8s, rbac):
    result = await _call(_server(None), "list_training_jobs", {"namespace": "team-a"}, "full")
    assert payload(result).get("error_code") == "PERMISSION_DENIED"
    assert k8s.calls == [] and rbac.reviews == []


@pytest.mark.parametrize(
    "auth_provider", [None, APIKeyVerifier(expected_token="t")], ids=["no-auth", "api-key"]
)
def test_sar_without_per_caller_identity_fails_at_startup(auth_provider):
    with pytest.raises(ValueError, match="needs JWT auth"):
        create_server(clients=CLIENTS, auth_provider=auth_provider, authz=SAR)


def test_authz_off_adds_no_check(k8s, rbac):
    server = create_server(clients=CLIENTS, authz=AuthzConfig())
    assert server is not None and rbac.reviews == []
