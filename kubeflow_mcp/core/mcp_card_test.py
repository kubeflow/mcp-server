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

import pytest
from fastmcp import FastMCP
from starlette.testclient import TestClient

from kubeflow_mcp.core.mcp_card import (
    SERVER_CARD_PATH,
    SERVER_CARD_SCHEMA,
    SERVER_CARD_WELL_KNOWN_PATH,
    build_server_card,
    register_server_card_routes,
)


def _card_client(*, authenticated: bool = False) -> TestClient:
    from kubeflow_mcp.core.auth import APIKeyVerifier

    auth = APIKeyVerifier(expected_token="secret") if authenticated else None
    mcp = FastMCP("test-server", auth=auth)
    register_server_card_routes(mcp)
    return TestClient(mcp.http_app(transport="streamable-http"))


@pytest.mark.parametrize("path", [SERVER_CARD_PATH, SERVER_CARD_WELL_KNOWN_PATH])
def test_server_card_served_on_both_discovery_paths(path: str) -> None:
    """SEP-2127 has not settled on one path, so both must resolve."""
    with _card_client() as client:
        response = client.get(path)

    assert response.status_code == 200
    assert response.json()["name"] == "io.github.kubeflow/mcp-server"


def test_server_card_carries_required_schema_fields() -> None:
    with _card_client() as client:
        card = client.get(SERVER_CARD_PATH).json()

    assert card["$schema"] == SERVER_CARD_SCHEMA
    for field in ("name", "version", "description"):
        assert card.get(field), f"required field '{field}' missing from server card"


def test_server_card_name_uses_reverse_dns_with_single_slash() -> None:
    """The schema requires exactly one slash separating namespace from name."""
    card = build_server_card()

    assert card["name"].count("/") == 1


def test_server_card_omits_primitives() -> None:
    """SEP-2127 excludes tools/resources/prompts: the runtime surface is dynamic."""
    with _card_client() as client:
        card = client.get(SERVER_CARD_PATH).json()

    for excluded in ("tools", "resources", "prompts", "skills"):
        assert excluded not in card


def test_server_card_omits_local_package_metadata() -> None:
    """``packages`` belongs to server.json, not to a .well-known document."""
    with _card_client() as client:
        card = client.get(SERVER_CARD_PATH).json()

    assert "packages" not in card


def test_server_card_advertises_absolute_remote_endpoint() -> None:
    with _card_client() as client:
        card = client.get(SERVER_CARD_PATH).json()

    remotes = card["remotes"]
    assert remotes[0]["type"] == "streamable-http"
    assert remotes[0]["url"].startswith("http")
    assert remotes[0]["url"].endswith("/mcp")


def test_server_card_omits_remotes_when_url_unknown() -> None:
    """Better to advertise no endpoint than a relative one a crawler cannot use."""
    assert "remotes" not in build_server_card(mcp_url=None)


def test_server_card_sets_cors_and_cache_headers() -> None:
    with _card_client() as client:
        response = client.get(SERVER_CARD_PATH)

    assert response.headers["access-control-allow-origin"] == "*"
    assert response.headers["cache-control"] == "public, max-age=3600"
    assert response.headers["content-type"].startswith("application/json")


def test_server_card_stays_public_when_auth_is_enabled() -> None:
    """Discovery has to work before the caller holds a credential."""
    with _card_client(authenticated=True) as client:
        assert client.get(SERVER_CARD_PATH).status_code == 200


def test_server_card_falls_back_when_server_json_unreadable(monkeypatch) -> None:
    """A wheel that did not ship server.json still serves a valid card."""
    import kubeflow_mcp.core.mcp_card as mcp_card

    monkeypatch.setattr(mcp_card, "_load_server_json", lambda: {})

    card = mcp_card.build_server_card()

    assert card["name"] == "io.github.kubeflow/mcp-server"
    assert card["version"]
    assert card["description"]
    assert card["$schema"] == SERVER_CARD_SCHEMA
