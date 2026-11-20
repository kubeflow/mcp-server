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

from types import SimpleNamespace
from typing import Any

import pytest
from fastmcp import Client, FastMCP

from kubeflow_mcp.common.types import ToolPermission
from kubeflow_mcp.core.authz import (
    AuthzError,
    CallerIdentity,
    InvalidArguments,
    SarAuthorizationMiddleware,
    SubjectAccessReviewer,
    _controller_namespace,
    describe,
    identity_from_claims,
    required_checks,
)
from kubeflow_mcp.core.config import AuthzConfig

DELETE_JOB = ToolPermission("delete", "trainer.kubeflow.org", "trainjobs")
GET_JOB = ToolPermission("get", "trainer.kubeflow.org", "trainjobs")
CREATE_POD = ToolPermission("create", "", "pods", when="include_packages")
POD_LOG = ToolPermission("get", "", "pods/log")
LIST_NODES = ToolPermission("list", "", "nodes", cluster_wide=True)
ALICE = CallerIdentity("alice", ("team-a",))


class FakeAuthorizationApi:
    """Answers SubjectAccessReviews from a set of allowed (user, verb, resource, namespace)."""

    def __init__(self, allowed=(), error: Exception | None = None):
        self.allowed = set(allowed)
        self.error = error
        self.reviews: list[Any] = []

    def create_subject_access_review(self, review, **kwargs):
        self.reviews.append(review)
        if self.error is not None:
            raise self.error
        attrs = review.spec.resource_attributes
        resource = attrs.resource + (f"/{attrs.subresource}" if attrs.subresource else "")
        key = (review.spec.user, attrs.verb, resource, attrs.namespace)
        return SimpleNamespace(status=SimpleNamespace(allowed=key in self.allowed, denied=False))


def _default_namespace(namespace):
    return namespace or "default"


class TestIdentity:
    def test_user_and_groups_from_claims_with_prefixes(self):
        config = AuthzConfig(mode="sar", username_prefix="oidc:", groups_prefix="oidc:")
        identity = identity_from_claims({"sub": "alice", "groups": ["team-a", 7, ""]}, config)
        assert identity == CallerIdentity("oidc:alice", ("oidc:team-a", "system:authenticated"))

    def test_reserved_groups_are_dropped(self):
        claims = {"sub": "mallory", "groups": ["system:masters", "team-a"]}
        identity = identity_from_claims(claims, AuthzConfig(mode="sar"))
        assert identity.groups == ("team-a", "system:authenticated")

    def test_a_prefix_keeps_claims_out_of_reserved_names(self):
        config = AuthzConfig(mode="sar", username_prefix="oidc:", groups_prefix="oidc:")
        identity = identity_from_claims(
            {"sub": "system:admin", "groups": ["system:masters"]}, config
        )
        assert identity == CallerIdentity(
            "oidc:system:admin", ("oidc:system:masters", "system:authenticated")
        )

    def test_custom_claims_and_single_group_string(self):
        config = AuthzConfig(mode="sar", user_claim="email", groups_claim="roles")
        identity = identity_from_claims({"email": "a@example.com", "roles": "admins"}, config)
        assert identity == CallerIdentity("a@example.com", ("admins", "system:authenticated"))

    @pytest.mark.parametrize(
        "claims",
        [
            {},
            {"sub": ""},
            {"sub": 42},
            {"auth_method": "api_key"},
            {"sub": "system:admin"},
            {"sub": "system:serviceaccount:kube-system:replicaset-controller"},
        ],
    )
    def test_no_usable_user_claim(self, claims):
        assert identity_from_claims(claims, AuthzConfig(mode="sar")) is None


class TestRequiredChecks:
    def test_namespace_from_argument(self):
        checks = required_checks((DELETE_JOB,), {"namespace": "team-a"}, _default_namespace)
        assert checks == [(DELETE_JOB, "team-a")]

    def test_default_namespace_when_omitted(self):
        assert required_checks((DELETE_JOB,), {}, _default_namespace) == [(DELETE_JOB, "default")]

    def test_when_skips_until_the_argument_is_set(self):
        perms = (GET_JOB, CREATE_POD)
        assert required_checks(perms, {}, _default_namespace) == [(GET_JOB, "default")]
        assert required_checks(perms, {"include_packages": True}, _default_namespace) == [
            (GET_JOB, "default"),
            (CREATE_POD, "default"),
        ]

    def test_cluster_wide_has_no_namespace_and_needs_no_resolver(self):
        def unresolvable(namespace):
            raise AssertionError("not needed")

        assert required_checks((LIST_NODES,), {}, unresolvable) == [(LIST_NODES, None)]

    def test_resolver_without_a_namespace_checks_cluster_wide(self):
        assert required_checks((POD_LOG,), {}, lambda namespace: None) == [(POD_LOG, None)]

    def test_controller_namespace_follows_the_tool(self, monkeypatch):
        monkeypatch.setattr(
            "kubeflow_mcp.trainer.api.platform._get_controller_namespace", lambda: None
        )
        assert _controller_namespace(None) is None
        assert _controller_namespace("kubeflow-system") == "kubeflow-system"
        monkeypatch.setattr(
            "kubeflow_mcp.trainer.api.platform._get_controller_namespace", lambda: "trainer"
        )
        assert _controller_namespace(None) == "trainer"

    @pytest.mark.parametrize("namespace", ["Team_A", "../kube-system", "a" * 64, ""])
    def test_invalid_namespace_is_refused_before_any_review(self, namespace):
        with pytest.raises(InvalidArguments):
            required_checks((GET_JOB,), {"namespace": namespace}, _default_namespace)

    def test_non_string_namespace_is_refused(self):
        with pytest.raises(InvalidArguments):
            required_checks((GET_JOB,), {"namespace": ["a", "b"]}, _default_namespace)


def test_describe_names_resource_like_kubectl():
    assert describe(DELETE_JOB, "team-a") == (
        "delete trainjobs.trainer.kubeflow.org in namespace 'team-a'"
    )
    assert describe(POD_LOG, "ns") == "get pods/log in namespace 'ns'"
    assert describe(LIST_NODES, None) == "list nodes cluster-wide"


class TestReviewer:
    def test_sends_caller_and_resource_attributes(self):
        api = FakeAuthorizationApi(allowed={("alice", "get", "pods/log", "team-a")})
        reviewer = SubjectAccessReviewer(lambda: api, cache_ttl=0)
        assert reviewer.missing(ALICE, [(POD_LOG, "team-a")]) == []
        spec = api.reviews[0].spec
        assert (spec.user, spec.groups) == ("alice", ["team-a"])
        attrs = spec.resource_attributes
        assert (attrs.verb, attrs.group, attrs.resource, attrs.subresource, attrs.namespace) == (
            "get",
            "",
            "pods",
            "log",
            "team-a",
        )

    def test_lists_every_missing_permission(self):
        api = FakeAuthorizationApi(allowed={("alice", "get", "trainjobs", "team-a")})
        reviewer = SubjectAccessReviewer(lambda: api, cache_ttl=0)
        checks = [(GET_JOB, "team-a"), (DELETE_JOB, "team-a"), (LIST_NODES, None)]
        assert reviewer.missing(ALICE, checks) == [
            "delete trainjobs.trainer.kubeflow.org in namespace 'team-a'",
            "list nodes cluster-wide",
        ]

    def test_explicit_deny_wins(self):
        class DenyingApi(FakeAuthorizationApi):
            def create_subject_access_review(self, review, **kwargs):
                return SimpleNamespace(status=SimpleNamespace(allowed=True, denied=True))

        reviewer = SubjectAccessReviewer(lambda: DenyingApi(), cache_ttl=0)
        assert reviewer.missing(ALICE, [(GET_JOB, "team-a")]) != []

    @pytest.mark.parametrize(
        "api",
        [
            FakeAuthorizationApi(error=RuntimeError("timed out")),
            SimpleNamespace(create_subject_access_review=lambda *a, **k: SimpleNamespace()),
        ],
        ids=["api-error", "no-status"],
    )
    def test_undecided_raises(self, api):
        with pytest.raises(AuthzError):
            SubjectAccessReviewer(lambda: api, cache_ttl=0).missing(ALICE, [(GET_JOB, "ns")])

    def test_cache_reuses_a_decision_until_it_expires(self):
        api = FakeAuthorizationApi(allowed={("alice", "get", "trainjobs", "ns")})
        now = [0.0]
        reviewer = SubjectAccessReviewer(lambda: api, cache_ttl=30, clock=lambda: now[0])
        reviewer.missing(ALICE, [(GET_JOB, "ns")])
        now[0] = 29.0
        reviewer.missing(ALICE, [(GET_JOB, "ns")])
        assert len(api.reviews) == 1
        now[0] = 30.0
        reviewer.missing(ALICE, [(GET_JOB, "ns")])
        assert len(api.reviews) == 2

    def test_cache_is_per_group_set(self):
        api = FakeAuthorizationApi()
        reviewer = SubjectAccessReviewer(lambda: api, cache_ttl=30, clock=lambda: 0.0)
        reviewer.missing(CallerIdentity("alice", ("team-a",)), [(GET_JOB, "ns")])
        reviewer.missing(CallerIdentity("alice", ("team-b",)), [(GET_JOB, "ns")])
        assert len(api.reviews) == 2

    def test_cache_is_per_caller(self):
        api = FakeAuthorizationApi(allowed={("alice", "get", "trainjobs", "ns")})
        reviewer = SubjectAccessReviewer(lambda: api, cache_ttl=30, clock=lambda: 0.0)
        assert reviewer.missing(ALICE, [(GET_JOB, "ns")]) == []
        assert reviewer.missing(CallerIdentity("bob", ()), [(GET_JOB, "ns")]) != []

    def test_cache_stays_bounded_whatever_namespaces_callers_send(self, monkeypatch):
        monkeypatch.setattr("kubeflow_mcp.core.authz.MAX_CACHE_ENTRIES", 10)
        now = [0.0]
        reviewer = SubjectAccessReviewer(
            lambda: FakeAuthorizationApi(), cache_ttl=30, clock=lambda: now[0]
        )
        for i in range(10):
            reviewer.missing(ALICE, [(GET_JOB, f"old-{i}")])
        now[0] = 100.0  # the cache is full of expired entries: only they go
        reviewer.missing(ALICE, [(GET_JOB, "fresh")])
        assert len(reviewer._cache) == 1
        for i in range(25):  # a burst of fresh ones never passes the cap
            reviewer.missing(ALICE, [(GET_JOB, f"ns-{i}")])
        assert len(reviewer._cache) <= 10

    def test_zero_ttl_never_caches(self):
        api = FakeAuthorizationApi()
        reviewer = SubjectAccessReviewer(lambda: api, cache_ttl=0, clock=lambda: 0.0)
        reviewer.missing(ALICE, [(GET_JOB, "ns")])
        reviewer.missing(ALICE, [(GET_JOB, "ns")])
        assert len(api.reviews) == 2


def _server(api, claims, calls):
    """A server with one guarded tool, execute_tool and an unguarded meta-tool."""
    mcp = FastMCP("authz-test")

    def delete_training_job(name: str, namespace: str | None = None) -> dict:
        calls.append(("delete_training_job", name, namespace))
        return {"deleted": name}

    def execute_tool(tool_name: str, arguments: dict | None = None) -> dict:
        calls.append(("execute_tool", tool_name))
        return {"ran": tool_name}

    def find_tools(query: str) -> dict:
        calls.append(("find_tools", query))
        return {"tools": []}

    for tool in (delete_training_job, execute_tool, find_tools):
        mcp.tool()(tool)
    mcp.add_middleware(
        SarAuthorizationMiddleware(
            AuthzConfig(mode="sar"),
            {"delete_training_job": (DELETE_JOB,)},
            {"delete_training_job": _default_namespace},
            SubjectAccessReviewer(lambda: api, cache_ttl=0),
            lambda: claims,
        )
    )
    return mcp


ALLOW_ALICE = {("alice", "delete", "trainjobs", "team-a")}


class TestMiddleware:
    async def test_allowed_call_runs(self):
        calls = []
        mcp = _server(FakeAuthorizationApi(ALLOW_ALICE), {"sub": "alice"}, calls)
        async with Client(mcp) as client:
            result = await client.call_tool(
                "delete_training_job", {"name": "j", "namespace": "team-a"}
            )
        assert not result.is_error
        assert calls == [("delete_training_job", "j", "team-a")]

    async def test_denied_call_never_runs_and_names_the_permission(self):
        calls = []
        mcp = _server(FakeAuthorizationApi(ALLOW_ALICE), {"sub": "bob"}, calls)
        async with Client(mcp) as client:
            result = await client.call_tool(
                "delete_training_job", {"name": "j", "namespace": "team-a"}, raise_on_error=False
            )
        assert result.is_error
        body = result.structured_content
        assert body["error_code"] == "PERMISSION_DENIED"
        assert body["details"]["missing_permissions"] == [
            "delete trainjobs.trainer.kubeflow.org in namespace 'team-a'"
        ]
        assert calls == []

    async def test_other_namespace_is_denied(self):
        calls = []
        mcp = _server(FakeAuthorizationApi(ALLOW_ALICE), {"sub": "alice"}, calls)
        async with Client(mcp) as client:
            result = await client.call_tool(
                "delete_training_job", {"name": "j", "namespace": "team-b"}, raise_on_error=False
            )
        assert result.is_error and calls == []

    async def test_execute_tool_is_checked_as_the_tool_it_runs(self):
        calls = []
        mcp = _server(FakeAuthorizationApi(ALLOW_ALICE), {"sub": "bob"}, calls)
        async with Client(mcp) as client:
            result = await client.call_tool(
                "execute_tool",
                {
                    "tool_name": "delete_training_job",
                    "arguments": {"name": "j", "namespace": "team-a"},
                },
                raise_on_error=False,
            )
        assert result.is_error and calls == []

    async def test_tool_without_permissions_passes_through(self):
        calls = []
        api = FakeAuthorizationApi()
        mcp = _server(api, None, calls)
        async with Client(mcp) as client:
            await client.call_tool("find_tools", {"query": "jobs"})
        assert calls == [("find_tools", "jobs")] and api.reviews == []

    @pytest.mark.parametrize("claims", [None, {"auth_method": "api_key"}])
    async def test_no_verified_user_is_refused(self, claims):
        calls = []
        mcp = _server(FakeAuthorizationApi(ALLOW_ALICE), claims, calls)
        async with Client(mcp) as client:
            result = await client.call_tool(
                "delete_training_job", {"name": "j", "namespace": "team-a"}, raise_on_error=False
            )
        assert result.is_error and calls == []
        assert "no usable verified 'sub' claim" in result.structured_content["error"]

    async def test_review_failure_refuses_the_call(self):
        calls = []
        mcp = _server(FakeAuthorizationApi(error=RuntimeError("boom")), {"sub": "alice"}, calls)
        async with Client(mcp) as client:
            result = await client.call_tool(
                "delete_training_job", {"name": "j", "namespace": "team-a"}, raise_on_error=False
            )
        assert result.is_error and calls == []
        assert result.structured_content["error_code"] == "PERMISSION_DENIED"

    @pytest.mark.parametrize(
        "arguments",
        [
            {"tool_name": ["delete_training_job"], "arguments": {}},
            {"tool_name": "delete_training_job", "arguments": "name=j"},
        ],
    )
    async def test_malformed_execute_tool_is_refused(self, arguments):
        calls = []
        mcp = _server(FakeAuthorizationApi(ALLOW_ALICE), {"sub": "alice"}, calls)
        async with Client(mcp) as client:
            result = await client.call_tool("execute_tool", arguments, raise_on_error=False)
        assert result.is_error and calls == []
        assert result.structured_content["error_code"] == "VALIDATION_ERROR"

    async def test_invalid_namespace_is_a_validation_error_and_sends_no_review(self):
        calls = []
        api = FakeAuthorizationApi(ALLOW_ALICE)
        mcp = _server(api, {"sub": "alice"}, calls)
        async with Client(mcp) as client:
            result = await client.call_tool(
                "delete_training_job", {"name": "j", "namespace": "Team_A"}, raise_on_error=False
            )
        assert result.structured_content["error_code"] == "VALIDATION_ERROR"
        assert calls == [] and api.reviews == []

    async def test_reserved_user_is_refused(self):
        calls = []
        api = FakeAuthorizationApi({("system:admin", "delete", "trainjobs", "team-a")})
        mcp = _server(api, {"sub": "system:admin"}, calls)
        async with Client(mcp) as client:
            result = await client.call_tool(
                "delete_training_job", {"name": "j", "namespace": "team-a"}, raise_on_error=False
            )
        assert result.is_error and calls == [] and api.reviews == []

    async def test_bad_namespace_type_refuses_the_call(self):
        calls = []
        mcp = _server(FakeAuthorizationApi(ALLOW_ALICE), {"sub": "alice"}, calls)
        async with Client(mcp) as client:
            result = await client.call_tool(
                "execute_tool",
                {"tool_name": "delete_training_job", "arguments": {"name": "j", "namespace": 3}},
                raise_on_error=False,
            )
        assert result.is_error and calls == []


def test_builder_maps_each_registered_tool_to_its_namespace_resolver():
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    import kubeflow_mcp.trainer as trainer
    from kubeflow_mcp.common import utils
    from kubeflow_mcp.core.authz import build_authz_middleware

    middleware = build_authz_middleware(
        AuthzConfig(mode="sar"),
        {"trainer": trainer},
        JWTVerifier(jwks_uri="https://issuer.invalid/jwks.json"),
        {"inspect_controller", "list_training_jobs"},
    )
    assert set(middleware._permissions) == {"inspect_controller", "list_training_jobs"}
    assert middleware._resolvers["inspect_controller"] is _controller_namespace
    assert middleware._resolvers["list_training_jobs"] is utils.get_trainer_effective_namespace
