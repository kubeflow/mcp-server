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

"""Every served tool declares the Kubernetes permissions its requests need (#565).

The declarations are what a per-caller authorization check will ask about, so a
tool without one, or one that hides a write, must fail here rather than in a cluster.
"""

import importlib
import inspect
import re

import pytest

from kubeflow_mcp.common.types import ToolPermission
from kubeflow_mcp.core.health import (
    HEALTH_TOOL_ANNOTATIONS,
    HEALTH_TOOL_PERMISSIONS,
    HEALTH_TOOLS,
)
from kubeflow_mcp.core.server import CLIENT_MODULES

_VERBS = {"get", "list", "watch", "create", "update", "patch", "delete", "deletecollection"}
_WRITE_VERBS = {"create", "update", "patch", "delete", "deletecollection"}
_NAMED_VERBS = {"get", "update", "patch", "delete"}
_RESOURCE = re.compile(r"^[a-z][a-z0-9]*(/[a-z]+)?$")
# "" (core), single-label groups such as "apps" and "batch", and dotted ones.
_GROUP = re.compile(r"^([a-z0-9-]+(\.[a-z0-9-]+)*)?$")

# Read-only tools whose requests still write, and why. Keep this list short.
_READ_ONLY_TOOLS_THAT_WRITE = {
    "get_runtime": "include_packages=True runs a short-lived Pod to list packages (#535)",
}
# Clients whose writes go to another API server that authorizes them itself.
_WRITES_AUTHORIZED_ELSEWHERE = {"pipelines"}


def _clients():
    for name, module_path in CLIENT_MODULES.items():
        module = importlib.import_module(module_path)
        if getattr(module, "TOOLS", []):
            yield name, module


def _all_tools():
    for name, module in _clients():
        permissions = getattr(module, "CLIENT_TOOL_PERMISSIONS", {})
        for tool in module.TOOLS:
            yield (
                name,
                tool,
                permissions.get(tool.__name__),
                module.CLIENT_TOOL_ANNOTATIONS.get(tool.__name__, {}),
            )
    for tool in HEALTH_TOOLS:
        yield (
            "health",
            tool,
            HEALTH_TOOL_PERMISSIONS.get(tool.__name__),
            HEALTH_TOOL_ANNOTATIONS.get(tool.__name__, {}),
        )


@pytest.mark.parametrize("client,module", list(_clients()), ids=lambda v: getattr(v, "__name__", v))
def test_every_tool_declares_permissions(client, module):
    tools = {tool.__name__ for tool in module.TOOLS}
    declared = set(getattr(module, "CLIENT_TOOL_PERMISSIONS", {}))
    assert tools - declared == set(), (
        f"{client}: add these tools to CLIENT_TOOL_PERMISSIONS (use () if they make no "
        f"Kubernetes requests): {sorted(tools - declared)}"
    )
    assert declared - tools == set(), f"{client}: entries for unknown tools {declared - tools}"


def test_every_health_tool_declares_permissions():
    assert set(HEALTH_TOOL_PERMISSIONS) == {tool.__name__ for tool in HEALTH_TOOLS}


def _tool_id(value):
    return value.__name__ if callable(value) else value if isinstance(value, str) else ""


@pytest.mark.parametrize("client,tool,permissions,annotations", list(_all_tools()), ids=_tool_id)
def test_permissions_are_well_formed(client, tool, permissions, annotations):
    name = tool.__name__
    for permission in permissions:
        assert isinstance(permission, ToolPermission), f"{name}: {permission!r}"
        assert permission.verb in _VERBS, f"{name}: unknown verb {permission.verb!r}"
        assert _GROUP.match(permission.group), f"{name}: bad group {permission.group!r}"
        assert _RESOURCE.match(permission.resource), f"{name}: bad resource {permission.resource!r}"
        # RBAC matches resourceNames only for requests on one named object.
        if permission.name is not None:
            assert permission.verb in _NAMED_VERBS, f"{name}: {permission.verb} cannot be named"
    assert len(set(permissions)) == len(permissions), f"{name}: duplicate entries"


@pytest.mark.parametrize("client,tool,permissions,annotations", list(_all_tools()), ids=_tool_id)
def test_conditional_permissions_name_a_parameter(client, tool, permissions, annotations):
    """``when``, ``unless`` and ``name`` must name a real parameter, or a check is wrong forever."""
    parameters = set(inspect.signature(tool).parameters)
    for permission in permissions:
        assert not (permission.when and permission.unless), f"{tool.__name__}: {permission}"
        for field in ("when", "unless", "name"):
            value = getattr(permission, field)
            if value is not None:
                assert value in parameters, f"{tool.__name__}: {field}={value!r} is not a parameter"


@pytest.mark.parametrize("client,tool,permissions,annotations", list(_all_tools()), ids=_tool_id)
def test_write_verbs_match_the_read_only_hint(client, tool, permissions, annotations):
    name = tool.__name__
    # Without annotations the tool would silently count as a write tool here.
    assert "readOnlyHint" in annotations, f"{name}: no readOnlyHint annotation to check against"
    writes = {p.verb for p in permissions} & _WRITE_VERBS
    if annotations.get("readOnlyHint"):
        if name in _READ_ONLY_TOOLS_THAT_WRITE:
            assert writes, f"{name} no longer writes; drop it from _READ_ONLY_TOOLS_THAT_WRITE"
        else:
            assert not writes, f"read-only tool {name} declares writes: {sorted(writes)}"
    elif client not in _WRITES_AUTHORIZED_ELSEWHERE:
        assert writes, f"write tool {name} declares no Kubernetes write"


def test_read_only_writes_are_conditional():
    """A read-only tool may only write for an opt-in argument, never on every call."""
    for _client, tool, permissions, annotations in _all_tools():
        if not annotations.get("readOnlyHint"):
            continue
        for permission in permissions:
            if permission.verb in _WRITE_VERBS:
                assert permission.when, f"{tool.__name__} writes {permission} unless opted in"


@pytest.mark.parametrize("group", ["", "apps", "batch", "trainer.kubeflow.org"])
def test_group_pattern_accepts_real_api_groups(group):
    assert _GROUP.match(group)


@pytest.mark.parametrize("group", ["Apps", "trainer..kubeflow.org", ".io", "a b"])
def test_group_pattern_rejects_malformed_groups(group):
    assert not _GROUP.match(group)


def test_inspect_crd_declares_each_branch():
    """Without a name inspect_crd lists CRDs; with one it reads only that CRD."""
    import kubeflow_mcp.trainer as trainer

    crd = ("apiextensions.k8s.io", "customresourcedefinitions")
    by_verb = {
        p.verb: p
        for p in trainer.CLIENT_TOOL_PERMISSIONS["inspect_crd"]
        if (p.group, p.resource) == crd
    }
    assert (by_verb["list"].when, by_verb["list"].unless) == (None, "name")
    assert (by_verb["get"].when, by_verb["get"].unless) == ("name", None)


@pytest.mark.parametrize(
    "client,tool_name",
    [
        ("trainer", "delete_training_job"),
        ("trainer", "update_training_job"),
        ("trainer", "patch_runtime"),
        ("trainer", "delete_runtime"),
        ("spark", "delete_spark_session"),
    ],
)
def test_writes_on_one_object_name_it(client, tool_name):
    """A Role limited to one object by resourceNames only matches a request that names it."""
    permissions = importlib.import_module(f"kubeflow_mcp.{client}").CLIENT_TOOL_PERMISSIONS
    writes = [p for p in permissions[tool_name] if p.verb in _WRITE_VERBS]
    assert writes and all(p.name == "name" for p in writes), writes
