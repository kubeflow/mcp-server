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

"""Spark client module — MCP tools for SparkConnect sessions on Kubernetes.

Wraps ``kubeflow.spark.SparkClient`` (KEP-107) so AI agents can create, inspect,
monitor, and tear down SparkConnect sessions. Structure mirrors ``trainer/``:

├── api/       # Tool implementations (discovery, sessions, monitoring)
├── types/     # SDK-object serialization helpers
└── resources/ # Agent-facing guides (SKILL/how-to markdown)

The ``kubeflow.spark`` SDK and its ``pyspark`` dependency ship behind the
optional ``kubeflow[spark]`` extra; every SDK import in this module is lazy so
the module — and its metadata — load even when the extra is absent.
"""

from kubeflow_mcp.spark.api.discovery import get_spark_session, list_spark_sessions
from kubeflow_mcp.spark.api.monitoring import get_spark_session_logs
from kubeflow_mcp.spark.api.sessions import create_spark_session, delete_spark_session

MODULE_INFO = {
    "name": "spark",
    "description": "SparkConnect session management on Kubernetes",
    "sdk_client": "kubeflow.spark.SparkClient",
    "sdk_version": ">=0.4.0",
    "extra": "spark",
    "status": "implemented",
}

TOOLS = [
    list_spark_sessions,
    get_spark_session,
    get_spark_session_logs,
    create_spark_session,
    delete_spark_session,
]

# ─── Tool metadata (owned by this client module) ───────────────────────────

CLIENT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "list_spark_sessions": "List SparkConnect sessions. Filter by state or namespace.",
    "get_spark_session": "Get details of a SparkConnect session (state, driver pod, service).",
    "get_spark_session_logs": "Get driver-pod logs from a SparkConnect session. Pass tail_lines to bound output.",
    "create_spark_session": (
        "Create a SparkConnect session. Preview first (confirmed=False), then set "
        "confirmed=True to provision. Returns connect info for PySpark to attach."
    ),
    "delete_spark_session": (
        "[DESTRUCTIVE] Delete a SparkConnect session permanently. Set confirmed=True to execute."
    ),
}

CLIENT_TOOL_ANNOTATIONS: dict[str, dict] = {
    "list_spark_sessions": {
        "title": "List SparkConnect Sessions",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "spark", "sessions"],
    },
    "get_spark_session": {
        "title": "Get SparkConnect Session",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["discovery", "spark", "sessions"],
    },
    "get_spark_session_logs": {
        "title": "Get SparkConnect Session Logs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["monitoring", "spark", "logs", "debug"],
    },
    "create_spark_session": {
        "title": "Create SparkConnect Session",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
        "tags": ["spark", "sessions", "create"],
    },
    "delete_spark_session": {
        "title": "Delete SparkConnect Session",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
        "tags": ["spark", "sessions", "lifecycle", "cleanup"],
    },
}

# ─── Resources (owned by this client module) ───────────────────────────────

CLIENT_RESOURCES: dict[str, tuple[str, str]] = {
    "spark://guides/session-patterns": (
        "resources/session-patterns.md",
        "SparkConnect session workflow and PySpark attach patterns.",
    ),
    "spark://guides/troubleshooting": (
        "resources/troubleshooting.md",
        "SparkConnect error-to-fix tables and known limitations.",
    ),
}

# ─── Instruction sections (full tier; compact/minimal auto-derived) ────────
# Spark phases reuse the server's fixed section slots so guidance surfaces for
# the right personas: spark_monitoring/spark_discovery -> "monitoring",
# spark_sessions -> "training".

PHASE_TO_SECTION: dict[str, str | None] = {
    "spark_discovery": None,
    "spark_monitoring": "monitoring",
    "spark_sessions": "training",
}

INSTRUCTION_SECTIONS: dict[str, dict[str, str]] = {
    "monitoring": {
        "full": """\
SPARK SESSIONS (SparkConnect):
- list_spark_sessions() -> discover sessions; filter by state=Ready/Failed/Provisioning
- get_spark_session(name) -> state, driver pod, service name, connect info
- get_spark_session_logs(name) -> driver-pod logs (bounded; pass tail_lines). Streaming is not exposed
- A session with no driver pod is still provisioning — poll get_spark_session(name) until state=Ready
- Read spark://guides/troubleshooting for SparkConnect error-to-fix tables""",
    },
    "training": {
        "full": """\
SPARK SESSION LIFECYCLE:
- create_spark_session(...) -> ALWAYS preview first (confirmed=False), show the config, then confirmed=True
- The MCP server provisions the session and returns connect info; the data plane attaches with PySpark
  (the server does NOT proxy Spark RPCs). Requires the kubeflow[spark] extra on the server host
- Size with num_executors + executor_resources={"cpu": "1", "memory": "2Gi"}; driver_resources for the driver
- delete_spark_session(name, confirmed=True) -> tear down permanently; preview first
- Read spark://guides/session-patterns for end-to-end create -> attach -> delete examples""",
    },
}


__all__ = [
    "MODULE_INFO",
    "TOOLS",
    "CLIENT_TOOL_DESCRIPTIONS",
    "CLIENT_TOOL_ANNOTATIONS",
    "CLIENT_RESOURCES",
    "INSTRUCTION_SECTIONS",
    "PHASE_TO_SECTION",
    *[t.__name__ for t in TOOLS],
]
