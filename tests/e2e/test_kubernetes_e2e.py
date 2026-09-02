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

"""Kubernetes E2E tests validating MCP tools against a real cluster."""

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from kubernetes.client import V1Namespace, V1ObjectMeta
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from kubeflow_mcp.common.utils import get_core_v1_api

logger = logging.getLogger(__name__)

# Mark all tests in this file as requiring KUBEFLOW_MCP_E2E=true
pytestmark = pytest.mark.skipif(
    os.getenv("KUBEFLOW_MCP_E2E") != "true",
    reason="KUBEFLOW_MCP_E2E=true environment variable not set",
)

NOOP_TRAIN_SCRIPT = (
    "import torch\n"
    "def train():\n"
    "    print(f'PyTorch version: {torch.__version__}')\n"
    "if __name__ == '__main__':\n"
    "    train()\n"
)

INVALID_K8S_NAME = "INVALID_UPPERCASE_NAME"
NON_EXISTENT_JOB = "non-existent-e2e-job-999"
NON_EXISTENT_RUNTIME = "non-existent-runtime-999"
NON_EXISTENT_NAMESPACE = "non-existent-ns-999"


@pytest.fixture
async def mcp_session() -> AsyncGenerator[ClientSession, None]:
    """Fixture to start the MCP server and yield an initialized ClientSession."""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "kubeflow-mcp", "serve", "--persona", "platform-admin", "--no-banner"],
        env=os.environ.copy(),
    )

    session_future: asyncio.Future[ClientSession] = asyncio.Future()
    stop_event = asyncio.Event()

    async def run_server() -> None:
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    session_future.set_result(session)
                    await stop_event.wait()
        except Exception as exc:
            if not session_future.done():
                session_future.set_exception(exc)
            else:
                raise

    task = asyncio.create_task(run_server())
    try:
        session = await session_future
        yield session
    finally:
        stop_event.set()
        await asyncio.gather(task, return_exceptions=True)


async def _call_tool(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Helper to invoke an MCP tool and parse its single-content JSON response."""
    resp = await session.call_tool(name, arguments=arguments or {})
    assert not resp.isError, f"Tool '{name}' failed with unexpected MCP protocol error: {resp}"
    assert len(resp.content) == 1, (
        f"Expected exactly 1 content block from '{name}', got {len(resp.content)}"
    )
    return json.loads(resp.content[0].text)


async def _get_custom_runtime_name(session: ClientSession) -> str:
    """Discover installed torch runtime suitable for custom CPU training in Kind."""
    data = await _call_tool(session, "list_runtimes")
    assert data.get("success") is True, f"list_runtimes failed: {data}"
    runtimes = data.get("data", {}).get("runtimes", [])
    for r in runtimes:
        name = r.get("name", "")
        if name.startswith("torch-distributed"):
            return name
    for r in runtimes:
        name = r.get("name", "")
        if name.startswith("torch-"):
            return name
    pytest.skip("No custom torch- training runtime found in the cluster")


async def _get_torchtune_runtime_name(session: ClientSession) -> str:
    """Discover installed torchtune runtime in the cluster."""
    data = await _call_tool(session, "list_runtimes")
    assert data.get("success") is True, f"list_runtimes failed: {data}"
    runtimes = data.get("data", {}).get("runtimes", [])
    for r in runtimes:
        name = r.get("name", "")
        if "torchtune" in name or "torch-tune" in name:
            return name
    pytest.skip("No torchtune runtime found in the cluster")


@pytest.mark.asyncio
async def test_kubernetes_e2e_flow(mcp_session: ClientSession) -> None:
    """Validate full flow of read-only and mutating trainer MCP tools against a live cluster."""
    # 1. pre_flight() detects CRDs, K8s version, platform
    data = await _call_tool(mcp_session, "pre_flight")
    assert data["success"] is True
    compat = data["data"]["compatibility"]
    assert compat["compatible"] is True
    assert compat["checks"]["trainer_crd"]["status"] == "pass"
    assert data["data"]["cluster"]["node_count"] > 0
    assert compat["platform"] == "kubernetes"

    # 2. check_compatibility() validates Trainer CRD presence
    data = await _call_tool(mcp_session, "check_compatibility")
    assert data["success"] is True
    assert data["data"]["compatible"] is True
    assert data["data"]["checks"]["trainer_crd"]["status"] == "pass"

    # 3. get_cluster_resources() returns real node/GPU info
    data = await _call_tool(mcp_session, "get_cluster_resources")
    assert data["success"] is True
    cluster_res = data["data"]
    assert cluster_res["node_count"] > 0
    assert cluster_res["gpu_total"] >= 0
    assert cluster_res["nodes_with_gpu"] >= 0

    # 4. list_runtimes() discovers installed ClusterTrainingRuntimes
    data = await _call_tool(mcp_session, "list_runtimes")
    assert data["success"] is True
    runtimes = data["data"]["runtimes"]
    assert len(runtimes) > 0

    # Pick runtime for fine_tune preview (requires torchtune)
    torchtune_runtime_name = await _get_torchtune_runtime_name(mcp_session)

    # Pick runtime for run_custom_training
    custom_runtime_name = await _get_custom_runtime_name(mcp_session)

    # 5. list_training_jobs() verify test job names are not present
    namespace = "default"
    job_name = f"e2e-ft-{uuid.uuid4().hex[:6]}"
    custom_job_name = f"e2e-custom-{uuid.uuid4().hex[:6]}"

    data = await _call_tool(mcp_session, "list_training_jobs", {"namespace": namespace})
    assert data["success"] is True
    existing_jobs = [j["name"] for j in data["data"]["jobs"]]
    assert job_name not in existing_jobs
    assert custom_job_name not in existing_jobs

    # 6. fine_tune(confirmed=False) returns preview (or GPU validation error on CPU-only cluster)
    data = await _call_tool(
        mcp_session,
        "fine_tune",
        {
            "model": "hf://google/gemma-2b",
            "dataset": "hf://tatsu-lab/alpaca",
            "runtime": torchtune_runtime_name,
            "name": job_name,
            "namespace": namespace,
            "confirmed": False,
        },
    )
    if cluster_res.get("gpu_total", 0) == 0:
        assert data["success"] is False
        assert data["error_code"] == "VALIDATION_ERROR"
        assert "requires GPUs" in data["error"]
    else:
        assert data["success"] is True
        assert data["status"] == "preview"
        assert data["config"]["name"] == job_name

    # Verify no TrainJob was created
    data = await _call_tool(mcp_session, "list_training_jobs", {"namespace": namespace})
    job_names = [job["name"] for job in data["data"]["jobs"]]
    assert job_name not in job_names

    # 7. run_custom_training(confirmed=True) asserts TrainJob CR exists in cluster (CPU mode)
    try:
        data = await _call_tool(
            mcp_session,
            "run_custom_training",
            {
                "script": NOOP_TRAIN_SCRIPT,
                "runtime": custom_runtime_name,
                "name": custom_job_name,
                "namespace": namespace,
                "gpu_per_node": 0,
                "confirmed": True,
            },
        )
        assert data["success"] is True
        assert data["data"]["job_name"] == custom_job_name
        assert data["data"]["status"] == "Created"

        # 8. get_training_job() reads back the created job
        data = await _call_tool(
            mcp_session,
            "get_training_job",
            {"name": custom_job_name, "namespace": namespace},
        )
        assert data["success"] is True
        assert data["data"]["name"] == custom_job_name
        assert data["data"]["status"] == "Created"

        # 9. delete_training_job(confirmed=True) removes it, verify gone
        data = await _call_tool(
            mcp_session,
            "delete_training_job",
            {"name": custom_job_name, "namespace": namespace, "confirmed": True},
        )
        assert data["success"] is True
        assert data["data"]["deleted"] is True

        # Verify job is gone
        data = await _call_tool(
            mcp_session,
            "get_training_job",
            {"name": custom_job_name, "namespace": namespace},
        )
        assert data["success"] is False
        assert data["error_code"] == "RESOURCE_NOT_FOUND"
    finally:
        try:
            await mcp_session.call_tool(
                "delete_training_job",
                arguments={"name": custom_job_name, "namespace": namespace, "confirmed": True},
            )
        except Exception:
            logger.warning("cleanup failed for %s", custom_job_name, exc_info=True)


@pytest.mark.asyncio
async def test_e2e_negative_paths(mcp_session: ClientSession) -> None:
    """Validate error boundaries and input validation against live server."""
    # 1. Invalid K8s names rejected before K8s API call
    resp = await _call_tool(
        mcp_session,
        "get_training_job",
        {"name": INVALID_K8S_NAME, "namespace": "default"},
    )
    assert resp.get("success") is False
    assert resp.get("error_code") == "VALIDATION_ERROR"

    resp = await _call_tool(
        mcp_session,
        "run_custom_training",
        {"script": NOOP_TRAIN_SCRIPT, "name": INVALID_K8S_NAME},
    )
    assert resp.get("success") is False
    assert resp.get("error_code") == "VALIDATION_ERROR"

    resp = await _call_tool(
        mcp_session,
        "delete_training_job",
        {"name": INVALID_K8S_NAME, "confirmed": True},
    )
    assert resp.get("success") is False
    assert resp.get("error_code") == "VALIDATION_ERROR"

    # 2. Non-existent job returns RESOURCE_NOT_FOUND
    resp = await _call_tool(
        mcp_session,
        "get_training_job",
        {"name": NON_EXISTENT_JOB, "namespace": "default"},
    )
    assert resp.get("success") is False
    assert resp.get("error_code") == "RESOURCE_NOT_FOUND"

    # 3. Non-existent namespace returns RESOURCE_NOT_FOUND
    resp = await _call_tool(
        mcp_session,
        "get_training_job",
        {"name": "valid-job-name", "namespace": NON_EXISTENT_NAMESPACE},
    )
    assert resp.get("success") is False
    assert resp.get("error_code") == "RESOURCE_NOT_FOUND"

    # 4. Non-existent runtime returns RESOURCE_NOT_FOUND
    resp = await _call_tool(
        mcp_session,
        "get_runtime",
        {"name": NON_EXISTENT_RUNTIME},
    )
    assert resp.get("success") is False
    assert resp.get("error_code") == "RESOURCE_NOT_FOUND"

    # 5. Invalid lifecycle action returns VALIDATION_ERROR
    resp = await _call_tool(
        mcp_session,
        "update_training_job",
        {"name": "valid-job-name", "action": "invalid_action"},
    )
    assert resp.get("success") is False
    assert resp.get("error_code") == "VALIDATION_ERROR"

    # 6. fine_tune on GPU-less cluster returns VALIDATION_ERROR
    torchtune_runtime_name = await _get_torchtune_runtime_name(mcp_session)
    cluster_res = await _call_tool(mcp_session, "get_cluster_resources")
    if cluster_res.get("data", {}).get("gpu_total", 0) == 0:
        resp = await _call_tool(
            mcp_session,
            "fine_tune",
            {
                "model": "hf://google/gemma-2b",
                "dataset": "hf://tatsu-lab/alpaca",
                "runtime": torchtune_runtime_name,
                "name": "e2e-neg-gpu-job",
                "confirmed": False,
            },
        )
        assert resp.get("success") is False
        assert resp.get("error_code") == "VALIDATION_ERROR"
        assert "requires GPUs" in resp.get("error", "")

    # Invalid fine_tune parameter rejects before K8s call
    resp = await _call_tool(
        mcp_session,
        "fine_tune",
        {
            "model": "hf://google/gemma-2b",
            "dataset": "hf://tatsu-lab/alpaca",
            "runtime": torchtune_runtime_name,
            "name": "e2e-neg-dtype-job",
            "dtype": "invalid_dtype",
            "confirmed": False,
        },
    )
    assert resp.get("success") is False
    assert resp.get("error_code") == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_e2e_confirm_gate(mcp_session: ClientSession) -> None:
    """Validate confirm gate blocks mutations and returns preview without altering cluster state."""
    custom_runtime_name = await _get_custom_runtime_name(mcp_session)
    job_name = f"e2e-gate-{uuid.uuid4().hex[:6]}"
    namespace = "default"

    try:
        # 1. run_custom_training(confirmed=False) returns preview and creates no CR
        preview_resp = await _call_tool(
            mcp_session,
            "run_custom_training",
            {
                "script": NOOP_TRAIN_SCRIPT,
                "runtime": custom_runtime_name,
                "name": job_name,
                "namespace": namespace,
                "gpu_per_node": 0,
                "confirmed": False,
            },
        )
        assert preview_resp.get("success") is True
        assert preview_resp.get("status") == "preview"
        assert "confirmed=True" in preview_resp.get("message", "")

        # Verify no TrainJob was created
        check_resp = await _call_tool(
            mcp_session,
            "get_training_job",
            {"name": job_name, "namespace": namespace},
        )
        assert check_resp.get("success") is False
        assert check_resp.get("error_code") == "RESOURCE_NOT_FOUND"

        # 2. Create the job with confirmed=True
        create_resp = await _call_tool(
            mcp_session,
            "run_custom_training",
            {
                "script": NOOP_TRAIN_SCRIPT,
                "runtime": custom_runtime_name,
                "name": job_name,
                "namespace": namespace,
                "gpu_per_node": 0,
                "confirmed": True,
            },
        )
        assert create_resp.get("success") is True

        # 3. delete_training_job(confirmed=False) returns preview without deleting
        del_prev = await _call_tool(
            mcp_session,
            "delete_training_job",
            {"name": job_name, "namespace": namespace, "confirmed": False},
        )
        assert del_prev.get("success") is True
        assert del_prev.get("status") == "preview"
        assert "confirmed=True" in del_prev.get("message", "")

        # Verify job still exists
        still_there = await _call_tool(
            mcp_session,
            "get_training_job",
            {"name": job_name, "namespace": namespace},
        )
        assert still_there.get("success") is True
        assert still_there.get("data", {}).get("name") == job_name

        # 4. delete_training_job(confirmed=True) executes deletion
        del_exec = await _call_tool(
            mcp_session,
            "delete_training_job",
            {"name": job_name, "namespace": namespace, "confirmed": True},
        )
        assert del_exec.get("success") is True
        assert del_exec.get("data", {}).get("deleted") is True

        # Verify job is gone
        gone_resp = await _call_tool(
            mcp_session,
            "get_training_job",
            {"name": job_name, "namespace": namespace},
        )
        assert gone_resp.get("success") is False
        assert gone_resp.get("error_code") == "RESOURCE_NOT_FOUND"
    finally:
        try:
            await mcp_session.call_tool(
                "delete_training_job",
                arguments={"name": job_name, "namespace": namespace, "confirmed": True},
            )
        except Exception:
            logger.warning("cleanup failed for %s", job_name, exc_info=True)


@pytest.mark.asyncio
async def test_e2e_job_lifecycle(mcp_session: ClientSession) -> None:
    """Validate TrainJob suspend, resume, and delete lifecycle against live cluster."""
    custom_runtime_name = await _get_custom_runtime_name(mcp_session)
    job_name = f"e2e-life-{uuid.uuid4().hex[:6]}"
    namespace = "default"

    try:
        # 1. Create job
        create_resp = await _call_tool(
            mcp_session,
            "run_custom_training",
            {
                "script": NOOP_TRAIN_SCRIPT,
                "runtime": custom_runtime_name,
                "name": job_name,
                "namespace": namespace,
                "gpu_per_node": 0,
                "confirmed": True,
            },
        )
        assert create_resp.get("success") is True
        assert create_resp.get("data", {}).get("status") == "Created"

        # 2. Suspend job
        suspend_resp = await _call_tool(
            mcp_session,
            "update_training_job",
            {"name": job_name, "action": "suspend", "namespace": namespace},
        )
        assert suspend_resp.get("success") is True
        assert suspend_resp.get("data", {}).get("action") == "suspend"
        assert "suspended" in suspend_resp.get("data", {}).get("message", "")

        # Verify job is accessible (suspended jobs show status 'Created' in the API, known controller behavior)
        get_resp = await _call_tool(
            mcp_session,
            "get_training_job",
            {"name": job_name, "namespace": namespace},
        )
        assert get_resp.get("success") is True
        assert get_resp.get("data", {}).get("name") == job_name

        # 3. Resume job
        resume_resp = await _call_tool(
            mcp_session,
            "update_training_job",
            {"name": job_name, "action": "resume", "namespace": namespace},
        )
        assert resume_resp.get("success") is True
        assert resume_resp.get("data", {}).get("action") == "resume"
        assert "resumed" in resume_resp.get("data", {}).get("message", "")

        # 4. Delete job
        del_resp = await _call_tool(
            mcp_session,
            "delete_training_job",
            {"name": job_name, "namespace": namespace, "confirmed": True},
        )
        assert del_resp.get("success") is True
        assert del_resp.get("data", {}).get("deleted") is True

        # Verify job is gone
        gone_resp = await _call_tool(
            mcp_session,
            "get_training_job",
            {"name": job_name, "namespace": namespace},
        )
        assert gone_resp.get("success") is False
        assert gone_resp.get("error_code") == "RESOURCE_NOT_FOUND"
    finally:
        try:
            await mcp_session.call_tool(
                "delete_training_job",
                arguments={"name": job_name, "namespace": namespace, "confirmed": True},
            )
        except Exception:
            logger.warning("cleanup failed for %s", job_name, exc_info=True)


@pytest.mark.asyncio
async def test_e2e_monitoring(mcp_session: ClientSession) -> None:
    """Validate get_training_events and get_training_logs on a live job."""
    custom_runtime_name = await _get_custom_runtime_name(mcp_session)
    job_name = f"e2e-mon-{uuid.uuid4().hex[:6]}"
    namespace = "default"

    try:
        create_resp = await _call_tool(
            mcp_session,
            "run_custom_training",
            {
                "script": NOOP_TRAIN_SCRIPT,
                "runtime": custom_runtime_name,
                "name": job_name,
                "namespace": namespace,
                "gpu_per_node": 0,
                "confirmed": True,
            },
        )
        assert create_resp.get("success") is True

        # 1. get_training_events on live job
        events_resp = await _call_tool(
            mcp_session,
            "get_training_events",
            {"name": job_name, "namespace": namespace},
        )
        assert events_resp.get("success") is True
        data = events_resp.get("data", {})
        assert data.get("job") == job_name
        assert isinstance(data.get("events"), list)
        assert data.get("total") >= 0

        # 2. get_training_logs
        # On slow Kind runners, if the pod hasn't scheduled yet, the SDK may return
        # an error (e.g. SDK_ERROR / pod not found) or empty logs.
        logs_resp = await _call_tool(
            mcp_session,
            "get_training_logs",
            {"name": job_name, "namespace": namespace},
        )
        if logs_resp.get("success"):
            data = logs_resp.get("data", {})
            assert data.get("job") == job_name
            assert "logs" in data
        else:
            assert logs_resp.get("error_code") in ("SDK_ERROR", "RESOURCE_NOT_FOUND")
    finally:
        try:
            await mcp_session.call_tool(
                "delete_training_job",
                arguments={"name": job_name, "namespace": namespace, "confirmed": True},
            )
        except Exception:
            logger.warning("cleanup failed for %s", job_name, exc_info=True)


@pytest.mark.asyncio
async def test_e2e_discovery(mcp_session: ClientSession) -> None:
    """Validate list_runtimes and get_runtime discovery tools."""
    # 1. list_runtimes discovers installed ClusterTrainingRuntimes
    list_resp = await _call_tool(mcp_session, "list_runtimes")
    assert list_resp.get("success") is True
    runtimes = list_resp.get("data", {}).get("runtimes", [])
    assert len(runtimes) > 0

    first_rt_name = runtimes[0]["name"]

    # 2. get_runtime for installed runtime extracts metadata
    get_resp = await _call_tool(mcp_session, "get_runtime", {"name": first_rt_name})
    assert get_resp.get("success") is True
    rt_data = get_resp.get("data", {})
    assert rt_data.get("name") == first_rt_name

    # 3. get_runtime for non-existent runtime
    neg_resp = await _call_tool(mcp_session, "get_runtime", {"name": NON_EXISTENT_RUNTIME})
    assert neg_resp.get("success") is False
    assert neg_resp.get("error_code") == "RESOURCE_NOT_FOUND"


@pytest.mark.asyncio
async def test_e2e_platform_tools(mcp_session: ClientSession) -> None:
    """Validate health_check, inspect_crd, and inspect_controller platform tools."""
    # 1. health_check
    health = await _call_tool(mcp_session, "health_check")
    assert health.get("success") is True
    assert health.get("data", {}).get("kubernetes") is True
    assert health.get("data", {}).get("status") in ("healthy", "degraded")

    # 2. inspect_crd (list all Trainer CRDs)
    crd_list = await _call_tool(mcp_session, "inspect_crd")
    assert crd_list.get("success") is True
    crds = crd_list.get("data", {}).get("crds", [])
    crd_names = [c.get("name") for c in crds]
    assert "trainjobs.trainer.kubeflow.org" in crd_names

    # 3. inspect_crd detail
    crd_detail = await _call_tool(
        mcp_session, "inspect_crd", {"name": "trainjobs.trainer.kubeflow.org"}
    )
    assert crd_detail.get("success") is True
    assert crd_detail.get("data", {}).get("name") == "trainjobs.trainer.kubeflow.org"
    assert crd_detail.get("data", {}).get("group") == "trainer.kubeflow.org"
    assert len(crd_detail.get("data", {}).get("versions", [])) > 0

    # 4. inspect_controller
    controller = await _call_tool(mcp_session, "inspect_controller")
    assert controller.get("success") is True
    assert "pod" in controller.get("data", {})
    assert "logs" in controller.get("data", {})


@pytest.mark.asyncio
async def test_e2e_multi_namespace(mcp_session: ClientSession) -> None:
    """Validate multi-namespace job creation, query, and cleanup."""
    custom_runtime_name = await _get_custom_runtime_name(mcp_session)
    temp_ns = f"mcp-e2e-{uuid.uuid4().hex[:6]}"
    job_name = f"e2e-ns-{uuid.uuid4().hex[:6]}"

    core = get_core_v1_api()
    core.create_namespace(body=V1Namespace(metadata=V1ObjectMeta(name=temp_ns)))

    try:
        # 1. Run job in the temporary namespace
        create_resp = await _call_tool(
            mcp_session,
            "run_custom_training",
            {
                "script": NOOP_TRAIN_SCRIPT,
                "runtime": custom_runtime_name,
                "name": job_name,
                "namespace": temp_ns,
                "gpu_per_node": 0,
                "confirmed": True,
            },
        )
        assert create_resp.get("success") is True
        assert create_resp.get("data", {}).get("job_name") == job_name

        # 2. Get job from temp namespace
        get_resp = await _call_tool(
            mcp_session,
            "get_training_job",
            {"name": job_name, "namespace": temp_ns},
        )
        assert get_resp.get("success") is True
        assert get_resp.get("data", {}).get("name") == job_name

        # Verify job is not found in default namespace
        get_def = await _call_tool(
            mcp_session,
            "get_training_job",
            {"name": job_name, "namespace": "default"},
        )
        assert get_def.get("success") is False
        assert get_def.get("error_code") == "RESOURCE_NOT_FOUND"

        # 3. List jobs in temp namespace
        list_temp = await _call_tool(
            mcp_session,
            "list_training_jobs",
            {"namespace": temp_ns},
        )
        assert list_temp.get("success") is True
        temp_jobs = [j["name"] for j in list_temp.get("data", {}).get("jobs", [])]
        assert job_name in temp_jobs

        # 4. List jobs in default namespace - should not include the temp namespace job
        list_def = await _call_tool(
            mcp_session,
            "list_training_jobs",
            {"namespace": "default"},
        )
        assert list_def.get("success") is True
        def_jobs = [j["name"] for j in list_def.get("data", {}).get("jobs", [])]
        assert job_name not in def_jobs

        # 5. Delete job in temp namespace
        del_resp = await _call_tool(
            mcp_session,
            "delete_training_job",
            {"name": job_name, "namespace": temp_ns, "confirmed": True},
        )
        assert del_resp.get("success") is True
        assert del_resp.get("data", {}).get("deleted") is True
    finally:
        try:
            await mcp_session.call_tool(
                "delete_training_job",
                arguments={"name": job_name, "namespace": temp_ns, "confirmed": True},
            )
        except Exception:
            logger.warning("cleanup failed for %s", job_name, exc_info=True)
        try:
            core.delete_namespace(name=temp_ns, grace_period_seconds=0)
        except Exception:
            logger.warning("cleanup failed for %s", temp_ns, exc_info=True)
