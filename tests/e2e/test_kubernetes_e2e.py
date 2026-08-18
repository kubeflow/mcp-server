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
import os
from collections.abc import AsyncGenerator

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Mark all tests in this file as requiring KUBEFLOW_MCP_E2E=true
pytestmark = pytest.mark.skipif(
    os.getenv("KUBEFLOW_MCP_E2E") != "true",
    reason="KUBEFLOW_MCP_E2E=true environment variable not set",
)


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


@pytest.mark.asyncio
async def test_kubernetes_e2e_flow(mcp_session: ClientSession) -> None:
    """Validate full flow of read-only and mutating trainer MCP tools against a live cluster."""
    # 1. pre_flight() detects CRDs, K8s version, platform
    resp = await mcp_session.call_tool("pre_flight", arguments={})
    assert not resp.isError
    assert len(resp.content) == 1
    data = json.loads(resp.content[0].text)
    assert data["success"] is True
    compat = data["data"]["compatibility"]
    assert compat["compatible"] is True
    assert compat["checks"]["trainer_crd"]["status"] == "pass"
    assert data["data"]["cluster"]["node_count"] > 0
    # In Kind/local K8s, it should detect platform "kubernetes"
    assert compat["platform"] == "kubernetes"

    # 2. check_compatibility() validates Trainer CRD presence
    resp = await mcp_session.call_tool("check_compatibility", arguments={})
    assert not resp.isError
    assert len(resp.content) == 1
    data = json.loads(resp.content[0].text)
    assert data["success"] is True
    assert data["data"]["compatible"] is True
    assert data["data"]["checks"]["trainer_crd"]["status"] == "pass"

    # 3. get_cluster_resources() returns real node/GPU info
    resp = await mcp_session.call_tool("get_cluster_resources", arguments={})
    assert not resp.isError
    assert len(resp.content) == 1
    data = json.loads(resp.content[0].text)
    assert data["success"] is True
    cluster_res = data["data"]
    assert cluster_res["node_count"] > 0
    assert cluster_res["gpu_total"] >= 0
    assert cluster_res["nodes_with_gpu"] >= 0

    # 4. list_runtimes() discovers installed ClusterTrainingRuntimes
    resp = await mcp_session.call_tool("list_runtimes", arguments={})
    assert not resp.isError
    assert len(resp.content) == 1
    data = json.loads(resp.content[0].text)
    assert data["success"] is True
    runtimes = data["data"]["runtimes"]
    assert len(runtimes) > 0

    # Pick runtime for fine_tune preview (requires torchtune)
    torchtune_runtime_name = None
    for r in runtimes:
        name = r.get("name", "")
        if "torchtune" in name or "torch-tune" in name:
            torchtune_runtime_name = name
            break
    if not torchtune_runtime_name:
        pytest.skip("No torchtune runtime found in the cluster")

    # Pick runtime for run_custom_training (prefer torch-distributed or torch-*)
    custom_runtime_name = None
    for r in runtimes:
        name = r.get("name", "")
        if name.startswith("torch-distributed"):
            custom_runtime_name = name
            break
    if not custom_runtime_name:
        for r in runtimes:
            name = r.get("name", "")
            if name.startswith("torch-"):
                custom_runtime_name = name
                break
    if not custom_runtime_name:
        pytest.skip("No custom torch- training runtime found in the cluster")

    # 5. list_training_jobs() verify test job names are not present
    namespace = "default"
    job_name = "e2e-fine-tune-job"
    custom_job_name = "e2e-custom-train-job"

    resp = await mcp_session.call_tool("list_training_jobs", arguments={"namespace": namespace})
    assert not resp.isError
    assert len(resp.content) == 1
    data = json.loads(resp.content[0].text)
    assert data["success"] is True
    existing_jobs = [j["name"] for j in data["data"]["jobs"]]
    assert job_name not in existing_jobs
    assert custom_job_name not in existing_jobs

    # 6. fine_tune(confirmed=False) returns preview (or GPU validation error on CPU-only cluster)
    resp = await mcp_session.call_tool(
        "fine_tune",
        arguments={
            "model": "hf://google/gemma-2b",
            "dataset": "hf://tatsu-lab/alpaca",
            "runtime": torchtune_runtime_name,
            "name": job_name,
            "namespace": namespace,
            "confirmed": False,
        },
    )
    assert not resp.isError
    assert len(resp.content) == 1
    data = json.loads(resp.content[0].text)
    if cluster_res.get("gpu_total", 0) == 0:
        assert data["success"] is False
        assert data["error_code"] == "VALIDATION_ERROR"
        assert "requires GPUs" in data["error"]
    else:
        assert data["success"] is True
        assert data["status"] == "preview"
        assert data["config"]["name"] == job_name

    # Verify no TrainJob was created
    resp = await mcp_session.call_tool("list_training_jobs", arguments={"namespace": namespace})
    assert not resp.isError
    data = json.loads(resp.content[0].text)
    job_names = [job["name"] for job in data["data"]["jobs"]]
    assert job_name not in job_names

    # 7. run_custom_training(confirmed=True) asserts TrainJob CR exists in cluster (CPU mode)
    noop_script = (
        "import torch\n"
        "def train():\n"
        "    print(f'PyTorch version: {torch.__version__}')\n"
        "if __name__ == '__main__':\n"
        "    train()\n"
    )
    resp = await mcp_session.call_tool(
        "run_custom_training",
        arguments={
            "script": noop_script,
            "runtime": custom_runtime_name,
            "name": custom_job_name,
            "namespace": namespace,
            "gpu_per_node": 0,
            "confirmed": True,
        },
    )
    assert not resp.isError
    assert len(resp.content) == 1
    data = json.loads(resp.content[0].text)
    assert data["success"] is True
    assert data["data"]["job_name"] == custom_job_name
    assert data["data"]["status"] == "Created"

    # 8. get_training_job() reads back the created job
    resp = await mcp_session.call_tool(
        "get_training_job", arguments={"name": custom_job_name, "namespace": namespace}
    )
    assert not resp.isError
    assert len(resp.content) == 1
    data = json.loads(resp.content[0].text)
    assert data["success"] is True
    assert data["data"]["name"] == custom_job_name
    assert data["data"]["status"] == "Created"

    # 9. delete_training_job(confirmed=True) removes it, verify gone
    resp = await mcp_session.call_tool(
        "delete_training_job",
        arguments={"name": custom_job_name, "namespace": namespace, "confirmed": True},
    )
    assert not resp.isError
    assert len(resp.content) == 1
    data = json.loads(resp.content[0].text)
    assert data["success"] is True
    assert data["data"]["deleted"] is True

    # Verify job is gone
    resp = await mcp_session.call_tool(
        "get_training_job", arguments={"name": custom_job_name, "namespace": namespace}
    )
    assert not resp.isError
    data = json.loads(resp.content[0].text)
    assert data["success"] is False
    assert data["error_code"] == "RESOURCE_NOT_FOUND"
