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

"""Workflow integration tests — validate tool chaining across the MCP lifecycle.

Each test simulates a realistic multi-tool workflow the way an MCP agent would
use the server: preview → inspect → submit → monitor → diagnose.

All SDK and K8s calls are mocked. These tests verify that the **data contracts**
between tools are consistent (e.g. job_name from submit feeds into get_logs).
"""

from unittest.mock import MagicMock, patch

from kubeflow_mcp.trainer.api.monitoring import get_training_logs, wait_for_training
from kubeflow_mcp.trainer.api.planning import estimate_resources, get_cluster_resources
from kubeflow_mcp.trainer.api.training import fine_tune, run_custom_training


class TestFineTuneWorkflow:
    """Full fine-tune workflow: estimate → preview → submit → wait → logs."""

    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    @patch("kubeflow_mcp.trainer.api.planning._get_model_info_from_hf")
    @patch("kubeflow_mcp.common.utils.get_core_v1_api")
    def test_full_fine_tune_lifecycle(self, mock_v1, mock_hf, mock_train_client):
        mock_v1_api = MagicMock()
        node = MagicMock()
        node.metadata.name = "gpu-node"
        node.status.allocatable = {"nvidia.com/gpu": "4", "memory": "64Gi", "cpu": "16"}
        mock_v1_api.list_node.return_value.items = [node]
        mock_v1.return_value = mock_v1_api

        mock_hf.return_value = {
            "model_id": "meta-llama/Llama-3.2-1B",
            "params": 1_000_000_000,
            "library": "transformers",
        }

        # Step 1: Check cluster resources
        cluster = get_cluster_resources()
        assert cluster["success"] is True
        assert cluster["data"]["gpu_total"] == 4

        # Step 2: Estimate requirements
        est = estimate_resources(model="meta-llama/Llama-3.2-1B")
        assert est["success"] is True
        assert est["data"]["params_billions"] == 1.0
        assert "breakdown" in est["data"]

        # Step 3: Preview the fine-tune config
        preview = fine_tune(
            model="hf://meta-llama/Llama-3.2-1B",
            dataset="hf://tatsu-lab/alpaca",
            confirmed=False,
        )
        assert preview["status"] == "preview"
        assert preview["config"]["model"] == "hf://meta-llama/Llama-3.2-1B"

        # Step 4: Submit
        mock_client = MagicMock()
        mock_client.train.return_value = "train-llama-abc"
        mock_train_client.return_value = mock_client

        result = fine_tune(
            model="hf://meta-llama/Llama-3.2-1B",
            dataset="hf://tatsu-lab/alpaca",
            confirmed=True,
        )
        assert result["success"] is True
        job_name = result["data"]["job_name"]
        assert job_name == "train-llama-abc"

        # Step 5: Wait for completion — pass the actual job_name from step 4
        mock_client.wait_for_job_status.return_value = MagicMock(status="Complete")
        with patch(
            "kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace",
            return_value=mock_client,
        ):
            wait_result = wait_for_training(name=job_name)
        assert wait_result["success"] is True
        assert wait_result["data"]["status"] == "Complete"

        # Step 6: Get logs using the same job_name
        mock_client.get_job_logs.return_value = iter(["Epoch 1/1: loss=2.34", "Training complete"])
        with patch(
            "kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace",
            return_value=mock_client,
        ):
            logs = get_training_logs(name=job_name)
        assert logs["success"] is True
        assert logs["data"]["job"] == job_name
        assert "failure_hint" not in logs["data"]


class TestCustomTrainingWithFailureHints:
    """Custom training workflow: preview (with warnings) → submit → logs with failure hint."""

    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    @patch.dict("os.environ", {"KUBEFLOW_MCP_UNSAFE_SCRIPTS": "true"})
    def test_unsafe_script_preview_then_submit_then_oom(self, mock_train_client):
        script = "import os\nos.system('echo hi')\nprint(os.environ.get('LR', '1e-4'))"

        # Step 1: Preview — gets safety warnings (not blocking)
        preview = run_custom_training(script=script, confirmed=False)
        assert preview["status"] == "preview"
        assert len(preview["config"]["safety_warnings"]) > 0

        # Step 2: Agent decides to submit anyway (unsafe override enabled)
        mock_client = MagicMock()
        mock_client.train.return_value = "custom-oom-job"
        mock_train_client.return_value = mock_client

        result = run_custom_training(
            script=script,
            env={"LR": "1e-4"},
            confirmed=True,
        )
        assert result["success"] is True
        job_name = result["data"]["job_name"]

        # Step 3: Job fails with OOM — logs contain failure hint
        mock_client.get_job_logs.return_value = iter(
            [
                "Starting training with lr=1e-4",
                "RuntimeError: CUDA out of memory. Tried to allocate 4.00 GiB",
            ]
        )
        with patch(
            "kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace",
            return_value=mock_client,
        ):
            logs = get_training_logs(name=job_name)

        assert logs["success"] is True
        assert logs["data"]["job"] == job_name
        assert "failure_hint" in logs["data"]
        assert logs["data"]["failure_hint"]["category"] == "OOM"
        assert "batch_size" in logs["data"]["failure_hint"]["suggestion"].lower()


class TestIterativeEnvTuning:
    """Env-based iteration: same script, different env → different jobs."""

    @patch("kubeflow_mcp.trainer.api.training.get_trainer_client")
    def test_two_iterations_with_env(self, mock_train_client):
        script = "import os; print(f'LR={os.environ[\"LR\"]}')"

        mock_client = MagicMock()
        mock_train_client.return_value = mock_client

        # Iteration 1
        mock_client.train.return_value = "iter-1"
        r1 = run_custom_training(script=script, env={"LR": "1e-4"}, confirmed=True)
        assert r1["success"] is True

        # Iteration 2 — only env changes
        mock_client.train.return_value = "iter-2"
        r2 = run_custom_training(script=script, env={"LR": "5e-5"}, confirmed=True)
        assert r2["success"] is True
        assert r2["data"]["job_name"] == "iter-2"

        # Verify both calls had different env
        calls = mock_client.train.call_args_list
        assert calls[0].kwargs["trainer"].env == {"LR": "1e-4"}
        assert calls[1].kwargs["trainer"].env == {"LR": "5e-5"}

        # Logs for iteration 2 — missing module error
        mock_client.get_job_logs.return_value = iter(
            [
                "ModuleNotFoundError: No module named 'bitsandbytes'",
            ]
        )
        with patch(
            "kubeflow_mcp.trainer.api.monitoring.get_trainer_client_for_namespace",
            return_value=mock_client,
        ):
            logs = get_training_logs(name="iter-2")

        assert logs["data"]["failure_hint"]["category"] == "MISSING_MODULE"
