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

"""SDK Contract Tests for the Optimizer module.

Validates that the OptimizerClient SDK methods and types our MCP tools
depend on exist with expected signatures, without requiring a cluster.

Mirrors the pattern from trainer/api/sdk_contracts_test.py.
"""

import inspect

import pytest
from kubeflow.optimizer import OptimizerClient


class TestOptimizerClientMethods:
    """Verify OptimizerClient has the methods we depend on."""

    @pytest.mark.parametrize(
        "method_name",
        [
            "optimize",
            "list_jobs",
            "get_job",
            "delete_job",
            "get_best_results",
            "get_job_logs",
            "get_job_events",
            "wait_for_job_status",
        ],
    )
    def test_method_exists(self, method_name: str):
        assert hasattr(OptimizerClient, method_name), (
            f"OptimizerClient.{method_name}() not found — SDK may have changed"
        )
        assert callable(getattr(OptimizerClient, method_name))

    def test_optimize_signature(self):
        """optimize() must accept our expected parameters."""
        sig = inspect.signature(OptimizerClient.optimize)
        params = set(sig.parameters.keys()) - {"self"}
        expected = {"trial_template", "trial_config", "search_space", "objectives", "algorithm"}
        assert expected <= params, f"Missing params in optimize(): {expected - params}"

    def test_get_job_takes_name(self):
        sig = inspect.signature(OptimizerClient.get_job)
        assert "name" in sig.parameters

    def test_list_jobs_no_required_args(self):
        sig = inspect.signature(OptimizerClient.list_jobs)
        required = {
            k
            for k, v in sig.parameters.items()
            if v.default is inspect.Parameter.empty and k != "self"
        }
        assert not required, f"list_jobs() has unexpected required params: {required}"

    def test_delete_job_takes_name(self):
        sig = inspect.signature(OptimizerClient.delete_job)
        assert "name" in sig.parameters

    def test_get_best_results_takes_name(self):
        sig = inspect.signature(OptimizerClient.get_best_results)
        assert "name" in sig.parameters

    def test_get_job_logs_signature(self):
        sig = inspect.signature(OptimizerClient.get_job_logs)
        params = set(sig.parameters.keys()) - {"self"}
        assert "name" in params
        assert "trial_name" in params
        assert "follow" in params

    def test_get_job_events_takes_name(self):
        sig = inspect.signature(OptimizerClient.get_job_events)
        assert "name" in sig.parameters

    def test_wait_for_job_status_signature(self):
        sig = inspect.signature(OptimizerClient.wait_for_job_status)
        params = set(sig.parameters.keys()) - {"self"}
        assert "name" in params
        assert "timeout" in params
        assert "polling_interval" in params

    def test_optimize_returns_str(self):
        """optimize() should return str (generated experiment name)."""
        hints = inspect.get_annotations(OptimizerClient.optimize)
        assert hints.get("return") is str


class TestOptimizerTypeImports:
    """Verify SDK types used by MCP tools are importable."""

    def test_optimizer_client_importable(self):
        from kubeflow.optimizer import OptimizerClient  # noqa: F811

        assert OptimizerClient is not None

    def test_optimization_job_importable(self):
        from kubeflow.optimizer import OptimizationJob

        assert OptimizationJob is not None

    def test_result_importable(self):
        from kubeflow.optimizer import Result

        assert Result is not None

    def test_trial_config_importable(self):
        from kubeflow.optimizer import TrialConfig

        assert TrialConfig is not None

    def test_objective_importable(self):
        from kubeflow.optimizer import Objective

        assert Objective is not None

    def test_search_importable(self):
        from kubeflow.optimizer import Search

        assert hasattr(Search, "uniform")
        assert hasattr(Search, "loguniform")
        assert hasattr(Search, "choice")

    def test_random_search_importable(self):
        from kubeflow.optimizer import RandomSearch

        assert RandomSearch is not None

    def test_grid_search_importable(self):
        from kubeflow.optimizer import GridSearch

        assert GridSearch is not None

    def test_train_job_template_importable(self):
        from kubeflow.optimizer import TrainJobTemplate

        assert TrainJobTemplate is not None

    def test_kubernetes_backend_config_importable(self):
        from kubeflow.optimizer import KubernetesBackendConfig

        assert KubernetesBackendConfig is not None


class TestOptimizerTypeFields:
    """Verify SDK dataclass fields match our usage."""

    def test_optimization_job_fields(self):
        from kubeflow.optimizer import OptimizationJob

        field_names = set(OptimizationJob.__dataclass_fields__.keys())
        expected = {"name", "status", "trials", "search_space", "objectives", "algorithm"}
        assert expected <= field_names, f"Missing fields: {expected - field_names}"

    def test_trial_config_fields(self):
        from kubeflow.optimizer import TrialConfig

        field_names = set(TrialConfig.__dataclass_fields__.keys())
        assert "num_trials" in field_names
        assert "parallel_trials" in field_names

    def test_objective_fields(self):
        from kubeflow.optimizer import Objective

        field_names = set(Objective.__dataclass_fields__.keys())
        assert "metric" in field_names
        assert "direction" in field_names

    def test_result_fields(self):
        from kubeflow.optimizer import Result

        field_names = set(Result.__dataclass_fields__.keys())
        assert "parameters" in field_names
        assert "metrics" in field_names


class TestKatibApiModels:
    """Verify kubeflow_katib_api models used for raw spec creation."""

    def test_v1beta1_experiment_importable(self):
        from kubeflow_katib_api.models import V1beta1Experiment

        assert V1beta1Experiment is not None

    def test_v1beta1_experiment_spec_importable(self):
        from kubeflow_katib_api.models import V1beta1ExperimentSpec

        assert V1beta1ExperimentSpec is not None

    def test_v1beta1_trial_template_importable(self):
        from kubeflow_katib_api.models import V1beta1TrialTemplate

        assert V1beta1TrialTemplate is not None

    def test_v1beta1_algorithm_spec_importable(self):
        from kubeflow_katib_api.models import V1beta1AlgorithmSpec

        assert V1beta1AlgorithmSpec is not None

    def test_v1beta1_objective_spec_importable(self):
        from kubeflow_katib_api.models import V1beta1ObjectiveSpec

        assert V1beta1ObjectiveSpec is not None

    def test_v1beta1_early_stopping_spec_importable(self):
        from kubeflow_katib_api.models import V1beta1EarlyStoppingSpec

        assert V1beta1EarlyStoppingSpec is not None


class TestOptimizerConstants:
    """Verify SDK constants used by our code."""

    def test_constants_importable(self):
        from kubeflow.optimizer.constants import constants

        assert constants is not None

    def test_api_group(self):
        from kubeflow.optimizer.constants import constants

        assert constants.GROUP == "kubeflow.org"

    def test_api_version(self):
        from kubeflow.optimizer.constants import constants

        assert constants.VERSION == "v1beta1"

    def test_experiment_plural(self):
        from kubeflow.optimizer.constants import constants

        assert constants.EXPERIMENT_PLURAL == "experiments"

    def test_status_values(self):
        from kubeflow.optimizer.constants import constants

        assert constants.OPTIMIZATION_JOB_COMPLETE == "Complete"
        assert constants.OPTIMIZATION_JOB_FAILED == "Failed"
        assert constants.OPTIMIZATION_JOB_RUNNING == "Running"
        assert constants.OPTIMIZATION_JOB_CREATED == "Created"
