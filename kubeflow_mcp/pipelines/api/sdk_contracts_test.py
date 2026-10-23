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

"""Pin the parts of the Kubeflow SDK and kfp that the pipelines tools rely on.

These run only where the ``kubeflow[pipelines]`` extra is installed; without ``kfp``
the whole module is skipped. A failure here means an SDK upgrade changed something
the tools call, so fix the tools rather than this test.
"""

import inspect

import pytest

pytest.importorskip("kfp")
kfp_server_api = pytest.importorskip("kfp_server_api")

from kfp.client import Client  # noqa: E402
from kubeflow.pipelines import KubernetesBackendConfig, PipelinesClient  # noqa: E402


def _params(method) -> list[str]:
    return list(inspect.signature(method).parameters)


class TestPipelinesClient:
    def test_backend_config_takes_the_fields_we_pass(self):
        config = KubernetesBackendConfig(base_url=None, user_token=None, namespace="ns")
        assert config.namespace == "ns"

    @pytest.mark.parametrize(
        ("method", "params"),
        [
            ("list_pipelines", ["self", "page_token", "page_size"]),
            ("get_pipeline", ["self", "name"]),
            ("get_pipeline_version", ["self", "name", "version"]),
            ("list_pipeline_versions", ["self", "name", "page_token", "page_size"]),
            ("get_experiment", ["self", "name"]),
            ("get_run", ["self", "run_id"]),
            (
                "list_runs",
                ["self", "pipeline", "experiment", "status", "page_token", "page_size"],
            ),
            ("run", ["self", "pipeline", "params", "name", "experiment", "version"]),
            ("kfp_client", ["self"]),
        ],
    )
    def test_method_signatures(self, method, params):
        assert _params(getattr(PipelinesClient, method)) == params

    def test_kfp_client_can_terminate_and_holds_a_run_api(self):
        # stop_pipeline_run uses Client.terminate_run; retry_pipeline_run goes through
        # Client._run_api because PipelinesClient has no retry method.
        assert _params(Client.terminate_run) == ["self", "run_id"]
        assert "self._run_api = " in inspect.getsource(Client.__init__)
        assert "run_id" in _params(kfp_server_api.RunServiceApi.run_service_retry_run)


class TestModels:
    @pytest.mark.parametrize(
        ("model", "fields"),
        [
            (
                "V2beta1Pipeline",
                {"pipeline_id", "display_name", "description", "namespace", "created_at"},
            ),
            ("V2beta1PipelineVersion", {"pipeline_version_id", "display_name", "created_at"}),
            (
                "V2beta1Run",
                {
                    "run_id",
                    "display_name",
                    "state",
                    "experiment_id",
                    "pipeline_version_reference",
                    "runtime_config",
                    "run_details",
                    "error",
                    "created_at",
                    "finished_at",
                },
            ),
            (
                "V2beta1PipelineTaskDetail",
                {"display_name", "state", "pod_name", "child_tasks", "error"},
            ),
            ("V2beta1ListRunsResponse", {"runs", "next_page_token", "total_size"}),
        ],
    )
    def test_fields_the_serializers_read(self, model, fields):
        assert fields <= set(getattr(kfp_server_api, model).openapi_types)
