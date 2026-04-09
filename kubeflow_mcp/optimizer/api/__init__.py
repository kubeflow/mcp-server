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

"""Optimizer API tools - Ready for contributors.

This package will contain MCP tools for Katib hyperparameter optimization.

Suggested file structure:
    api/
    ├── __init__.py         # This file - exports all tools
    ├── optimization.py     # create_optimization_job
    ├── discovery.py        # list_optimization_jobs, get_optimization_job
    ├── monitoring.py       # get_optimization_logs, get_optimization_events, wait_for_optimization
    └── lifecycle.py        # delete_optimization_job

Example tool implementation:

    def create_optimization_job(
        objective: str,
        search_space: dict,
        algorithm: str = "random",
        max_trials: int = 10,
        parallel_trials: int = 2,
        confirmed: bool = False,
    ) -> dict:
        '''Create a hyperparameter optimization experiment.

        Args:
            objective: Metric to optimize (e.g., "accuracy", "loss")
            search_space: Parameter ranges to explore
            algorithm: Search algorithm (random, grid, bayesian, hyperband)
            max_trials: Maximum number of trials
            parallel_trials: Trials to run in parallel
            confirmed: Set True to submit (preview mode by default)

        Returns:
            Preview config or job creation result
        '''
        if not confirmed:
            return PreviewResponse(
                status="preview",
                config={...},
            ).model_dump()

        client = get_optimizer_client()
        job_name = client.optimize(...)
        return ToolResponse(data={"job_id": job_name}).model_dump()
"""

__all__: list[str] = []
