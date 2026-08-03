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

"""Optimizer-specific constants.

Uses ``kubeflow.optimizer.constants.constants`` for CRD group/version/plurals
so values stay in sync with the SDK. Additional MCP-specific constants
(controller labels, CRD names) are defined here.
"""

from kubeflow.optimizer.constants import constants as sdk_constants

# ─── Re-exported from SDK ───────────────────────────────────────────────────
# Use these instead of hardcoding strings so they track SDK changes.

KATIB_API_GROUP = sdk_constants.GROUP  # "kubeflow.org"
KATIB_API_VERSION = sdk_constants.VERSION  # "v1beta1"
EXPERIMENT_PLURAL = sdk_constants.EXPERIMENT_PLURAL  # "experiments"
TRIAL_PLURAL = sdk_constants.TRIAL_PLURAL  # "trials"
EXPERIMENT_KIND = sdk_constants.EXPERIMENT_KIND  # "Experiment"

# The SDK does not expose a Suggestion plural constant, so it is defined here.
# Used for direct CustomObjectsApi access (list_suggestions, get_suggestion).
SUGGESTION_PLURAL = "suggestions"

# ─── OptimizationJob status values (from SDK) ──────────────────────────────

OPTIMIZATION_JOB_CREATED = sdk_constants.OPTIMIZATION_JOB_CREATED
OPTIMIZATION_JOB_RUNNING = sdk_constants.OPTIMIZATION_JOB_RUNNING
OPTIMIZATION_JOB_COMPLETE = sdk_constants.OPTIMIZATION_JOB_COMPLETE
OPTIMIZATION_JOB_FAILED = sdk_constants.OPTIMIZATION_JOB_FAILED

# ─── MCP-specific constants ────────────────────────────────────────────────
# These are not in the SDK and are specific to MCP server's pre-flight checks.

EXPERIMENT_CRD_NAME = f"{EXPERIMENT_PLURAL}.{KATIB_API_GROUP}"
KATIB_CONTROLLER_LABELS = "katib.kubeflow.org/component=controller"
KATIB_CONTROLLER_NAMESPACE_DEFAULT = "kubeflow"
