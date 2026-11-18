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

"""Trainer-specific constants."""

MAX_DIAGNOSTIC_PODS = 32
MAX_DIAGNOSTIC_CHARS = 16000
MAX_DIAGNOSTIC_MESSAGE_CHARS = 1000

TRAINER_CONTROLLER_LABEL_SELECTORS: tuple[str, ...] = (
    # Default Helm chart labels.
    "app.kubernetes.io/name=kubeflow-trainer,app.kubernetes.io/component=manager",
    # Kustomize installations still use trainer, including current releases.
    "app.kubernetes.io/name=trainer,app.kubernetes.io/component=manager",
    # Retain the existing compatibility fallback.
    "app.kubernetes.io/name=trainer,app.kubernetes.io/part-of=kubeflow",
)
