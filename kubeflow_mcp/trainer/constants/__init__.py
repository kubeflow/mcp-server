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

TRAINER_CONTROLLER_LABEL_SELECTORS: tuple[str, ...] = (
    "app.kubernetes.io/name=kubeflow-trainer,app.kubernetes.io/component=manager",
    "app.kubernetes.io/name=trainer,app.kubernetes.io/component=manager",
    "app.kubernetes.io/name=trainer,app.kubernetes.io/part-of=kubeflow",
)
