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
"""Canary tests for safety-critical constants.

These exist to catch silent regressions if someone tweaks a limit without
realising the downstream security/resource impact.
"""

from kubeflow_mcp.common.constants import MIN_K8S_VERSION, MIN_TRAINER_CRD_VERSION
from kubeflow_mcp.core.security import (
    MAX_BATCH_SIZE,
    MAX_EPOCHS,
    MAX_GPU_PER_NODE,
    MAX_LORA_RANK,
    MAX_NAME_LENGTH,
    MAX_NODES,
    MAX_PACKAGES,
    MAX_SCRIPT_SIZE,
)

# ─── Training parameter safety bounds ─────────────────────────────────────────


def test_max_batch_size_is_sane():
    assert 1 <= MAX_BATCH_SIZE <= 65536, f"MAX_BATCH_SIZE={MAX_BATCH_SIZE} looks wrong"


def test_max_epochs_is_sane():
    assert 1 <= MAX_EPOCHS <= 100_000, f"MAX_EPOCHS={MAX_EPOCHS} looks wrong"


def test_max_nodes_caps_at_reasonable_cluster_size():
    assert 1 <= MAX_NODES <= 10_000, f"MAX_NODES={MAX_NODES} looks wrong"


def test_max_gpu_per_node_matches_hardware_reality():
    # Current largest instances have 8–16 GPUs; leave headroom but cap at 128
    assert 1 <= MAX_GPU_PER_NODE <= 128, f"MAX_GPU_PER_NODE={MAX_GPU_PER_NODE} looks wrong"


def test_max_lora_rank_is_power_of_two_or_reasonable():
    assert 1 <= MAX_LORA_RANK <= 1024, f"MAX_LORA_RANK={MAX_LORA_RANK} looks wrong"


def test_max_script_size_is_at_least_1kb_and_under_10mb():
    assert 1024 <= MAX_SCRIPT_SIZE <= 10_000_000, f"MAX_SCRIPT_SIZE={MAX_SCRIPT_SIZE} looks wrong"


def test_max_packages_is_reasonable():
    assert 1 <= MAX_PACKAGES <= 500, f"MAX_PACKAGES={MAX_PACKAGES} looks wrong"


def test_max_name_length_matches_k8s_spec():
    # K8s names are capped at 63 chars for DNS label compliance
    assert MAX_NAME_LENGTH == 63, f"MAX_NAME_LENGTH={MAX_NAME_LENGTH}, expected 63"


# ─── Kubernetes version floor ─────────────────────────────────────────────────


def test_min_k8s_version_is_tuple_of_two_ints():
    assert isinstance(MIN_K8S_VERSION, tuple)
    assert len(MIN_K8S_VERSION) == 2
    major, minor = MIN_K8S_VERSION
    assert isinstance(major, int) and isinstance(minor, int)


def test_min_k8s_version_is_at_least_1_25():
    major, minor = MIN_K8S_VERSION
    assert (major, minor) >= (1, 25), f"MIN_K8S_VERSION={MIN_K8S_VERSION} is below 1.25"


def test_min_trainer_crd_version_format():
    assert MIN_TRAINER_CRD_VERSION.startswith("v"), (
        f"MIN_TRAINER_CRD_VERSION='{MIN_TRAINER_CRD_VERSION}' should start with 'v'"
    )
