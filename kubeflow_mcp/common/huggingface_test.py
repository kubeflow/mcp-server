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

"""Tests for the shared Hugging Face reference checks."""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from kubeflow_mcp.common.huggingface import (
    check_hf_dataset_exists,
    hf_repo_id,
    hub_confirms_missing,
    is_confirmed_missing,
    validate_hf_reference,
)


def _hub_error(cls, status: int):
    request = httpx.Request("GET", "https://huggingface.co/api/datasets/x")
    return cls("missing", response=httpx.Response(status, request=request))


@pytest.mark.parametrize(
    ("uri", "dataset", "expected"),
    [
        ("hf://org/ds", True, "org/ds"),
        ("hf://org/ds/sub/path", True, "org/ds"),
        ("hf://ds", True, None),
        ("hf://org/../x", True, None),
        ("hf://gpt2", False, "gpt2"),
        ("hf://org/model", False, "org/model"),
        ("hf://org/model/extra", False, None),
        ("hf://org/bad name", True, None),
    ],
)
def test_hf_repo_id(uri, dataset, expected):
    assert hf_repo_id(uri, dataset=dataset) == expected


class TestValidateHfReference:
    def test_valid_dataset_passes(self):
        assert validate_hf_reference("hf://org/ds", "dataset", dataset=True) is None

    def test_non_hf_scheme_is_left_alone(self):
        assert validate_hf_reference("s3://bucket/my data", "dataset", dataset=True) is None

    def test_whitespace_is_rejected(self):
        err = validate_hf_reference(" hf://org/ds", "dataset", dataset=True)
        assert err.error_code == "VALIDATION_ERROR"
        assert "whitespace" in err.error

    def test_malformed_dataset_names_the_expected_form(self):
        err = validate_hf_reference("hf://ds", "dataset", dataset=True)
        assert "hf://<org>/<dataset>" in err.error


class TestHubChecks:
    def test_only_a_plain_404_is_confirmed_missing(self):
        assert is_confirmed_missing(_hub_error(RepositoryNotFoundError, 404))
        assert not is_confirmed_missing(_hub_error(RepositoryNotFoundError, 401))
        assert not is_confirmed_missing(_hub_error(GatedRepoError, 404))
        assert not is_confirmed_missing(TimeoutError("slow"))

    def test_hub_outage_does_not_block(self):
        fetch = MagicMock(side_effect=TimeoutError("hub down"))
        assert hub_confirms_missing(fetch, "org/ds", "dataset") is False
        fetch.assert_called_once_with("org/ds", timeout=10)

    def test_missing_dataset_is_rejected(self):
        with patch(
            "huggingface_hub.dataset_info", side_effect=_hub_error(RepositoryNotFoundError, 404)
        ):
            err = check_hf_dataset_exists("hf://org/missing")
        assert err.error_code == "VALIDATION_ERROR"
        assert "org/missing" in err.error

    def test_existing_dataset_passes(self):
        with patch("huggingface_hub.dataset_info") as info:
            assert check_hf_dataset_exists("hf://org/ds/train") is None
        info.assert_called_once_with("org/ds", timeout=10)

    @pytest.mark.parametrize("uri", ["s3://bucket/data", "hf://ds"])
    def test_non_hf_or_malformed_is_not_looked_up(self, uri):
        with patch("huggingface_hub.dataset_info") as info:
            assert check_hf_dataset_exists(uri) is None
        info.assert_not_called()
