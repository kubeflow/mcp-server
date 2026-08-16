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

"""Tests for planning helpers: HuggingFace model ID validation and suggestions."""

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from huggingface_hub.errors import RepositoryNotFoundError
from tests.common import TestCase

from kubeflow_mcp.trainer.api.planning import (
    _estimate_from_params,
    _get_model_info_from_hf,
    _parse_k8s_version,
    _suggest_hf_model_ids,
    estimate_resources,
)


def _fake_models(*ids):
    return [SimpleNamespace(id=model_id) for model_id in ids]


def _repo_not_found():
    # The concrete exception model_info raises for a nonexistent repo.
    request = httpx.Request("GET", "https://huggingface.co/api/models/x")
    return RepositoryNotFoundError(
        "Repository Not Found", response=httpx.Response(404, request=request)
    )


def test_suggest_normalizes_ollama_tag():
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct")
        suggestions = _suggest_hf_model_ids("qwen3:8b")

    assert suggestions == ["Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"]
    # The Ollama-style ":8b" tag is dropped before searching the Hub.
    assert mock_list.call_args.kwargs["search"] == "qwen3"


def test_suggest_drops_hf_prefix():
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models("google/gemma-2b")
        _suggest_hf_model_ids("hf://google/gemma:2b")

    assert mock_list.call_args.kwargs["search"] == "google/gemma"


def test_suggest_returns_empty_when_hub_errors():
    with patch("huggingface_hub.list_models", side_effect=RuntimeError("offline")):
        assert _suggest_hf_model_ids("qwen3:8b") == []


def test_suggest_returns_empty_for_blank_input():
    # No Hub call is needed when normalization leaves nothing to search for.
    with patch("huggingface_hub.list_models") as mock_list:
        assert _suggest_hf_model_ids("hf://") == []
    mock_list.assert_not_called()


def test_invalid_format_attaches_suggestions():
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct")
        result = _get_model_info_from_hf("qwen3:8b")

    assert result["error"] == "Invalid HuggingFace model ID format: 'qwen3:8b'"
    assert result["suggestions"] == ["Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"]


def test_invalid_format_omits_suggestions_when_none_found():
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models()
        result = _get_model_info_from_hf("qwen3:8b")

    assert result["error"] == "Invalid HuggingFace model ID format: 'qwen3:8b'"
    assert "suggestions" not in result


def test_invalid_format_omits_suggestions_when_hub_errors():
    with patch("huggingface_hub.list_models", side_effect=RuntimeError("offline")):
        result = _get_model_info_from_hf("not a model id")

    assert "Invalid HuggingFace model ID format" in result["error"]
    assert "suggestions" not in result


# Tool-boundary tests: estimate_resources re-wraps the helper's error, so verify
# the suggestions actually survive into the user-facing response. pre_flight()
# delegates model handling to estimate_resources(), so this covers both tools.
def test_estimate_resources_surfaces_suggestions_at_tool_boundary():
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct")
        result = estimate_resources("qwen3:8b")

    assert result["success"] is False
    assert "Invalid HuggingFace model ID format" in result["error"]
    assert result["details"]["suggestions"] == ["Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"]


def test_estimate_resources_omits_suggestions_when_none_found():
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models()
        result = estimate_resources("qwen3:8b")

    assert result["success"] is False
    assert "suggestions" not in result["details"]


def test_suggest_ranks_by_downloads():
    # Ranking by downloads surfaces canonical repos ahead of community fine-tunes.
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models("Qwen/Qwen3-8B")
        _suggest_hf_model_ids("qwen3:8b")

    assert mock_list.call_args.kwargs.get("sort") == "downloads"


def test_suggest_falls_back_to_model_name_when_org_typo_finds_nothing():
    # A typo'd org (meta-lama vs meta-llama) makes the full "org/name" search
    # empty; retry on just the model name so the user still gets a suggestion.
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.side_effect = [_fake_models(), _fake_models("meta-llama/Llama-3.2-1B")]
        suggestions = _suggest_hf_model_ids("meta-lama/Llama-3")

    assert suggestions == ["meta-llama/Llama-3.2-1B"]
    assert [c.kwargs["search"] for c in mock_list.call_args_list] == [
        "meta-lama/Llama-3",
        "Llama-3",
    ]


def test_suggest_skips_name_fallback_when_name_too_short():
    # A very short model name ("ab") makes a bare search too broad to be a useful
    # "did you mean", so the name-only retry is skipped: only the full term runs.
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models()
        suggestions = _suggest_hf_model_ids("wrongorg/ab")

    assert suggestions == []
    assert mock_list.call_count == 1
    assert mock_list.call_args.kwargs["search"] == "wrongorg/ab"


def test_not_found_id_attaches_suggestions():
    # A valid-format but nonexistent id fails at model_info (not the regex check),
    # and should still get "did you mean" suggestions.
    with (
        patch("huggingface_hub.model_info", side_effect=_repo_not_found()),
        patch("huggingface_hub.list_models") as mock_list,
    ):
        mock_list.return_value = _fake_models("meta-llama/Llama-3.2-1B")
        result = _get_model_info_from_hf("meta-lama/Llama-3")

    assert "error" in result
    assert result["suggestions"] == ["meta-llama/Llama-3.2-1B"]


def test_not_found_id_omits_suggestions_when_hub_errors():
    with (
        patch("huggingface_hub.model_info", side_effect=_repo_not_found()),
        patch("huggingface_hub.list_models", side_effect=RuntimeError("offline")),
    ):
        result = _get_model_info_from_hf("meta-lama/Llama-3")

    assert "error" in result
    assert "suggestions" not in result


def test_estimate_resources_surfaces_suggestions_on_not_found():
    with (
        patch("huggingface_hub.model_info", side_effect=_repo_not_found()),
        patch("huggingface_hub.list_models") as mock_list,
    ):
        mock_list.return_value = _fake_models("meta-llama/Llama-3.2-1B")
        result = estimate_resources("meta-lama/Llama-3")

    assert result["success"] is False
    assert result["details"]["suggestions"] == ["meta-llama/Llama-3.2-1B"]


def test_non_not_found_error_skips_suggestions():
    # Only a missing repo should trigger suggestions. Other failures (timeout,
    # auth/rate-limit, metadata errors) surface unchanged and make no extra Hub
    # request, so an outage or a gated model does not get misleading matches.
    with (
        patch("huggingface_hub.model_info", side_effect=TimeoutError("timed out")),
        patch("huggingface_hub.list_models") as mock_list,
    ):
        result = _get_model_info_from_hf("meta-llama/Llama-3.2-1B")

    assert "error" in result
    assert "suggestions" not in result
    mock_list.assert_not_called()


# ─── Pure helpers (version parse / resource estimate) ───────────────────────


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="parses v1.29.3",
            config={"git_version": "v1.29.3"},
            expected_output=(1, 29),
        ),
        TestCase(
            name="parses without v prefix",
            config={"git_version": "1.30.0"},
            expected_output=(1, 30),
        ),
        TestCase(
            name="invalid version returns None",
            config={"git_version": "garbage"},
            expected_output=None,
        ),
    ],
)
def test_parse_k8s_version(test_case):
    assert _parse_k8s_version(**test_case.config) == test_case.expected_output


def test_estimate_from_params_small_model():
    result = _estimate_from_params(7e9, batch_size=4, quantization="bf16")
    assert result["gpu_count"] >= 1
    assert result["gpu_memory_gb"] >= 1
    assert result["params_billions"] == 7.0
    assert result["quantization"] == "bf16"
    assert "weights_gb" in result["breakdown"]
