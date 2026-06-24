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

from kubeflow_mcp.trainer.api.planning import (
    _get_model_info_from_hf,
    _suggest_hf_model_ids,
    estimate_resources,
)


def _fake_models(*ids):
    return [SimpleNamespace(id=model_id) for model_id in ids]


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
