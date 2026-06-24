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
"""Tests for HuggingFace model ID validation and suggestion helpers."""

from types import SimpleNamespace
from unittest.mock import patch

from kubeflow_mcp.trainer.api.planning import (
    _get_model_info_from_hf,
    _suggest_hf_model_ids,
)


def _fake_models(*ids):
    """Mimic huggingface_hub.list_models output (objects exposing ``.id``)."""
    return [SimpleNamespace(id=model_id) for model_id in ids]


def test_suggest_normalises_ollama_tag():
    """An Ollama-style tag like 'qwen3:8b' is reduced to its base search term."""
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct")
        suggestions = _suggest_hf_model_ids("qwen3:8b")

    assert suggestions == ["Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct"]
    _, kwargs = mock_list.call_args
    assert kwargs["search"] == "qwen3"
    assert kwargs["limit"] == 3


def test_suggest_strips_hf_prefix_and_slash():
    """'hf://meta-lama/Llama-3' is normalised into a space-joined search term."""
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models("meta-llama/Llama-3.2-1B")
        suggestions = _suggest_hf_model_ids("hf://meta-lama/Llama-3")

    assert suggestions == ["meta-llama/Llama-3.2-1B"]
    _, kwargs = mock_list.call_args
    assert kwargs["search"] == "meta-lama Llama-3"


def test_suggest_returns_empty_on_lookup_error():
    """A Hub lookup error must not propagate — suggestions are best-effort."""
    with patch("huggingface_hub.list_models", side_effect=RuntimeError("network down")):
        assert _suggest_hf_model_ids("qwen3:8b") == []


def test_suggest_returns_empty_for_blank_term():
    """An input that normalises to empty skips the Hub call entirely."""
    with patch("huggingface_hub.list_models") as mock_list:
        assert _suggest_hf_model_ids("hf://") == []
    mock_list.assert_not_called()


def test_invalid_model_id_includes_suggestions():
    """An invalid model ID returns the format error plus suggestions when available."""
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = _fake_models("Qwen/Qwen3-8B")
        result = _get_model_info_from_hf("qwen3:8b")

    assert result is not None
    assert "Invalid HuggingFace model ID format" in result["error"]
    assert result["suggestions"] == ["Qwen/Qwen3-8B"]


def test_invalid_model_id_omits_suggestions_when_none_found():
    """When the Hub returns nothing, no empty 'suggestions' key is added."""
    with patch("huggingface_hub.list_models") as mock_list:
        mock_list.return_value = []
        result = _get_model_info_from_hf("qwen3:8b")

    assert result is not None
    assert "suggestions" not in result
