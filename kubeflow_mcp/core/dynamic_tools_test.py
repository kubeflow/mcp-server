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

"""Tests for dynamic tool discovery, execution, and cache invalidation."""

import math
import sys
import types

import pytest

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.core import dynamic_tools
from kubeflow_mcp.core.resilience import CircuitState, get_breaker

_calls: list[str] = []


def probe_tool(name: str) -> dict:
    """Read-only tool used to exercise execute_tool."""
    _calls.append(name)
    return {"success": True, "data": {"name": name}}


def failing_tool(name: str) -> dict:
    """Tool that raises, standing in for an SDK or API failure."""
    raise RuntimeError("cluster unreachable")


_admin_calls: list[str] = []


def admin_tool(name: str) -> dict:
    """Harmless admin-only stub used to verify registry isolation."""
    _admin_calls.append(name)
    return {"success": True, "data": {"admin_name": name}}


@pytest.fixture(autouse=True)
def _registry():
    _calls.clear()
    _admin_calls.clear()
    dynamic_tools._EmbeddingCache._shared_model = None
    dynamic_tools.init_dynamic_tools([probe_tool, failing_tool], {})
    yield
    dynamic_tools.TOOL_REGISTRY.clear()
    dynamic_tools.TOOL_HIERARCHY.clear()
    dynamic_tools._embedding_cache.reset()
    dynamic_tools._EmbeddingCache._shared_model = None


@pytest.fixture
def offline_model(monkeypatch):
    """sentence-transformers is installed, but the model download fails."""
    import sys
    from types import SimpleNamespace

    load_attempts = []

    def load_model(*_args, **_kwargs):
        load_attempts.append(1)
        raise OSError("We couldn't connect to 'https://huggingface.co' to load this model")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=load_model)
    )
    dynamic_tools._embedding_cache.reset()
    yield load_attempts
    dynamic_tools._embedding_cache.reset()


def test_find_tools_falls_back_to_keywords_when_model_cannot_load(offline_model):
    result = dynamic_tools.find_tools("probe tool")

    assert result["mode"] == "keyword_fallback"
    assert result["tools"][0]["name"] == "probe_tool"


@pytest.fixture
def model_that_cannot_encode(monkeypatch):
    """The model loads, but encoding the tool descriptions fails."""
    import sys
    from types import SimpleNamespace

    class _Model:
        def encode(self, *_args, **_kwargs):
            raise RuntimeError("CUDA error: no kernel image is available")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=lambda *_a, **_k: _Model()),
    )
    dynamic_tools._embedding_cache.reset()
    yield
    dynamic_tools._embedding_cache.reset()


def test_find_tools_falls_back_when_the_model_cannot_encode(model_that_cannot_encode):
    result = dynamic_tools.find_tools("probe tool")

    assert result["mode"] == "keyword_fallback"
    assert result["tools"][0]["name"] == "probe_tool"


def test_find_tools_does_not_retry_failed_model_load(offline_model):
    for _ in range(3):
        dynamic_tools.find_tools("probe tool")

    assert len(offline_model) == 1


def test_embedding_cache_reset_retries_model_load(offline_model):
    dynamic_tools.find_tools("probe tool")
    dynamic_tools._embedding_cache.reset()
    dynamic_tools.find_tools("probe tool")

    assert len(offline_model) == 2


@pytest.fixture
def broken_native_dependency(monkeypatch):
    """The package is installed, but importing from it fails."""
    import sys

    class _BrokenModule:
        def __getattr__(self, name):
            raise OSError("libtorch_cpu.so: cannot open shared object file")

    monkeypatch.setitem(sys.modules, "sentence_transformers", _BrokenModule())
    dynamic_tools._embedding_cache.reset()
    yield
    dynamic_tools._embedding_cache.reset()


def test_find_tools_falls_back_when_the_import_is_broken(broken_native_dependency):
    result = dynamic_tools.find_tools("probe tool")

    assert result["mode"] == "keyword_fallback"
    assert result["tools"][0]["name"] == "probe_tool"


def test_init_dynamic_tools_lets_the_model_be_retried(offline_model):
    dynamic_tools.find_tools("probe tool")
    assert len(offline_model) == 1

    dynamic_tools.init_dynamic_tools([probe_tool, failing_tool], {})
    dynamic_tools.find_tools("probe tool")

    assert len(offline_model) == 2


@pytest.fixture
def slow_offline_model(monkeypatch):
    """Loading takes a moment before it fails, so threads overlap inside the load."""
    import sys
    import time
    from types import SimpleNamespace

    load_attempts = []

    def load_model(*_args, **_kwargs):
        load_attempts.append(1)
        time.sleep(0.05)
        raise OSError("We couldn't connect to 'https://huggingface.co' to load this model")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=load_model)
    )
    dynamic_tools._embedding_cache.reset()
    yield load_attempts
    dynamic_tools._embedding_cache.reset()


def test_concurrent_find_tools_loads_the_model_once(slow_offline_model):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(dynamic_tools.find_tools, "probe tool") for _ in range(8)]
        results = [f.result() for f in futures]

    assert all(result["mode"] == "keyword_fallback" for result in results)
    assert len(slow_offline_model) == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {"nmae": "typo"},
        {"name": "ok", "extra": "unexpected"},
        {},
    ],
)
def test_bad_arguments_return_validation_error(arguments):
    result = dynamic_tools.execute_tool("probe_tool", arguments)

    assert result["error_code"] == ErrorCode.VALIDATION_ERROR
    assert result["tool"] == "probe_tool"
    assert _calls == []


def test_bad_arguments_do_not_open_breaker():
    for _ in range(get_breaker("probe_tool").failure_threshold + 2):
        dynamic_tools.execute_tool("probe_tool", {"nmae": "typo"})

    breaker = get_breaker("probe_tool")
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0

    result = dynamic_tools.execute_tool("probe_tool", {"name": "ok"})
    assert result == {"success": True, "data": {"name": "ok"}}


def test_bad_arguments_do_not_consume_half_open_slot():
    breaker = get_breaker("probe_tool")
    breaker.state = CircuitState.HALF_OPEN
    breaker.half_open_calls = 0

    dynamic_tools.execute_tool("probe_tool", {"nmae": "typo"})

    assert breaker.half_open_calls == 0


def test_tool_exception_still_counts_as_breaker_failure():
    breaker = get_breaker("failing_tool")
    for _ in range(breaker.failure_threshold):
        result = dynamic_tools.execute_tool("failing_tool", {"name": "x"})
        assert result["error_code"] == ErrorCode.SDK_ERROR

    assert breaker.state == CircuitState.OPEN


def test_dynamic_tool_registries_are_independent():
    registry_a = dynamic_tools.DynamicToolRegistry([probe_tool], {})
    registry_b = dynamic_tools.DynamicToolRegistry([admin_tool], {})

    listed_a = registry_a.list_tools("tool")["matching_tools"]
    listed_b = registry_b.list_tools("tool")["matching_tools"]
    assert [tool["name"] for tool in listed_a] == ["probe_tool"]
    assert [tool["name"] for tool in listed_b] == ["admin_tool"]

    described_a = registry_a.describe_tools(["probe_tool", "admin_tool"])["tools"]
    described_b = registry_b.describe_tools(["probe_tool", "admin_tool"])["tools"]
    assert described_a[0]["name"] == "probe_tool"
    assert described_a[1] == {"name": "admin_tool", "error": "Tool not found"}
    assert described_b[0] == {"name": "probe_tool", "error": "Tool not found"}
    assert described_b[1]["name"] == "admin_tool"

    result_a = registry_a.execute_tool("admin_tool", {"name": "hidden"})
    assert result_a["error"] == "Tool 'admin_tool' not found"
    assert result_a["available"] == ["probe_tool"]
    assert _admin_calls == []

    result_b = registry_b.execute_tool("admin_tool", {"name": "allowed"})
    assert result_b == {"success": True, "data": {"admin_name": "allowed"}}
    assert _admin_calls == ["allowed"]


# Semantic search needs sentence-transformers and numpy, which are not dependencies of
# this package, so the test fakes both to exercise the embedding path deterministically.


class _Vector(list):
    def tolist(self) -> list[float]:
        return list(self)


class _FakeEmbeddingModel:
    """Embeds text as keyword counts, standing in for sentence-transformers."""

    _VOCAB = ("alpha", "beta")

    def encode(self, texts: list[str]) -> list[_Vector]:
        return [_Vector(float(t.lower().count(w)) for w in self._VOCAB) for t in texts]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


_FAKE_NUMPY = types.SimpleNamespace(
    dot=_dot, linalg=types.SimpleNamespace(norm=lambda v: math.sqrt(_dot(v, v)))
)


def alpha_tool() -> dict:
    """Alpha tool."""
    return {}


def beta_tool() -> dict:
    """Beta tool."""
    return {}


def test_semantic_search_is_isolated_between_registries(monkeypatch):
    fake_module = types.SimpleNamespace(SentenceTransformer=lambda _name: _FakeEmbeddingModel())
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setitem(sys.modules, "numpy", _FAKE_NUMPY)

    registry_a = dynamic_tools.DynamicToolRegistry([alpha_tool], {})
    registry_b = dynamic_tools.DynamicToolRegistry([beta_tool], {})

    results_a = registry_a.find_tools("alpha")["tools"]
    results_b = registry_b.find_tools("beta")["tools"]

    assert [tool["name"] for tool in results_a] == ["alpha_tool"]
    assert [tool["name"] for tool in results_b] == ["beta_tool"]
    assert registry_a._embedding_cache is not registry_b._embedding_cache
    assert registry_a._embedding_cache._embeddings is not registry_b._embedding_cache._embeddings


def test_embedding_model_is_shared_between_registries(monkeypatch):
    fake_module = types.SimpleNamespace(SentenceTransformer=lambda _name: _FakeEmbeddingModel())
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setitem(sys.modules, "numpy", _FAKE_NUMPY)
    monkeypatch.setattr(dynamic_tools._EmbeddingCache, "_shared_model", None)

    registry_a = dynamic_tools.DynamicToolRegistry([alpha_tool], {})
    registry_b = dynamic_tools.DynamicToolRegistry([beta_tool], {})

    registry_a.find_tools("alpha")
    registry_b.find_tools("beta")

    assert registry_a._embedding_cache._model is registry_b._embedding_cache._model
    assert registry_a._embedding_cache._embeddings is not registry_b._embedding_cache._embeddings
    assert "alpha_tool" in registry_a._embedding_cache._embeddings
    assert "beta_tool" in registry_b._embedding_cache._embeddings
    assert "beta_tool" not in registry_a._embedding_cache._embeddings
    assert "alpha_tool" not in registry_b._embedding_cache._embeddings


def test_keyword_fallback_is_isolated_between_registries(monkeypatch):
    def fail_model_load(*_args, **_kwargs):
        raise OSError("model unavailable")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=fail_model_load),
    )
    registry_a = dynamic_tools.DynamicToolRegistry([alpha_tool], {})
    registry_b = dynamic_tools.DynamicToolRegistry([beta_tool], {})

    results_a = registry_a.find_tools("beta")
    results_b = registry_b.find_tools("beta")

    assert results_a["mode"] == "keyword_fallback"
    assert results_a["tools"] == []
    assert results_b["mode"] == "keyword_fallback"
    assert [tool["name"] for tool in results_b["tools"]] == ["beta_tool"]


def test_find_tools_uses_rebuilt_registry_after_reinit(monkeypatch):
    fake_module = types.SimpleNamespace(SentenceTransformer=lambda _name: _FakeEmbeddingModel())
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setitem(sys.modules, "numpy", _FAKE_NUMPY)

    dynamic_tools.init_dynamic_tools([alpha_tool], {})
    first = dynamic_tools.find_tools("alpha")  # populates the embedding cache
    assert [t["name"] for t in first["tools"]] == ["alpha_tool"]

    dynamic_tools.init_dynamic_tools([beta_tool], {})
    second = dynamic_tools.find_tools("beta")

    assert "mode" not in second  # semantic search, not the keyword fallback
    assert [t["name"] for t in second["tools"]] == ["beta_tool"]
