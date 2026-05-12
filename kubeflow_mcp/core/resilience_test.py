"""Tests for resilience patterns."""

import pytest

from kubeflow_mcp.core.resilience import (
    CircuitBreaker,
    CircuitState,
    RateLimiter,
    get_breaker,
    reset_breakers,
    retry_with_backoff,
    with_circuit_breaker,
)


def test_circuit_breaker_starts_closed():
    cb = CircuitBreaker()
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute()


def test_circuit_breaker_opens_on_failures():
    cb = CircuitBreaker(failure_threshold=3)
    assert cb.can_execute()

    for _ in range(3):
        cb.record_failure()

    assert cb.state == CircuitState.OPEN
    assert not cb.can_execute()


def test_circuit_breaker_success_resets_count():
    cb = CircuitBreaker(failure_threshold=3)

    cb.record_failure()
    cb.record_failure()
    cb.record_success()

    assert cb.failure_count == 0
    assert cb.state == CircuitState.CLOSED


def test_rate_limiter_allows_under_limit():
    rl = RateLimiter(rate=10, capacity=10)
    for _ in range(5):
        assert rl.acquire()


def test_rate_limiter_blocks_over_limit():
    rl = RateLimiter(rate=1, capacity=2)
    assert rl.acquire()
    assert rl.acquire()
    assert not rl.acquire()


def test_retry_with_backoff_succeeds():
    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary error")
        return "success"

    result = flaky_function()
    assert result == "success"
    assert call_count == 3


def test_retry_with_backoff_exhausted():
    @retry_with_backoff(max_retries=2, base_delay=0.01)
    def always_fails():
        raise ValueError("Permanent error")

    with pytest.raises(ValueError, match="Permanent error"):
        always_fails()


def test_with_circuit_breaker_decorator():
    cb = CircuitBreaker(failure_threshold=2)
    call_count = 0

    @with_circuit_breaker(cb)
    def failing_function():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("fail")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            failing_function()

    assert cb.state == CircuitState.OPEN

    with pytest.raises(RuntimeError, match="Circuit breaker open"):
        failing_function()

    assert call_count == 2


# --- Named per-function circuit breakers ---


def test_get_breaker_same_name_returns_same_object():
    reset_breakers()
    b1 = get_breaker("tool_a")
    b2 = get_breaker("tool_a")
    assert b1 is b2


def test_get_breaker_different_names_are_independent():
    reset_breakers()
    b_a = get_breaker("tool_a")
    b_b = get_breaker("tool_b")
    assert b_a is not b_b

    # Trip tool_a
    b_a.failure_threshold = 1
    b_a.record_failure()
    assert b_a.state == CircuitState.OPEN

    # tool_b unaffected
    assert b_b.state == CircuitState.CLOSED
    assert b_b.can_execute()


def test_reset_breakers_clears_state():
    reset_breakers()
    b = get_breaker("tool_x")
    b.failure_threshold = 1
    b.record_failure()
    assert b.state == CircuitState.OPEN

    reset_breakers()
    fresh = get_breaker("tool_x")
    assert fresh.state == CircuitState.CLOSED
    assert fresh is not b  # new object after reset


def test_with_circuit_breaker_auto_naming_uses_function_name():
    reset_breakers()

    @with_circuit_breaker()
    def auto_named_tool():
        raise RuntimeError("fail")

    b = get_breaker("auto_named_tool")
    assert b is not None

    b.failure_threshold = 1
    with pytest.raises(RuntimeError, match="fail"):
        auto_named_tool()

    assert b.state == CircuitState.OPEN

    # A second decorated function gets its own independent breaker
    @with_circuit_breaker()
    def another_tool():
        return "ok"

    assert another_tool() == "ok"  # unaffected by auto_named_tool's open circuit
