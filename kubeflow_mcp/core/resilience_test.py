"""Tests for resilience patterns."""

import asyncio

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


# ─── retry_with_backoff_async ─────────────────────────────────────────────────


def test_retry_with_backoff_async_succeeds_first_try():
    from kubeflow_mcp.core.resilience import retry_with_backoff_async

    async def _run():
        async def succeed():
            return 42

        return await retry_with_backoff_async(succeed, base_delay=0)

    assert asyncio.run(_run()) == 42


def test_retry_with_backoff_async_retries_then_succeeds():
    from kubeflow_mcp.core.resilience import retry_with_backoff_async

    call_count = 0

    async def _run():
        nonlocal call_count

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "ok"

        return await retry_with_backoff_async(flaky, max_retries=3, base_delay=0)

    assert asyncio.run(_run()) == "ok"
    assert call_count == 3


def test_retry_with_backoff_async_raises_after_exhaustion():
    from kubeflow_mcp.core.resilience import retry_with_backoff_async

    async def _run():
        async def always_fail():
            raise RuntimeError("always")

        return await retry_with_backoff_async(always_fail, max_retries=2, base_delay=0)

    with pytest.raises(RuntimeError, match="always"):
        asyncio.run(_run())


# ─── RateLimiter ──────────────────────────────────────────────────────────────


def test_rate_limiter_allows_initial_burst():
    from kubeflow_mcp.core.resilience import RateLimiter

    rl = RateLimiter(rate=10.0, capacity=10.0)
    for _ in range(10):
        assert rl.acquire() is True


def test_rate_limiter_denies_when_empty():
    from kubeflow_mcp.core.resilience import RateLimiter

    rl = RateLimiter(rate=1.0, capacity=1.0)
    rl.acquire()  # consume the single token
    assert rl.acquire() is False


def test_rate_limiter_partial_tokens():
    from kubeflow_mcp.core.resilience import RateLimiter

    rl = RateLimiter(rate=10.0, capacity=5.0)
    assert rl.acquire(tokens=3.0) is True
    assert rl.acquire(tokens=3.0) is False  # only 2 remain


# ─── SessionManager ───────────────────────────────────────────────────────────


def test_session_manager_not_stale_initially():
    from kubeflow_mcp.core.resilience import SessionManager

    sm = SessionManager(max_age=300.0)
    assert sm.is_stale() is False


def test_session_manager_stale_after_max_age():
    import time

    from kubeflow_mcp.core.resilience import SessionManager

    sm = SessionManager(max_age=0.01)
    sm.record_activity()
    time.sleep(0.05)
    assert sm.is_stale() is True


def test_session_manager_not_stale_after_recent_activity():
    from kubeflow_mcp.core.resilience import SessionManager

    sm = SessionManager(max_age=300.0)
    sm.record_activity()
    assert sm.is_stale() is False


# ─── configure_circuit_breaker ────────────────────────────────────────────────


def test_configure_circuit_breaker_affects_new_breakers():
    from kubeflow_mcp.core.resilience import configure_circuit_breaker, get_breaker, reset_breakers

    reset_breakers()
    configure_circuit_breaker(failure_threshold=2, recovery_timeout=5.0)
    breaker = get_breaker("configured-tool")
    assert breaker.failure_threshold == 2
    assert breaker.recovery_timeout == 5.0
    reset_breakers()
