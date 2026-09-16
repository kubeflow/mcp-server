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

"""Tests for core/resilience.py — circuit breaker, rate limiter, retry."""

from __future__ import annotations

import time

import pytest

from kubeflow_mcp.core.resilience import (
    CircuitBreaker,
    CircuitState,
    RateLimiter,
    SessionManager,
    configure_circuit_breaker,
    get_breaker,
    reset_breakers,
    retry_with_backoff,
    with_circuit_breaker,
)

# ─── CircuitBreaker ─────────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0

    @pytest.mark.slow
    def test_transitions_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.02)
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    @pytest.mark.slow
    def test_half_open_to_closed_on_successes(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=2)
        cb.record_failure()
        time.sleep(0.02)
        cb.can_execute()  # triggers HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.slow
    def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.can_execute()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    @pytest.mark.slow
    def test_half_open_max_calls_limit(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05, half_open_max_calls=2)
        cb.record_failure()
        time.sleep(0.06)

        assert cb.can_execute() is True
        assert cb.can_execute() is True
        assert cb.can_execute() is False
        assert cb.half_open_calls == 2

    @pytest.mark.slow
    def test_full_window_is_not_replaced_while_probes_are_out(self):
        """A slow probe keeps its window, however long it takes to report."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=1)
        cb.record_failure()
        time.sleep(0.02)

        assert cb.can_execute() is True
        time.sleep(0.02)
        assert cb.can_execute() is False

    def test_release_hands_back_a_half_open_slot(self):
        """A call that says nothing about the backend frees its slot without counting."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0, half_open_max_calls=2)
        cb.record_failure()

        first = cb.acquire()
        cb.acquire()
        assert cb.acquire() is None

        cb.release(first)
        assert cb.half_open_calls == 1
        assert cb.acquire() is not None
        assert cb.state == CircuitState.HALF_OPEN

    def test_release_does_not_reset_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.release(cb.acquire())
        assert cb.failure_count == 2

    def test_late_report_from_previous_window_is_ignored(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0, half_open_max_calls=3)
        cb.record_failure()

        slow = cb.acquire()
        cb.record_failure(cb.acquire())  # sends that window back to OPEN
        current = cb.acquire()  # and this opens the next one

        cb.record_success(slow)
        cb.release(slow)
        cb.record_failure(slow)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.half_open_calls == 1

        # Two fresh successes are one short of closing, so the late one did not count.
        cb.record_success(current)
        cb.record_success(current)
        assert cb.state == CircuitState.HALF_OPEN

    def test_recovery_ignores_wall_clock_jumps(self, monkeypatch):
        """A wall-clock jump must not release the circuit before its timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        cb.record_failure()

        jumped = time.time() + 3600
        monkeypatch.setattr("time.time", lambda: jumped)
        assert cb.can_execute() is False
        assert cb.state == CircuitState.OPEN

    # TODO(test): test thread safety with concurrent record_failure/record_success


# ─── get_breaker / configure ────────────────────────────────────────────────


class TestGetBreaker:
    def test_returns_same_instance(self):
        a = get_breaker("tool_a")
        b = get_breaker("tool_a")
        assert a is b

    def test_different_names_different_instances(self):
        a = get_breaker("tool_x")
        b = get_breaker("tool_y")
        assert a is not b

    def test_configure_affects_new_breakers(self):
        configure_circuit_breaker(failure_threshold=99, recovery_timeout=999.0)
        cb = get_breaker("configured_tool")
        assert cb.failure_threshold == 99
        assert cb.recovery_timeout == 999.0
        configure_circuit_breaker(failure_threshold=5, recovery_timeout=30.0)


class TestResetBreakers:
    def test_clears_all(self):
        original = get_breaker("should_disappear")
        for _ in range(3):
            original.record_failure()
        assert original.failure_count >= 3

        reset_breakers()
        new = get_breaker("should_disappear")
        assert new is not original
        assert new.failure_count == 0


# ─── with_circuit_breaker decorator ─────────────────────────────────────────


class TestWithCircuitBreaker:
    def test_success_path(self):
        @with_circuit_breaker()
        def ok():
            return 42

        assert ok() == 42

    def test_failure_raises_and_records(self):
        breaker = CircuitBreaker(failure_threshold=5)

        @with_circuit_breaker(breaker=breaker)
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            fail()

        assert breaker.failure_count == 1

    def test_open_breaker_raises_runtime_error(self):
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure()

        @with_circuit_breaker(breaker=breaker)
        def blocked():
            return "should not run"

        with pytest.raises(RuntimeError, match="Circuit breaker open"):
            blocked()


# ─── RateLimiter ─────────────────────────────────────────────────────────────


class TestRateLimiter:
    def test_acquire_succeeds_with_capacity(self):
        rl = RateLimiter(rate=100.0, capacity=10.0)
        assert rl.acquire() is True

    def test_acquire_fails_when_exhausted(self):
        rl = RateLimiter(rate=0.0, capacity=1.0)
        rl.acquire()
        assert rl.acquire() is False

    @pytest.mark.slow
    def test_tokens_refill_over_time(self):
        rl = RateLimiter(rate=1000.0, capacity=5.0)
        for _ in range(5):
            rl.acquire()
        time.sleep(0.01)
        assert rl.acquire() is True

    def test_acquire_ignores_wall_clock_jumps(self, monkeypatch):
        """A backward wall-clock step must not drain the bucket.

        time.time is patched because that is the clock the limiter used to read.
        Patching monotonic would assert a step it cannot make.
        """
        rl = RateLimiter(rate=10.0, capacity=5.0)

        monkeypatch.setattr("time.time", lambda: 0.0)
        assert rl.acquire() is True

    # TODO(test): test thread safety with concurrent acquire


# ─── retry_with_backoff ─────────────────────────────────────────────────────


class TestRetryWithBackoff:
    def test_succeeds_on_first_try(self):
        @retry_with_backoff(max_retries=3, base_delay=0.001)
        def ok():
            return "done"

        assert ok() == "done"

    def test_retries_then_succeeds(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.001)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "ok"

        assert flaky() == "ok"
        assert call_count == 3

    def test_raises_after_all_retries_exhausted(self):
        @retry_with_backoff(max_retries=2, base_delay=0.001)
        def always_fail():
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            always_fail()

    # TODO(test): test retryable_exceptions filtering
    # TODO(test): test retry_with_backoff_async


# ─── SessionManager ─────────────────────────────────────────────────────────


class TestSessionManager:
    def test_not_stale_after_activity(self):
        sm = SessionManager(max_age=60.0)
        sm.record_activity()
        assert sm.is_stale() is False

    @pytest.mark.slow
    def test_stale_after_max_age(self):
        sm = SessionManager(max_age=0.01)
        sm.record_activity()
        time.sleep(0.02)
        assert sm.is_stale() is True

    def test_not_stale_when_empty(self):
        sm = SessionManager()
        assert sm.is_stale() is False
