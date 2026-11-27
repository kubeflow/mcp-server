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

"""Resilience patterns: circuit breaker, retry, rate limiting."""

import asyncio
import logging
import random
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Thread-safe circuit breaker for Kubernetes API calls."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time: float = field(default=0.0)
    half_open_calls: int = field(default=0)
    _half_open_successes: int = field(default=0)
    _generation: int = field(default=0)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be greater than zero")
        if self.recovery_timeout < 0:
            raise ValueError("recovery_timeout must be non-negative")
        if self.half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be greater than zero")
        self._lock = threading.Lock()

    def acquire(self) -> int | None:
        """Reserve a call and return its probe generation, or None if it is refused.

        The generation moves on each time a half-open window opens. Passing it back
        when recording lets a late report from an older window be ignored.
        """
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return self._generation

            if self.state == CircuitState.OPEN:
                if time.monotonic() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self._half_open_successes = 0
                    self._generation += 1
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN")
                    self.half_open_calls += 1
                    return self._generation
                return None

            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return self._generation
            return None

    def can_execute(self) -> bool:
        """Check if call is allowed and atomically reserve a slot if half-open."""
        return self.acquire() is not None

    def _is_stale(self, generation: int | None) -> bool:
        """Whether a report belongs to an older probe window. Caller holds the lock."""
        return generation is not None and generation != self._generation

    def record_success(self, generation: int | None = None) -> None:
        """Record successful call."""
        with self._lock:
            if self._is_stale(generation):
                return
            if self.state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_max_calls:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self._half_open_successes = 0
                    logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
            else:
                self.failure_count = 0

    def record_failure(self, generation: int | None = None) -> None:
        """Record failed call."""
        with self._lock:
            if self._is_stale(generation):
                return
            self.failure_count += 1
            self.last_failure_time = time.monotonic()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self._half_open_successes = 0
                logger.warning("Circuit breaker: HALF_OPEN -> OPEN")
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker: CLOSED -> OPEN (failures={self.failure_count})")

    def release(self, generation: int | None = None) -> None:
        """Hand back a call whose result says nothing about backend health.

        It counts as neither a success nor a failure, so the failure count stays
        put, and a half-open slot is freed for the next probe.
        """
        with self._lock:
            if self._is_stale(generation):
                return
            if self.state == CircuitState.HALF_OPEN and self.half_open_calls > 0:
                self.half_open_calls -= 1


_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()
_cb_defaults: dict[str, float | int] = {
    "failure_threshold": 5,
    "recovery_timeout": 30.0,
}


def configure_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> None:
    """Set default thresholds for new circuit breakers. Call once at startup."""
    if failure_threshold <= 0:
        raise ValueError("failure_threshold must be greater than zero")
    if recovery_timeout < 0:
        raise ValueError("recovery_timeout must be non-negative")
    _cb_defaults["failure_threshold"] = failure_threshold
    _cb_defaults["recovery_timeout"] = recovery_timeout


def get_breaker(name: str) -> CircuitBreaker:
    """Return the named CircuitBreaker, creating it on first access (thread-safe).

    Using per-name breakers prevents a failure in one tool (e.g.
    ``get_runtime(include_packages=True)``) from tripping the circuit for unrelated
    tools (e.g. ``fine_tune``).
    """
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(
                failure_threshold=int(_cb_defaults["failure_threshold"]),
                recovery_timeout=float(_cb_defaults["recovery_timeout"]),
            )
        return _breakers[name]


def reset_breakers() -> None:
    """Clear all circuit breakers (for testing)."""
    with _breakers_lock:
        _breakers.clear()


def with_circuit_breaker(
    breaker: CircuitBreaker | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to wrap function with a circuit breaker.

    If *breaker* is ``None`` a dedicated breaker keyed by the function name
    is created automatically, so each decorated function has its own
    independent failure budget.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cb = breaker if breaker is not None else get_breaker(func.__name__)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            generation = cb.acquire()
            if generation is None:
                raise RuntimeError(f"Circuit breaker open for {func.__name__}")

            try:
                result = func(*args, **kwargs)
                cb.record_success(generation)
                return result
            except Exception:
                cb.record_failure(generation)
                raise

        return wrapper

    return decorator


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.1,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retry with exponential backoff and jitter."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    delay = min(base_delay * (2**attempt), max_delay)
                    delay += random.uniform(-jitter * delay, jitter * delay)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} in {delay:.2f}s"
                    )
                    time.sleep(delay)

            raise last_exception  # type: ignore

        return wrapper

    return decorator


async def retry_with_backoff_async(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.1,
    **kwargs: Any,
) -> Any:
    """Async retry with exponential backoff, max-delay cap, and jitter.

    Mirrors the behaviour of :func:`retry_with_backoff` for async callables.
    ``max_delay`` caps the exponential growth; ``jitter`` randomises the delay
    within ±jitter*delay to avoid the thundering-herd effect.
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                break

            delay = min(base_delay * (2**attempt), max_delay)
            delay += random.uniform(-jitter * delay, jitter * delay)
            name = getattr(func, "__name__", repr(func))
            logger.debug(f"Retry {attempt + 1}/{max_retries} for {name} in {delay:.2f}s")
            await asyncio.sleep(delay)

    raise last_exception  # type: ignore


@dataclass
class RateLimiter:
    """Token bucket rate limiter (thread-safe)."""

    rate: float = 10.0
    capacity: float = 10.0
    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError("rate must be greater than zero")
        if self.capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        self._tokens = self.capacity
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0, user_id: str | None = None) -> bool:
        """Try to acquire tokens. Returns True if successful.

        The *user_id* argument is accepted for interface compatibility with
        :class:`UserRateLimiter`, but ignored by this single-bucket limiter.
        """
        if tokens <= 0:
            raise ValueError("tokens must be greater than zero")

        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False


@dataclass
class UserRateLimiter:
    """Per-user token bucket rate limiter bucketed by user identity (sub claim).

    Prevents noisy-neighbor starvation in multi-user deployments (ROADMAP Phase 4).
    Each distinct user identity receives an independent :class:`RateLimiter` token bucket.
    Includes thread-safe TTL and LRU eviction to prevent unbounded memory growth under
    high-cardinality user identifiers.
    """

    user_rate: float = 10.0
    user_capacity: float = 10.0
    global_rate: float | None = None
    global_capacity: float | None = None
    bucket_ttl: float = 300.0
    max_buckets: int = 10000

    _buckets: dict[str, RateLimiter] = field(init=False, default_factory=dict)
    _last_access: dict[str, float] = field(init=False, default_factory=dict)
    _global_limiter: RateLimiter | None = field(init=False, default=None)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.user_rate <= 0:
            raise ValueError("user_rate must be greater than zero")
        if self.user_capacity <= 0:
            raise ValueError("user_capacity must be greater than zero")
        if self.bucket_ttl <= 0:
            raise ValueError("bucket_ttl must be greater than zero")
        if self.max_buckets <= 0:
            raise ValueError("max_buckets must be greater than zero")

        if self.global_rate is not None:
            if self.global_rate <= 0:
                raise ValueError("global_rate must be greater than zero")
            capacity = (
                self.global_capacity if self.global_capacity is not None else self.global_rate * 2
            )
            if capacity <= 0:
                raise ValueError("global_capacity must be greater than zero")
            self._global_limiter = RateLimiter(rate=self.global_rate, capacity=capacity)
        else:
            self._global_limiter = None

        self._buckets = {}
        self._last_access = OrderedDict()
        self._depleted: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def _save_depletion_state_locked(self, uid: str, bucket: RateLimiter, now: float) -> None:
        """Preserve un-refilled depletion state upon bucket eviction."""
        with bucket._lock:
            elapsed = max(0.0, now - bucket._last_update)
            current_tokens = min(bucket.capacity, bucket._tokens + elapsed * bucket.rate)
        if current_tokens < self.user_capacity:
            self._depleted[uid] = (current_tokens, now)

    def _cleanup_depleted_locked(self, now: float) -> None:
        """Purge depletion records that have reached full capacity over time."""
        full_refill_duration = self.user_capacity / self.user_rate
        expired = [
            uid
            for uid, (_, saved_time) in self._depleted.items()
            if now - saved_time >= full_refill_duration
        ]
        for uid in expired:
            self._depleted.pop(uid, None)

    def _evict_expired_locked(self, now: float) -> int:
        """Evict buckets inactive for longer than bucket_ttl from the oldest end."""
        self._cleanup_depleted_locked(now)
        evicted = 0
        while self._last_access:
            oldest_uid = next(iter(self._last_access))
            last_time = self._last_access[oldest_uid]
            if now - last_time > self.bucket_ttl:
                self._last_access.pop(oldest_uid)
                bucket = self._buckets.pop(oldest_uid, None)
                if bucket is not None:
                    self._save_depletion_state_locked(oldest_uid, bucket, now)
                evicted += 1
            else:
                break
        return evicted

    def _evict_lru_locked(self, now: float) -> None:
        """Evict the least recently used user bucket."""
        if not self._last_access:
            return
        oldest_uid = next(iter(self._last_access))
        self._last_access.pop(oldest_uid)
        bucket = self._buckets.pop(oldest_uid, None)
        if bucket is not None:
            self._save_depletion_state_locked(oldest_uid, bucket, now)

    def acquire(self, tokens: float = 1.0, user_id: str | None = None) -> bool:
        """Try to acquire tokens for a specific user.

        If a global limit is configured, tokens are acquired from the global
        limiter first. Then tokens are acquired from the user's specific
        bucket. If the user bucket denies the tokens, any acquired global
        tokens are refunded.
        """
        if tokens <= 0:
            raise ValueError("tokens must be greater than zero")

        user_key = (user_id or "").strip() or "__anonymous__"

        with self._lock:
            now = time.monotonic()
            self._evict_expired_locked(now)

            # Global ceiling check
            if self._global_limiter is not None and not self._global_limiter.acquire(tokens):
                return False

            bucket = self._buckets.get(user_key)
            if bucket is None:
                if len(self._buckets) >= self.max_buckets:
                    self._evict_lru_locked(now)

                initial_tokens = self.user_capacity
                if user_key in self._depleted:
                    dep_tokens, dep_time = self._depleted.pop(user_key)
                    elapsed = max(0.0, now - dep_time)
                    initial_tokens = min(self.user_capacity, dep_tokens + elapsed * self.user_rate)

                bucket = RateLimiter(rate=self.user_rate, capacity=self.user_capacity)
                with bucket._lock:
                    bucket._tokens = initial_tokens
                    bucket._last_update = now
                self._buckets[user_key] = bucket

            self._last_access[user_key] = now
            self._last_access.move_to_end(user_key)
            acquired = bucket.acquire(tokens)

            # Refund global limiter if user bucket denies
            if not acquired and self._global_limiter is not None:
                with self._global_limiter._lock:
                    self._global_limiter._tokens = min(
                        self._global_limiter.capacity,
                        self._global_limiter._tokens + tokens,
                    )

            return acquired

    def cleanup_expired(self, now: float | None = None) -> int:
        """Manually trigger eviction of expired user buckets."""
        with self._lock:
            current_time = time.monotonic() if now is None else now
            return self._evict_expired_locked(current_time)

    def active_user_count(self) -> int:
        """Return the number of currently tracked user buckets."""
        with self._lock:
            return len(self._buckets)

    def reset(self) -> None:
        """Reset all user buckets and global limiter state."""
        with self._lock:
            self._buckets.clear()
            self._last_access.clear()
            self._depleted.clear()
            if self._global_limiter is not None:
                with self._global_limiter._lock:
                    self._global_limiter._tokens = self._global_limiter.capacity
                    self._global_limiter._last_update = time.monotonic()


@dataclass
class SessionManager:
    """Detect and recover from stale Kubernetes sessions."""

    max_age: float = 300.0
    _timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def record_activity(self) -> None:
        """Record session activity."""
        self._timestamps.append(time.monotonic())

    def is_stale(self) -> bool:
        """Check if session appears stale."""
        if not self._timestamps:
            return False
        return time.monotonic() - self._timestamps[-1] > self.max_age
