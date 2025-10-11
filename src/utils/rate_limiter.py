# src/utils/rate_limiter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, MutableMapping, TypeVar, Coroutine
import time, threading, random, inspect, asyncio, functools

F = TypeVar("F")

class RateLimitExceeded(Exception):
    """Rzucane w trybie mode='raise' gdy przekroczono limit."""
    def __init__(self, retry_after: float):
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.3f}s")
        self.retry_after = float(max(0.0, retry_after))

@dataclass
class _BucketState:
    capacity: float          # maks. liczba tokenów (max_burst)
    tokens: float            # aktualna liczba tokenów
    fill_rate: float         # tokeny/sek (1/min_interval)
    last_refill: float       # znacznik czasu ostatniego refillu (monotonic)

class _TokenBucket:
    """Thread-safe token bucket. 1 wywołanie = 1 token."""
    def __init__(self, min_interval: float, max_burst: int = 1):
        if min_interval <= 0:
            # „bez limitu” – capacity duże, fill_rate bardzo wysoka
            min_interval = 0.0
        now = time.monotonic()
        capacity = max(1, int(max_burst))
        fill_rate = (1.0 / min_interval) if min_interval > 0 else float("inf")
        self._st = _BucketState(capacity=capacity, tokens=capacity, fill_rate=fill_rate, last_refill=now)
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        st = self._st
        if st.fill_rate == float("inf"):
            st.tokens = st.capacity
            st.last_refill = now
            return
        delta = max(0.0, now - st.last_refill)
        st.tokens = min(st.capacity, st.tokens + delta * st.fill_rate)
        st.last_refill = now

    def acquire_delay(self, now: float | None = None) -> float:
        """
        Zwraca 0 jeśli token dostępny (i konsumuje), albo >0 ile trzeba czekać.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            self._refill(now)
            if self._st.tokens >= 1.0:
                self._st.tokens -= 1.0
                return 0.0
            # ile czasu do pełnego jednego tokena?
            if self._st.fill_rate == float("inf"):
                return 0.0
            missing = 1.0 - self._st.tokens
            wait = missing / self._st.fill_rate
            # „zarezerwuj” przyszły token, aby równoległe wątki nie konsumowały go dwukrotnie
            self._st.tokens = max(0.0, self._st.tokens - 1.0)  # może zejść poniżej 0 -> dług
            return max(0.0, wait)

# Globalna mapa limiterów per klucz
_LIMITERS: MutableMapping[Any, _TokenBucket] = {}
_LIMITERS_LOCK = threading.Lock()

def _get_limiter(key: Any, min_interval: float, max_burst: int) -> _TokenBucket:
    k = key if key is not None else "__global__"
    with _LIMITERS_LOCK:
        lb = _LIMITERS.get(k)
        if lb is None:
            lb = _TokenBucket(min_interval=min_interval, max_burst=max_burst)
            _LIMITERS[k] = lb
        return lb

def rate_limited(
    min_interval: float,
    *,
    max_burst: int = 1,
    mode: str = "sleep",                      # 'sleep' | 'raise'
    jitter: float = 0.0,                      # dodatkowy losowy odstęp w sekundach (0..jitter)
    key_func: Optional[Callable[..., Any]] = None,  # np. lambda *a, **k: k.get("user_id")
) -> Callable[[F], F]:
    """
    Dekorator limitujący szybkość wywołań funkcji (token-bucket).
    - min_interval: minimalny odstęp między *średnimi* wywołaniami (1/min_interval ≈ RPS).
    - max_burst: ile wywołań można „zbuforować” bez czekania (zryw).
    - mode: 'sleep' (blokuje dozwolony czas) lub 'raise' (rzuca RateLimitExceeded).
    - jitter: dodaje losowy mikro-odstęp (0..jitter), aby rozproszyć równoczesne starty.
    - key_func: oddzielne kubełki per klucz (np. per użytkownik, per endpoint). Brak → globalny kubełek.

    Przykłady:
      @rate_limited(0.6)                          # min 0.6s między wywołaniami (jak Twoje)
      @rate_limited(0.2, max_burst=5)             # do 5 szybkich, potem ~1/0.2 = 5 RPS
      @rate_limited(1.0, mode="raise")            # nie śpij – rzuć RateLimitExceeded
      @rate_limited(0.1, key_func=lambda u, **k: k.get("user_id"))
    """
    if min_interval < 0:
        raise ValueError("min_interval must be >= 0")
    if max_burst <= 0:
        raise ValueError("max_burst must be >= 1")
    if mode not in {"sleep", "raise"}:
        raise ValueError("mode must be 'sleep' or 'raise'")

    def decorator(func: F) -> F:
        is_coro = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        async def aw(*args, **kwargs):  # type: ignore[override]
            key = key_func(*args, **kwargs) if key_func else None
            limiter = _get_limiter(key, min_interval, max_burst)
            delay = limiter.acquire_delay()
            if jitter > 0:
                delay += random.uniform(0.0, jitter)
            if delay > 0:
                if mode == "raise":
                    raise RateLimitExceeded(delay)
                await asyncio.sleep(delay)
            return await func(*args, **kwargs)  # type: ignore[misc]

        @functools.wraps(func)
        def sw(*args, **kwargs):  # type: ignore[override]
            key = key_func(*args, **kwargs) if key_func else None
            limiter = _get_limiter(key, min_interval, max_burst)
            delay = limiter.acquire_delay()
            if jitter > 0:
                delay += random.uniform(0.0, jitter)
            if delay > 0:
                if mode == "raise":
                    raise RateLimitExceeded(delay)
                time.sleep(delay)
            return func(*args, **kwargs)

        return aw if is_coro else sw  # type: ignore[return-value]

    return decorator
