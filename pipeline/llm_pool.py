#!/usr/bin/env python3
"""
llm_pool.py — robust concurrent LLM-call pool for long unattended runs
=====================================================================

Key rotation + adaptive worker reduction + token-bucket pacing + (near-)unlimited
retries, so a multi-hour job (e.g. 5_judgement_day.py judging on a VM) survives
provider rate limits (429) and network blips without dying. Provider-agnostic via
litellm. This is the same proven pattern used by sft_v3_generator.py for teacher
generation, packaged for reuse.

Usage:
    pool = RobustPool(api_keys=[...], workers=4, rpm_per_key=38)
    text = pool.complete(messages, model="nvidia_nim/minimaxai/minimax-m3", max_tokens=512)
The pool's AdaptiveSemaphore gates real concurrency below the ThreadPoolExecutor's
max_workers, so submit jobs to a pool of size `workers` and let `complete()` throttle.
"""

from __future__ import annotations

import random
import threading
import time
from datetime import datetime
from typing import List, Optional


def _tag() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Adaptive concurrency — halves workers on sustained rate limits, recovers slowly
# ---------------------------------------------------------------------------

class AdaptiveSemaphore:
    def __init__(self, initial: int, min_val: int = 1, scale_up_interval: float = 300.0):
        self._min = min_val
        self._max = initial
        self._current = initial
        self._sem = threading.Semaphore(initial)
        self._lock = threading.Lock()
        self._rl_events: List[float] = []
        self._rl_window = 60.0
        self._rl_threshold = 3
        self._scale_up_interval = scale_up_interval
        self._last_scale_up = time.monotonic()

    def acquire(self) -> None:
        self._sem.acquire()

    def release(self) -> None:
        self._sem.release()

    def on_rate_limit(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._rl_events.append(now)
            self._rl_events = [t for t in self._rl_events if now - t < self._rl_window]
            if len(self._rl_events) >= self._rl_threshold and self._current > self._min:
                target = max(self._min, self._current // 2)
                stolen = 0
                for _ in range(self._current - target):
                    if self._sem.acquire(blocking=False):
                        stolen += 1
                    else:
                        break
                if stolen:
                    self._current -= stolen
                    self._rl_events.clear()
                    print(f"  {_tag()} [adaptive] rate limited — workers down to {self._current}", flush=True)

    def try_scale_up(self) -> None:
        now = time.monotonic()
        with self._lock:
            recent = [t for t in self._rl_events if now - t < self._rl_window]
            if (self._current < self._max and not recent
                    and now - self._last_scale_up >= self._scale_up_interval):
                self._sem.release()
                self._current += 1
                self._last_scale_up = now
                print(f"  {_tag()} [adaptive] quiet — workers up to {self._current}/{self._max}", flush=True)

    @property
    def current(self) -> int:
        return self._current


# ---------------------------------------------------------------------------
# Token bucket — paces all threads to <= N calls/min
# ---------------------------------------------------------------------------

class TokenBucket:
    def __init__(self, rate_per_minute: float):
        self._rate = rate_per_minute / 60.0
        self._max = max(1.0, rate_per_minute / 10)
        self._tokens = self._max
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self._max, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Key rotation — cycles keys after N consecutive rate limits
# ---------------------------------------------------------------------------

class KeyRotator:
    def __init__(self, keys: List[str], threshold: int = 3):
        self._keys = [k for k in (keys or []) if k]
        self._threshold = threshold
        self._idx = 0
        self._hits = 0
        self._lock = threading.Lock()

    @property
    def current_key(self) -> Optional[str]:
        return self._keys[self._idx] if self._keys else None

    @property
    def n_keys(self) -> int:
        return len(self._keys)

    def reset_hits(self) -> None:
        self._hits = 0

    def on_rate_limit(self) -> None:
        if len(self._keys) <= 1:
            return
        with self._lock:
            self._hits += 1
            if self._hits >= self._threshold:
                old = self._idx + 1
                self._idx = (self._idx + 1) % len(self._keys)
                self._hits = 0
                print(f"  {_tag()} [key_rotator] key {old}/{len(self._keys)} exhausted "
                      f"— rotating to key {self._idx + 1}/{len(self._keys)}", flush=True)


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------

_TRANSIENT = ("ratelimit", "429", "too many", "timeout", "timed out", "apiconnection",
              "connection", "overload", "503", "502", "500", "unavailable", "service",
              "internalserver", "remoteprotocol", "readtimeout")
_FATAL = ("authentication", "invalid api key", "permissiondenied", "permission denied",
          "model_not_found", "not found", "404", "badrequest", "invalid_request")


class RobustPool:
    """Concurrent LLM caller that keeps going through rate limits.

    `complete()` retries transient failures effectively forever (capped exponential
    backoff), rotates API keys, and shrinks worker count under sustained 429s. Fatal
    errors (auth, 404, bad request) raise immediately so a misconfiguration fails fast
    rather than looping forever."""

    def __init__(self, api_keys: Optional[List[str]] = None, workers: int = 4,
                 min_workers: int = 1, rpm_per_key: float = 38.0,
                 max_attempts: int = 10_000, verbose: bool = True):
        self.keys = KeyRotator(api_keys or [])
        self.sem = AdaptiveSemaphore(workers, min_workers)
        n = max(1, self.keys.n_keys)
        self.bucket = TokenBucket(rpm_per_key * n)
        self.max_workers = workers
        self.max_attempts = max_attempts
        self.verbose = verbose

    def complete(self, messages: list, model: str, max_tokens: int,
                 timeout: int = 120, temperature: float = 0.1, **kw) -> str:
        import litellm
        attempt = 0
        while True:
            self.bucket.acquire()
            self.sem.acquire()
            wait = 0.0
            try:
                kwargs = dict(model=model, messages=messages, max_tokens=max_tokens,
                              timeout=timeout, temperature=temperature, **kw)
                if self.keys.current_key:
                    kwargs["api_key"] = self.keys.current_key
                resp = litellm.completion(**kwargs)
                self.keys.reset_hits()
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 — classify then retry/raise
                blob = (e.__class__.__name__ + " " + str(e)).lower()
                is_rl = ("ratelimit" in blob) or ("429" in blob) or ("too many" in blob)
                transient = any(k in blob for k in _TRANSIENT)
                fatal = any(k in blob for k in _FATAL) and not transient
                if is_rl:
                    self.keys.on_rate_limit()
                    self.sem.on_rate_limit()
                attempt += 1
                if fatal or attempt >= self.max_attempts:
                    raise
                wait = min(60.0, 2 ** min(attempt, 6)) + random.uniform(0, 2)
                if self.verbose:
                    print(f"  {_tag()} [retry {attempt}] {e.__class__.__name__} — {wait:.0f}s", flush=True)
            finally:
                self.sem.release()
            if wait:
                time.sleep(wait)
            self.sem.try_scale_up()


def keys_from_env() -> List[str]:
    """Collect NVIDIA NIM keys from the environment: NVIDIA_NIM_API_KEYS (comma/space
    separated) takes precedence, else the single NVIDIA_NIM_API_KEY."""
    import os
    raw = os.environ.get("NVIDIA_NIM_API_KEYS", "")
    keys = [k.strip() for k in raw.replace(",", " ").split() if k.strip()]
    if not keys:
        single = os.environ.get("NVIDIA_NIM_API_KEY", "").strip()
        if single:
            keys = [single]
    return keys
