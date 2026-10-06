"""Clocks: wall-clock time for live mode, an accelerated simulated clock for replay.

ReplayClock maps wall time to simulated time:  sim_now = sim_start + elapsed_wall * speed
(speed 3600 = one simulated hour per second). It can be paused, resumed and re-speeded from the
dashboard; everything time-dependent (sentiment decay, cooldowns, rebalance stamps) reads `now()`.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone


class Clock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def status(self) -> dict:
        return {"mode": "live", "now": self.now().isoformat()}


class ReplayClock(Clock):
    def __init__(self, start: datetime, speed: float = 3600.0):
        self._sim0 = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        self._wall0 = time.monotonic()
        self._speed = float(speed)
        self._paused_at: float | None = None
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            wall = self._paused_at if self._paused_at is not None else time.monotonic()
            return self._sim0 + timedelta(seconds=(wall - self._wall0) * self._speed)

    def _rebase(self) -> None:  # caller holds the lock
        wall = self._paused_at if self._paused_at is not None else time.monotonic()
        self._sim0 += timedelta(seconds=(wall - self._wall0) * self._speed)
        self._wall0 = wall

    def set_speed(self, speed: float) -> None:
        with self._lock:
            self._rebase()
            self._speed = max(0.0, float(speed))

    def pause(self) -> None:
        with self._lock:
            if self._paused_at is None:
                self._paused_at = time.monotonic()

    def resume(self) -> None:
        with self._lock:
            if self._paused_at is not None:
                self._wall0 += time.monotonic() - self._paused_at
                self._paused_at = None

    def jump(self, to: datetime) -> None:
        with self._lock:
            self._sim0, self._wall0 = to, self._paused_at if self._paused_at is not None else time.monotonic()

    @property
    def paused(self) -> bool:
        return self._paused_at is not None

    @property
    def speed(self) -> float:
        return self._speed

    def status(self) -> dict:
        return {"mode": "replay", "now": self.now().isoformat(), "speed": self._speed, "paused": self.paused}
