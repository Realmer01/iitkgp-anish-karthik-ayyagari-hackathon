"""In-process publish/subscribe bus: the engine publishes, subscribers receive filtered batches."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from riskpulse.schema import RiskSignal

log = logging.getLogger("riskpulse.bus")
Handler = Callable[[list[RiskSignal]], None]
Filter = Callable[[RiskSignal], bool]


@dataclass
class Subscription:
    name: str
    handler: Handler
    filter: Filter | None = None
    delivered: int = 0
    failures: int = 0


class SignalBus:
    def __init__(self) -> None:
        self._subs: list[Subscription] = []
        self._lock = threading.Lock()
        self.published = 0

    def subscribe(self, name: str, handler: Handler, filter: Filter | None = None) -> Subscription:
        sub = Subscription(name, handler, filter)
        with self._lock:
            self._subs.append(sub)
        return sub

    def publish(self, signals: list[RiskSignal]) -> None:
        self.published += len(signals)
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            batch = [s for s in signals if sub.filter is None or sub.filter(s)]
            if not batch:
                continue
            try:
                sub.handler(batch)
                sub.delivered += len(batch)
            except Exception:  # a broken subscriber must never take the engine down
                sub.failures += 1
                log.exception("subscriber %s failed", sub.name)

    def status(self) -> list[dict]:
        return [{"subscriber": s.name, "delivered": s.delivered, "failures": s.failures} for s in self._subs]
