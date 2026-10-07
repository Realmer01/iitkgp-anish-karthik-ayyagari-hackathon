"""Module A - live tactical index rebalancer. Subscribes to the Sentiment Score.

Every signal batch updates the decayed sentiment book; the index is then re-optimised (throttled to
`min_interval_seconds` of wall time) and every weight snapshot is persisted so the dashboard can chart
weights over time. Each weight change is logged with the headline that drove that name's sentiment.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd

from riskpulse import config
from riskpulse.aggregate import SentimentBook
from riskpulse.bus.subscriber import SignalSubscriber
from riskpulse.module_a.policy import PolicyParams, base_weights, rebalance, target_weights
from riskpulse.schema import RiskSignal


class LiveRebalancer:
    name = "module_a_rebalancer"

    def __init__(self, sink=None, now: Callable[[], datetime] | None = None):
        self.sink, self.now = sink, now or (lambda: datetime.now(UTC))
        self.params = PolicyParams.from_config()
        self.base = base_weights(self.params)
        self.current = self.base.copy()
        self.book = SentimentBook()
        self.min_interval = float(config.get("module_a.min_interval_seconds", 5))
        self.rebalances, self.cum_turnover, self.cum_cost = 0, 0.0, 0.0
        self._last = 0.0
        self._lock = threading.Lock()
        self._snapshot(self.now(), pd.Series(0.0, index=self.base.index))

    def attach(self, subscriber: SignalSubscriber) -> None:
        subscriber.subscribe(self.name, self.on_signals, self.wants)

    @staticmethod
    def wants(s: RiskSignal) -> bool:
        return s.entity.type in ("company", "sector")

    def on_signals(self, signals: list[RiskSignal]) -> None:
        if self.book.update(signals) and time.monotonic() - self._last >= self.min_interval:
            self.step()

    def step(self, now: datetime | None = None) -> float:
        with self._lock:
            now = now or self.now()
            snap = self.book.snapshot(now)
            target = target_weights(self.base, snap["sentiment"], self.params)
            new, turnover = rebalance(self.current, target, self.base, self.params)
            self._last = time.monotonic()
            if turnover > 0:
                log_rows = []
                for t in new.index[(new - self.current).abs() > 1e-4]:
                    recent = self.book.recent[t]
                    reason = recent[-1][2] if recent else "sentiment decay toward neutral"
                    log_rows.append((now.isoformat(), t, round(float(self.current[t]), 6), round(float(new[t]), 6),
                                     round(float(snap.loc[t, "sentiment"]), 4), reason))
                self.current = new
                self.rebalances += 1
                self.cum_turnover += turnover
                self.cum_cost += turnover * self.params.cost_bps / 1e4
                if self.sink is not None:
                    self.sink.add_rebalance_log(log_rows)
            self._snapshot(now, snap["sentiment"])
            return turnover

    def _snapshot(self, now: datetime, sentiment: pd.Series) -> None:
        if self.sink is not None:
            self.sink.add_weights(now.replace(microsecond=0).isoformat(),
                                  [(t, round(float(self.current[t]), 6), round(float(self.base[t]), 6),
                                    round(float(sentiment.get(t, 0.0)), 4)) for t in self.base.index])

    def state(self) -> dict:
        snap = self.book.snapshot(self.now())
        return {"rebalances": self.rebalances, "cumulative_turnover": round(self.cum_turnover, 4),
                "cumulative_cost_bps": round(self.cum_cost * 1e4, 2), "params": self.params.__dict__,
                "weights": [{"ticker": t, "weight": round(float(self.current[t]), 5),
                             "base_weight": round(float(self.base[t]), 5),
                             "active_weight": round(float(self.current[t] - self.base[t]), 5),
                             "sentiment": round(float(snap.loc[t, "sentiment"]), 4),
                             "n_signals": int(snap.loc[t, "n_signals"])} for t in self.base.index]}
