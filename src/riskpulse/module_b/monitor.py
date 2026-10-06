"""Module B - event-driven stress-test trigger. Subscribes to Event Classification + Impact Score.

A signal triggers a stress test when
  * its event type is stress-eligible (config module_b.eligible_events),
  * impact_score > the threshold for that event type (default 7, per-type overrides), and
  * event_confidence >= min_event_confidence.
One real-world event must cause one stress run, so a trigger is suppressed when
  * its story already triggered a run (story dedup), or
  * the same scenario + target ran within cooldown_minutes of simulated time.
Every run is logged with its triggering signal, the scenario applied, results and the
headline-to-result latency.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from riskpulse import config
from riskpulse.bus.subscriber import SignalSubscriber
from riskpulse.module_b import scenarios, stress
from riskpulse.schema import RiskSignal

log = logging.getLogger("riskpulse.module_b")


class StressMonitor:
    name = "module_b_stress"

    def __init__(self, sink=None, now: Callable[[], datetime] | None = None):
        c = config.get("module_b", {})
        self.default_threshold = float(c.get("default_threshold", 7.0))
        self.thresholds = {k: float(v) for k, v in (c.get("thresholds") or {}).items()}
        self.eligible = set(c.get("eligible_events", []))
        self.min_conf = float(c.get("min_event_confidence", 0.45))
        self.cooldown = timedelta(minutes=float(c.get("cooldown_minutes", 60)))
        self.sink, self.now = sink, now or (lambda: datetime.now(timezone.utc))
        self.runs: list[dict] = []
        self.suppressed = {"story_dedup": 0, "cooldown": 0}
        self._stories: set[str] = set()
        self._last: dict[tuple[str, str], datetime] = {}
        self._lock = threading.Lock()

    def attach(self, subscriber: SignalSubscriber) -> None:
        subscriber.subscribe(self.name, self.on_signals, self.wants)

    def threshold(self, event_type: str) -> float:
        return self.thresholds.get(event_type, self.default_threshold)

    def wants(self, s: RiskSignal) -> bool:
        return (s.event_type in self.eligible and s.impact_score > self.threshold(s.event_type)
                and s.event_confidence >= self.min_conf)

    def on_signals(self, signals: list[RiskSignal]) -> None:
        for s in sorted(signals, key=lambda x: -x.impact_score):
            self.consider(s)

    def consider(self, s: RiskSignal) -> dict | None:
        with self._lock:
            now = self.now()
            rule = scenarios.route(s.event_type, s.headline)
            target = s.entity.id if s.entity.type != "market" else ""
            key = (rule["scenario"], target)
            if s.story_id in self._stories:
                self.suppressed["story_dedup"] += 1
                return None
            if key in self._last and now - self._last[key] < self.cooldown:
                self.suppressed["cooldown"] += 1
                return None
            self._stories.add(s.story_id)
            self._last[key] = now
        shocks = scenarios.build(s.event_type, s.impact_score, s.headline, s.entity.type, s.entity.ticker or "",
                                 s.entity.sector or "", s.regions)
        trigger = {"type": "signal", "signal_id": s.signal_id, "story_id": s.story_id, "headline": s.headline,
                   "event_type": s.event_type, "impact_score": s.impact_score, "event_confidence": s.event_confidence,
                   "sentiment_score": s.sentiment_score, "entity": s.entity.id, "regions": s.regions,
                   "publisher": s.publisher, "published_at": s.published_at.isoformat(),
                   "threshold": self.threshold(s.event_type)}
        result = stress.run(shocks, trigger=trigger, now=now)
        lag = time.time() - s.generated_at.timestamp()
        result["latency_seconds"] = round(max(lag, 0.0), 3)   # engine output -> stress result, wall clock
        with self._lock:
            self.runs.insert(0, result)
            del self.runs[100:]
        if self.sink is not None:
            self.sink.add_stress_run(result)
        log.info("stress test %s: %s on '%s' -> P&L %.1fm", result["run_id"], shocks.name, s.headline[:60],
                 result["pnl"] / 1e6)
        return result

    def state(self) -> dict:
        return {"runs": len(self.runs), "suppressed": dict(self.suppressed), "eligible_events": sorted(self.eligible),
                "default_threshold": self.default_threshold, "thresholds": self.thresholds,
                "min_event_confidence": self.min_conf, "cooldown_minutes": self.cooldown.total_seconds() / 60}
