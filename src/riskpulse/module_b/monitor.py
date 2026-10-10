"""Module B - event-driven stress-test trigger. Subscribes to Event Classification + Impact Score.

A signal triggers a stress test when
  * its event type is stress-eligible (config module_b.eligible_events),
  * impact_score > the threshold for that event type (default 7, per-type overrides), and
  * event_confidence >= min_event_confidence,
and it reports an adverse event:
  * its tone is not positive (sentiment_score <= max_sentiment): a de-escalation is not a stress, and
  * the headline reports an event rather than discussing one (nlp.framing: questions, explainers,
    listicles and hedged speculation do not trigger; a manual what-if still can).
One real-world event must cause one stress run, so a trigger is suppressed when
  * its story already triggered a run (story dedup), or
  * the same scenario (and, for company overlays, the same company) ran within cooldown_minutes of
    news time (the signals' published_at, so replay speed and batching cannot change the result) -
    unless the new signal ESCALATES, i.e. its impact exceeds the last run's by at least
    escalation_delta. A desk wants one run per developing event, plus a re-run when it worsens.
Every run is logged with its triggering signal, the scenario applied, results and the
headline-to-result latency.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from riskpulse import config
from riskpulse.bus.subscriber import SignalSubscriber
from riskpulse.module_b import scenarios, stress
from riskpulse.nlp.framing import factuality
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
        self.cooldown = timedelta(minutes=float(c.get("cooldown_minutes", 360)))
        self.escalation = float(c.get("escalation_delta", 1.0))
        self.max_sentiment = float(c.get("max_sentiment", 0.0))
        self.min_factuality = float(c.get("min_factuality", 1.0))
        self.sink, self.now = sink, now or (lambda: datetime.now(UTC))
        self.runs: list[dict] = []
        self.suppressed = {"story_dedup": 0, "cooldown": 0, "commentary": 0, "not_adverse": 0}
        self.last_skip = ""
        self._stories: set[str] = set()
        self._last: dict[tuple[str, str], tuple[datetime, float]] = {}
        self._lock = threading.Lock()

    def attach(self, subscriber: SignalSubscriber) -> None:
        subscriber.subscribe(self.name, self.on_signals, self.wants)

    def threshold(self, event_type: str) -> float:
        return self.thresholds.get(event_type, self.default_threshold)

    def wants(self, s: RiskSignal) -> bool:
        return (s.event_type in self.eligible and s.impact_score > self.threshold(s.event_type)
                and s.event_confidence >= self.min_conf)

    def screen(self, s: RiskSignal) -> tuple[str, str]:
        """('', '') if a wanted signal reports an adverse event, else (suppression key, reason)."""
        if s.sentiment_score > self.max_sentiment:
            return "not_adverse", f"positive tone ({s.sentiment_score:+.2f}) - stress tests run on adverse events"
        fact, label = factuality(s.headline)
        if fact < self.min_factuality:
            return "commentary", f"commentary ({label}) - automatic runs need a reported event; try the what-if panel"
        return "", ""

    def explain(self, s: RiskSignal) -> str:
        """Why a signal did not (or would not) trigger a run - no side effects."""
        if s.event_type not in self.eligible:
            return f"{s.event_type} is not a stress-eligible event type"
        if s.impact_score <= self.threshold(s.event_type):
            return f"impact {s.impact_score:.1f} is not above the {self.threshold(s.event_type):.1f} threshold"
        if s.event_confidence < self.min_conf:
            return f"event confidence {s.event_confidence:.0%} is below {self.min_conf:.0%}"
        return self.screen(s)[1] or self.last_skip or "eligible"

    def on_signals(self, signals: list[RiskSignal]) -> None:
        for s in sorted(signals, key=lambda x: -x.impact_score):
            self.consider(s)

    def consider(self, s: RiskSignal) -> dict | None:
        with self._lock:
            now, news_time = self.now(), s.published_at
            rule = scenarios.route(s.event_type, s.headline)
            target = s.entity.id if s.entity.type == "company" and rule.get("overlay") else ""
            key = (rule["scenario"], target)
            why, self.last_skip = self.screen(s)
            if why:
                self.suppressed[why] += 1
                return None
            if s.story_id in self._stories:
                self.suppressed["story_dedup"] += 1
                self.last_skip = "this story already triggered a run"
                return None
            last = self._last.get(key)
            if last and news_time - last[0] < self.cooldown and s.impact_score < last[1] + self.escalation:
                self.suppressed["cooldown"] += 1
                hours = (news_time - last[0]).total_seconds() / 3600
                self.last_skip = (f"the same scenario ran {hours:.1f} h ago and impact did not rise by "
                                  f"{self.escalation:.0f}+ (cooldown)")
                return None
            self._stories.add(s.story_id)
            self._last[key] = (news_time, s.impact_score)
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
                "min_event_confidence": self.min_conf, "cooldown_minutes": self.cooldown.total_seconds() / 60,
                "escalation_delta": self.escalation, "max_sentiment": self.max_sentiment,
                "min_factuality": self.min_factuality}
