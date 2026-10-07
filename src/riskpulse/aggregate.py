"""Per-entity sentiment aggregate: exponentially decayed, evidence-weighted and shrunk toward zero.

    s_i(t) = sum_j e_j * s_j * exp(-(t - t_j)/tau)  /  ( k0 + sum_j e_j * exp(-(t - t_j)/tau) )

  e_j  evidence of signal j = sentiment confidence x source credibility x (0.5 + impact/20)
                              / sqrt(mention_count)   (follow-ups of the same story add less)
  tau  from the configured half-life;  k0 shrinkage prior: one lone post cannot swing a name to +/-1
Sector signals count for every constituent at half weight; market-wide signals are excluded because
they move all names alike and carry no relative information.
"""
from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from datetime import datetime

import pandas as pd

from riskpulse import config
from riskpulse.schema import RiskSignal
from riskpulse.universe import sector_members, tickers

SECTOR_SPILLOVER = 0.5


def evidence(sentiment_conf: float, credibility: float, impact: float, mention_count: int = 1) -> float:
    return sentiment_conf * credibility * (0.5 + impact / 20) / math.sqrt(max(mention_count, 1))


class SentimentBook:
    def __init__(self, half_life_hours: float | None = None, min_confidence: float | None = None,
                 k0: float | None = None):
        c = config.get("module_a", {})
        hl = float(half_life_hours if half_life_hours is not None else c.get("half_life_hours", 12))
        self.tau = hl * 3600 / math.log(2)
        self.min_conf = float(min_confidence if min_confidence is not None else c.get("min_confidence", 0.4))
        self.k0 = float(k0 if k0 is not None else c.get("shrinkage_k0", 1.0))
        self.tickers = tickers()
        self.sectors = sector_members()
        self._num, self._den = defaultdict(float), defaultdict(float)
        self._n = defaultdict(int)
        self.recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self._t: datetime | None = None
        self._lock = threading.Lock()

    def _decay_to(self, t: datetime) -> None:
        if self._t is not None and t > self._t:
            f = math.exp(-(t - self._t).total_seconds() / self.tau)
            for k in list(self._num):
                self._num[k] *= f
                self._den[k] *= f
        if self._t is None or t > self._t:
            self._t = t

    def add(self, ticker: str, sentiment: float, e: float, ts: datetime, headline: str = "") -> None:
        """Add one piece of evidence (timestamps may arrive slightly out of order)."""
        with self._lock:
            self._decay_to(ts)
            f = math.exp(-max(0.0, (self._t - ts).total_seconds()) / self.tau)
            self._num[ticker] += e * f * sentiment
            self._den[ticker] += e * f
            self._n[ticker] += 1
            if headline:
                self.recent[ticker].append((ts.isoformat(), round(sentiment, 3), headline[:160]))

    def update(self, signals: list[RiskSignal]) -> int:
        used = 0
        for s in sorted(signals, key=lambda x: x.published_at):
            if s.sentiment_confidence < self.min_conf or s.entity.type == "market":
                continue
            e = evidence(s.sentiment_confidence, s.impact_drivers.features.get("credibility", 0.8), s.impact_score,
                         s.mention_count)
            targets = [s.entity.id] if s.entity.type == "company" else self.sectors.get(s.entity.id, [])
            for t in targets:
                self.add(t, s.sentiment_score, e if s.entity.type == "company" else e * SECTOR_SPILLOVER,
                         s.published_at, s.headline)
            used += 1
        return used

    def vector(self, now: datetime) -> list[float]:
        """Sentiment per ticker (universe order) - the fast path used by the backtest."""
        with self._lock:
            self._decay_to(now)
            return [self._num[t] / (self.k0 + self._den[t]) for t in self.tickers]

    def snapshot(self, now: datetime | None = None) -> pd.DataFrame:
        with self._lock:
            if now is not None:
                self._decay_to(now)
            return pd.DataFrame({"sentiment": [self._num[t] / (self.k0 + self._den[t]) for t in self.tickers],
                                 "evidence": [self._den[t] for t in self.tickers],
                                 "n_signals": [self._n[t] for t in self.tickers]},
                                index=pd.Index(self.tickers, name="ticker"))

    def entity(self, ticker: str, now: datetime | None = None) -> dict:
        snap = self.snapshot(now)
        row = snap.loc[ticker]
        return {"ticker": ticker, "ewma_sentiment": round(float(row.sentiment), 4), "evidence": round(float(row.evidence), 4),
                "n_signals": int(row.n_signals), "half_life_hours": round(self.tau * math.log(2) / 3600, 2),
                "recent": list(self.recent[ticker])[::-1]}
