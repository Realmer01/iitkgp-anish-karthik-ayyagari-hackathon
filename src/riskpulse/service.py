"""RiskPulse service: wires sources -> engine -> sinks + bus -> Module A / Module B, on one clock.

Modes
  demo    replays precomputed full-tier signals for the demo window through the bus (instant start,
          fully offline); the engine still loads in the background for live "Analyze a headline" input
  replay  streams archived headlines (and real corpus posts) through the engine on the replay clock
  live    polls the live connectors (Yahoo RSS, Google News, GDELT, Mastodon, NewsAPI if a key is set)
"""
from __future__ import annotations

import gzip
import logging
import threading
import time
from datetime import datetime, timezone

import pandas as pd

from riskpulse import config
from riskpulse.aggregate import SentimentBook
from riskpulse.bus import InProcessSubscriber, SignalBus
from riskpulse.clock import Clock, ReplayClock
from riskpulse.config import OUTPUT, RAW
from riskpulse.module_a.live import LiveRebalancer
from riskpulse.module_b.monitor import StressMonitor
from riskpulse.schema import RiskSignal, utcnow
from riskpulse.store import Store

log = logging.getLogger("riskpulse.service")


class SignalReplay:
    """Releases precomputed signals as the replay clock passes their published_at."""

    name = "signal_replay"

    def __init__(self, path, clock: ReplayClock):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            self.signals = sorted((RiskSignal.model_validate_json(x) for x in fh if x.strip()),
                                  key=lambda s: s.published_at)
        self.clock, self.i = clock, 0

    @property
    def finished(self) -> bool:
        return self.i >= len(self.signals)

    def due(self) -> list[RiskSignal]:
        now, out = self.clock.now(), []
        while self.i < len(self.signals) and self.signals[self.i].published_at <= now:
            out.append(self.signals[self.i].model_copy(update={"generated_at": utcnow()}))
            self.i += 1
        return out

    def status(self) -> dict:
        return {"source": self.name, "released": self.i, "total": len(self.signals)}


class RiskPulseService:
    def __init__(self, mode: str = "demo", speed: float | None = None, start: str | None = None,
                 end: str | None = None, tier: str | None = None, reset: bool = True):
        self.mode, self.tier_requested = mode, tier
        self.window = (pd.Timestamp(start or config.get("demo.start")), pd.Timestamp(end or config.get("demo.end")))
        self.store = Store()
        if reset:
            self.store.reset()
        self.bus = SignalBus()
        if mode == "live":
            self.clock: Clock = Clock()
        else:
            self.clock = ReplayClock(self.window[0].to_pydatetime(),
                                     float(speed or config.get("demo.speed" if mode == "demo" else "replay.speed")))
        self.engine, self.engine_error = None, ""
        self.engine_ready = threading.Event()
        self._make_modules()
        self.sources, self.replay = [], None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.started_at = time.time()

    # ------------------------------------------------------------------ wiring
    def _make_modules(self) -> None:
        sub = InProcessSubscriber(self.bus)
        self.module_a = LiveRebalancer(sink=self.store, now=self.clock.now)
        self.module_b = StressMonitor(sink=self.store, now=self.clock.now)
        self.entities = SentimentBook()
        self.module_a.attach(sub)
        self.module_b.attach(sub)
        sub.subscribe("entity_sentiment", self.entities.update)

    def _make_sources(self) -> None:
        if self.mode == "demo":
            self.replay = SignalReplay(OUTPUT / "demo_signals.jsonl.gz", self.clock)
        elif self.mode == "replay":
            from riskpulse.ingestion import ReplaySource, SocialCorpusSource, load_documents
            docs = [d for d in load_documents(RAW / "news_history.csv.gz")
                    if self.window[0] <= pd.Timestamp(d.published_at) < self.window[1]]
            self.sources = [ReplaySource(docs, self.clock), SocialCorpusSource(per_poll=1, now=self.clock.now)]
        else:
            from riskpulse.ingestion import GdeltSource, GoogleNewsSource, MastodonSource, NewsAPISource, YahooRSSSource
            self.sources = [YahooRSSSource(), GoogleNewsSource(), GdeltSource(), MastodonSource(), NewsAPISource()]

    def _load_engine(self) -> None:
        try:
            from riskpulse.engine import RiskEngine
            self.engine = RiskEngine(self.tier_requested)
        except Exception as exc:  # pragma: no cover - keeps demo mode alive without models
            self.engine_error = str(exc)
            log.exception("engine failed to load")
        finally:
            self.engine_ready.set()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._make_sources()
        for target, name in ((self._load_engine, "engine-loader"), (self._loop, "pipeline")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()

    def publish(self, signals: list[RiskSignal]) -> None:
        if signals:
            self.store.add_signals(signals)
            self.bus.publish(signals)

    def _loop(self) -> None:
        last_poll: dict[str, float] = {}
        last_heartbeat = time.monotonic()
        while not self._stop.is_set():
            try:
                if self.replay is not None:
                    self.publish(self.replay.due())
                elif self.engine_ready.is_set() and self.engine is not None:
                    docs = []
                    for src in self.sources:
                        if time.monotonic() - last_poll.get(src.name, 0) >= (src.poll_seconds if self.mode == "live" else 0.5):
                            last_poll[src.name] = time.monotonic()
                            docs += src.poll()
                    for i in range(0, len(docs), 64):
                        self.publish(self.engine.process(docs[i:i + 64]))
                if time.monotonic() - last_heartbeat > 2.0:     # decay sentiment between signal batches
                    self.module_a.step()
                    last_heartbeat = time.monotonic()
            except Exception:
                log.exception("pipeline iteration failed")
            self._stop.wait(0.25)

    def restart_replay(self) -> None:
        """Rewind the demo / replay to the start of its window with fresh module state."""
        if not isinstance(self.clock, ReplayClock):
            return
        self.store.reset()
        self.bus = SignalBus()
        self.clock.jump(self.window[0].to_pydatetime())
        self._make_modules()
        self._make_sources()

    # ------------------------------------------------------------------ interactive analysis
    def analyze(self, text: str, source_type: str = "news", inject: bool = False, publisher: str = "analyst input") -> dict:
        if not self.engine_ready.wait(timeout=120) or self.engine is None:
            raise RuntimeError(f"engine unavailable: {self.engine_error or 'still loading'}")
        t0 = time.perf_counter()
        signals = self.engine.analyze_text(text, source_type, publisher)
        runs_before = len(self.module_b.runs)
        if inject and signals:
            now = self.clock.now()
            signals = [s.model_copy(update={"published_at": now, "generated_at": utcnow()}) for s in signals]
            self.publish(signals)
            self.module_a.step()
        return {"signals": [s.model_dump(mode="json") for s in signals], "injected": inject,
                "stress_runs": self.module_b.runs[: len(self.module_b.runs) - runs_before] if inject else [],
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def status(self) -> dict:
        srcs = [s.status() for s in self.sources] + ([self.replay.status()] if self.replay else [])
        return {"mode": self.mode, "clock": self.clock.status(), "uptime_s": round(time.time() - self.started_at, 1),
                "engine": self.engine.status() if self.engine else {"loading": not self.engine_ready.is_set(),
                                                                    "error": self.engine_error},
                "sources": srcs, "bus": self.bus.status(), "signals_stored": self.store.signal_count(),
                "replay_finished": bool(self.replay and self.replay.finished),
                "window": [str(self.window[0]), str(self.window[1])], "server_time": datetime.now(timezone.utc).isoformat()}
