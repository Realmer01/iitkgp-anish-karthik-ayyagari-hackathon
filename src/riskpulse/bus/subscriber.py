"""SignalSubscriber - the only way downstream modules receive signals.

Modules depend on the RiskSignal schema and this interface, never on engine internals, so a module
can run in the engine's process (InProcessSubscriber), in another process on the same machine
(SQLiteTailSubscriber), or on another machine (SSESubscriber over the API) without code changes -
the same producer/consumer decoupling a production risk platform gets from Kafka.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from riskpulse.bus.inprocess import Filter, Handler, SignalBus
from riskpulse.schema import RiskSignal

log = logging.getLogger("riskpulse.subscriber")


class SignalSubscriber(ABC):
    @abstractmethod
    def subscribe(self, name: str, handler: Handler, filter: Filter | None = None) -> None:
        ...

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class InProcessSubscriber(SignalSubscriber):
    def __init__(self, bus: SignalBus):
        self.bus = bus

    def subscribe(self, name: str, handler: Handler, filter: Filter | None = None) -> None:
        self.bus.subscribe(name, handler, filter)


class _PollingSubscriber(SignalSubscriber):
    """Runs one background thread per subscription that hands decoded batches to the handler."""

    def __init__(self, poll_seconds: float = 1.0):
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def subscribe(self, name: str, handler: Handler, filter: Filter | None = None) -> None:
        def run() -> None:
            for batch in self._batches():
                if self._stop.is_set():
                    return
                batch = [s for s in batch if filter is None or filter(s)]
                if batch:
                    try:
                        handler(batch)
                    except Exception:
                        log.exception("subscriber %s failed", name)

        t = threading.Thread(target=run, name=f"sub-{name}", daemon=True)
        t.start()
        self._threads.append(t)

    @abstractmethod
    def _batches(self):
        ...

    def close(self) -> None:
        self._stop.set()


class SQLiteTailSubscriber(_PollingSubscriber):
    """Tails the engine's SQLite sink (WAL mode allows concurrent readers)."""

    def __init__(self, db_path: Path, from_start: bool = False, poll_seconds: float = 1.0):
        super().__init__(poll_seconds)
        self.db_path, self.from_start = Path(db_path), from_start

    def _batches(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        last = 0 if self.from_start else conn.execute("SELECT COALESCE(MAX(id), 0) FROM signals").fetchone()[0]
        while not self._stop.is_set():
            rows = conn.execute("SELECT id, payload FROM signals WHERE id > ? ORDER BY id LIMIT 500", (last,)).fetchall()
            if rows:
                last = rows[-1][0]
                yield [RiskSignal.model_validate_json(p) for _, p in rows]
            else:
                time.sleep(self.poll_seconds)


class SSESubscriber(_PollingSubscriber):
    """Consumes the API's Server-Sent Events stream (GET /stream)."""

    def __init__(self, api_url: str, poll_seconds: float = 1.0):
        super().__init__(poll_seconds)
        self.url = api_url.rstrip("/") + "/stream"

    def _batches(self):
        import httpx

        while not self._stop.is_set():
            try:
                with httpx.stream("GET", self.url, timeout=None) as r:
                    for line in r.iter_lines():
                        if self._stop.is_set():
                            return
                        if line.startswith("data:"):
                            yield [RiskSignal.model_validate(json.loads(line[5:]))]
            except httpx.HTTPError as exc:
                log.warning("SSE connection lost (%s) - reconnecting", exc)
                time.sleep(self.poll_seconds * 3)
