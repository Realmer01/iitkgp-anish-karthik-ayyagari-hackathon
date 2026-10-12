"""Source connector interface.

Connectors are pull-based: `poll()` returns documents that are new since the last call, so a
scheduler can poll each feed at its own rate limit. Network calls go through `get()`, which adds
timeouts, retries with exponential backoff and rate-limit handling. A failing feed never raises
into the engine - it is logged and counted.
"""
from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

import requests

from riskpulse import config
from riskpulse.schema import RawDocument, SourceType

log = logging.getLogger("riskpulse.ingestion")
USER_AGENT = "Mozilla/5.0 (RiskPulse research prototype)"

# Nothing published by S&P Global or Crisil (or their businesses) is ingested, so no content of either firm
# enters the engine. Third-party news that merely mentions them, e.g. a reported public rating action, is kept.
EXCLUDED_PUBLISHERS = re.compile(r"s&p global|spglobal|standard (?:&|and) poor|s&p dow jones|platts|ihs markit|"
                                 r"kensho|crisil|coalition greenwich", re.I)


def excluded_publisher(publisher: str | None) -> bool:
    return bool(EXCLUDED_PUBLISHERS.search(publisher or ""))


class RateLimited(RuntimeError):
    pass


class Source(ABC):
    name: str = "source"
    source_type: SourceType = SourceType.NEWS
    poll_seconds: float = float(config.get("ingestion.poll_seconds", 120))

    def __init__(self) -> None:
        self._seen: OrderedDict[str, None] = OrderedDict()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.timeout = float(config.get("ingestion.request_timeout_seconds", 30))
        self.retries = int(config.get("ingestion.max_retries", 3))
        self.errors = 0
        self.emitted = 0
        self.last_error = ""

    def get(self, url: str, **kwargs) -> requests.Response:
        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                r = self.session.get(url, timeout=self.timeout, **kwargs)
                if r.status_code == 429 or "limit requests" in r.text[:200].lower():
                    raise RateLimited(f"{self.name} rate limited")
                r.raise_for_status()
                return r
            except (requests.RequestException, RateLimited) as exc:
                if isinstance(exc, RateLimited) or attempt == self.retries:
                    raise            # a rate limit is not retried: the next scheduled poll tries again
                log.info("%s: %s - retry %d in %.0fs", self.name, exc, attempt + 1, delay)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def _new(self, docs: list[RawDocument]) -> list[RawDocument]:
        fresh = [d for d in docs if d.id not in self._seen and not excluded_publisher(d.publisher)]
        for d in fresh:
            self._seen[d.id] = None
        while len(self._seen) > 50_000:
            self._seen.popitem(last=False)
        self.emitted += len(fresh)
        return fresh

    def poll(self) -> list[RawDocument]:
        try:
            return self._new(self.fetch())
        except Exception as exc:
            self.errors += 1
            self.last_error = str(exc)[:200]
            log.warning("%s fetch failed: %s", self.name, exc)
            return []

    @abstractmethod
    def fetch(self) -> list[RawDocument]:
        ...

    def status(self) -> dict:
        return {"source": self.name, "type": self.source_type.value, "emitted": self.emitted,
                "errors": self.errors, "last_error": self.last_error}
