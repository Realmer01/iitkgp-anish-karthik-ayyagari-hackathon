"""Replay of recorded, timestamped documents against a ReplayClock.

Documents keep their ORIGINAL published_at; the source releases each one once the simulated
clock passes that time. With speed 3600 a week of history streams in under three minutes, through
exactly the same engine code as live mode - reproducible and immune to network failures.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import pandas as pd

from riskpulse.clock import ReplayClock
from riskpulse.ingestion.base import Source
from riskpulse.schema import RawDocument, SourceType, as_utc


def load_documents(path: Path) -> list[RawDocument]:
    """Load a JSONL of RawDocuments or a headline CSV (published_at, publisher, title[, source])."""
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            docs = [RawDocument(**json.loads(line)) for line in fh if line.strip()]
    else:
        df = pd.read_csv(path)
        docs = [RawDocument(source=str(r.get("source", "google_news")), source_type=SourceType.NEWS,
                            publisher=str(r.get("publisher", "")), title=str(r["title"]),
                            published_at=pd.Timestamp(r["published_at"]).to_pydatetime())
                for r in df.to_dict("records")]
    return sorted(docs, key=lambda d: d.published_at)


class ReplaySource(Source):
    name = "replay"
    poll_seconds = 0.5

    def __init__(self, docs: list[RawDocument], clock: ReplayClock, batch_max: int = 64):
        super().__init__()
        self.docs = docs
        self._times = [as_utc(d.published_at) for d in docs]
        self.clock, self.batch_max = clock, batch_max
        self._i = bisect.bisect_left(self._times, clock.now())

    @property
    def finished(self) -> bool:
        return self._i >= len(self.docs)

    @property
    def progress(self) -> float:
        return self._i / max(1, len(self.docs))

    def fetch(self) -> list[RawDocument]:
        hi = min(bisect.bisect_right(self._times, self.clock.now()), self._i + self.batch_max)
        out, self._i = self.docs[self._i:hi], max(self._i, hi)
        return out

    def _new(self, docs: list[RawDocument]) -> list[RawDocument]:
        self.emitted += len(docs)
        return docs

    def status(self) -> dict:
        return {**super().status(), "progress": round(self.progress, 4), "remaining": len(self.docs) - self._i}
