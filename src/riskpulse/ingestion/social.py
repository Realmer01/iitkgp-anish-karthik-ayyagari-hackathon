"""Social-media connectors.

* MastodonSource     - live public hashtag timelines, no key. (X/Twitter's API is paid-only; Reddit,
                       StockTwits and Bluesky refused anonymous clients when checked on 2026-10-06.)
* SocialCorpusSource - replays REAL finance tweets from the public zeroshot/twitter-financial-news
                       corpora (MIT). The corpus has no timestamps, so posts are stamped with the
                       replay clock's current time - documented in data/README.md, and excluded from
                       the backtest so no synthetic timing leaks into results.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from datetime import datetime, timezone

import pandas as pd

from riskpulse import config
from riskpulse.config import RAW
from riskpulse.ingestion.base import Source
from riskpulse.ingestion.news import parse_date
from riskpulse.nlp.preprocess import clean
from riskpulse.schema import RawDocument, SourceType

COMPANY_TAGS = ["apple", "microsoft", "nvidia", "google", "tesla", "amazon", "jpmorgan", "boeing", "pfizer"]


class MastodonSource(Source):
    name = "mastodon"
    source_type = SourceType.SOCIAL

    def __init__(self, instance: str | None = None, tags: list[str] | None = None):
        super().__init__()
        self.instance = (instance or config.get("ingestion.mastodon_instance", "https://mastodon.social")).rstrip("/")
        self.tags = list(tags or config.get("ingestion.mastodon_tags", ["stocks"])) + COMPANY_TAGS
        self._since: dict[str, str] = {}

    def fetch(self) -> list[RawDocument]:
        out = []
        for tag in self.tags:
            params = {"limit": 40, **({"since_id": self._since[tag]} if tag in self._since else {})}
            posts = self.get(f"{self.instance}/api/v1/timelines/tag/{tag}", params=params).json()
            if posts:
                self._since[tag] = posts[0]["id"]
            for p in posts:
                if p.get("reblog") or (p.get("language") or "en") != "en":
                    continue
                acct = p.get("account", {})
                out.append(RawDocument(source=self.name, source_type=SourceType.SOCIAL, publisher="mastodon",
                                       author="@" + acct.get("acct", "unknown"),
                                       published_at=parse_date(p.get("created_at")),
                                       text=clean(p.get("content", ""))[:600], url=p.get("url", ""),
                                       raw_meta={"followers": acct.get("followers_count", 0),
                                                 "reblogs": p.get("reblogs_count", 0),
                                                 "favourites": p.get("favourites_count", 0)}))
        return out


def corpus_posts(only_linked: bool = True, seed: int = 7) -> list[str]:
    frames = [pd.read_csv(RAW / f) for f in ("twitter_sentiment_valid.csv", "twitter_topic_valid.csv")]
    texts = pd.concat(frames)["text"].map(clean).drop_duplicates().tolist()
    if only_linked:
        from riskpulse.nlp.entities import link
        texts = [t for t in texts if any(e.type == "company" for e in link(t))]
    random.Random(seed).shuffle(texts)
    return texts


class SocialCorpusSource(Source):
    """Emits `per_poll` real corpus posts per poll, timestamped by the supplied clock."""

    name = "social_corpus"
    source_type = SourceType.SOCIAL
    poll_seconds = 5.0

    def __init__(self, per_poll: int = 2, now: Callable[[], datetime] | None = None, seed: int = 7):
        super().__init__()
        self.texts, self.per_poll, self._i = corpus_posts(seed=seed), per_poll, 0
        self.now = now or (lambda: datetime.now(timezone.utc))

    def fetch(self) -> list[RawDocument]:
        out = []
        for _ in range(self.per_poll):
            t = self.texts[self._i % len(self.texts)]
            self._i += 1
            out.append(RawDocument(source=self.name, source_type=SourceType.SOCIAL, publisher="x-corpus",
                                   author="@fin_corpus", published_at=self.now(), text=t,
                                   raw_meta={"corpus_index": self._i, "timestamp": "assigned at replay"}))
        return out
