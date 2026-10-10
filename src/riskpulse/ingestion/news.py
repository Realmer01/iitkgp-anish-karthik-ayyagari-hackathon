"""News connectors: GDELT DOC 2.0, Yahoo Finance RSS, Google News RSS and NewsAPI (optional key)."""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse

import feedparser

from riskpulse import config
from riskpulse.ingestion.base import Source, excluded_publisher
from riskpulse.nlp.preprocess import clean, strip_publisher
from riskpulse.schema import RawDocument, SourceType
from riskpulse.universe import companies

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MACRO_QUERIES = ["federal reserve interest rates", "inflation economy", "sanctions tariffs trade war",
                 "stock market selloff", "credit downgrade default", "oil prices OPEC"]


def parse_date(value: str | None) -> datetime:
    """RSS (RFC 822), GDELT (20260101T120000Z) and ISO 8601 timestamps -> aware UTC datetime."""
    if value:
        for parse in (parsedate_to_datetime,
                      lambda v: datetime.strptime(v, "%Y%m%dT%H%M%SZ"),
                      lambda v: datetime.fromisoformat(v.replace("Z", "+00:00"))):
            try:
                dt = parse(value)
                return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
            except (TypeError, ValueError):
                continue
    return datetime.now(UTC)


def short_name(name: str) -> str:
    for suffix in (" Group Inc.", " Inc.", " Corp.", " & Co.", " Co."):   # longest first
        name = name.replace(suffix, "")
    return name


def parse_gdelt_articles(articles: list[dict]) -> list[RawDocument]:
    return [RawDocument(source="gdelt", source_type=SourceType.NEWS, publisher=a.get("domain", ""),
                        published_at=parse_date(a.get("seendate")), title=clean(a.get("title", "")),
                        url=a.get("url", ""), lang=(a.get("language") or "English")[:2].lower(),
                        raw_meta={"country": a.get("sourcecountry", "")})
            for a in articles if not excluded_publisher(a.get("domain", ""))]


class GdeltSource(Source):
    """Global news, updated every 15 minutes, no key. One OR-query per group of companies + macro."""

    name = "gdelt"

    def __init__(self, timespan: str = "1h", max_records: int = 75):
        super().__init__()
        self.timespan, self.max_records = timespan, max_records
        self.min_interval = float(config.get("ingestion.gdelt_min_interval_seconds", 6))
        names = [f'"{short_name(c.name)}"' for c in companies()]
        self.queries = [f"({' OR '.join(names[i:i + 6])}) sourcelang:english" for i in range(0, len(names), 6)]
        self.queries.append("(sanctions OR tariffs OR inflation OR default OR invasion) sourcelang:english")
        self._qi, self._last = 0, 0.0

    def fetch(self) -> list[RawDocument]:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        q = self.queries[self._qi % len(self.queries)]
        self._qi += 1
        self._last = time.monotonic()
        r = self.get(GDELT_URL, params={"query": q, "mode": "artlist", "format": "json", "sort": "datedesc",
                                        "maxrecords": self.max_records, "timespan": self.timespan})
        return parse_gdelt_articles(r.json().get("articles", []))


class YahooRSSSource(Source):
    """Yahoo Finance per-ticker headline RSS for the whole index universe."""

    name = "yahoo_rss"
    URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbols}&region=US&lang=en-US"

    def fetch(self) -> list[RawDocument]:
        out, tickers = [], [c.ticker for c in companies()]
        for i in range(0, len(tickers), 8):
            feed = feedparser.parse(self.get(self.URL.format(symbols=",".join(tickers[i:i + 8]))).content)
            for e in feed.entries:
                out.append(RawDocument(source=self.name, source_type=SourceType.NEWS,
                                       publisher=urlparse(e.get("link", "")).netloc or "finance.yahoo.com",
                                       published_at=parse_date(e.get("published")), title=clean(e.get("title", "")),
                                       text=clean(e.get("summary", "")), url=e.get("link", "")))
        return out


class GoogleNewsSource(Source):
    """Google News RSS search: macro / geopolitical queries live, and date-bounded historical back-fills."""

    name = "google_news"
    URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

    def __init__(self, queries: list[str] | None = None):
        super().__init__()
        self.queries = queries or [f"{q} when:1d" for q in MACRO_QUERIES]

    def search(self, query: str) -> list[RawDocument]:
        out = []
        for e in feedparser.parse(self.get(self.URL.format(q=quote_plus(query))).content).entries:
            publisher = (e.get("source") or {}).get("title", "")
            if excluded_publisher(publisher):
                continue
            out.append(RawDocument(source=self.name, source_type=SourceType.NEWS, publisher=publisher,
                                   published_at=parse_date(e.get("published")),
                                   title=clean(strip_publisher(e.get("title", ""), publisher)), url=e.get("link", "")))
        return out

    def fetch(self) -> list[RawDocument]:
        out = []
        for q in self.queries:
            out += self.search(q)
            time.sleep(1.0)
        return out


class NewsAPISource(Source):
    """newsapi.org free developer plan; enabled only when NEWSAPI_KEY is set (.env)."""

    name = "newsapi"
    URL = "https://newsapi.org/v2/everything"

    def __init__(self, key: str | None = None):
        super().__init__()
        self.key = key or os.environ.get("NEWSAPI_KEY", "")

    def fetch(self) -> list[RawDocument]:
        if not self.key:
            return []
        q = " OR ".join(f'"{short_name(c.name)}"' for c in companies()[:12])
        r = self.get(self.URL, params={"q": q, "language": "en", "sortBy": "publishedAt", "pageSize": 100},
                     headers={"X-Api-Key": self.key})
        return [RawDocument(source=self.name, source_type=SourceType.NEWS,
                            publisher=(a.get("source") or {}).get("name", ""), published_at=parse_date(a.get("publishedAt")),
                            title=clean(a.get("title") or ""), text=clean(a.get("description") or ""),
                            url=a.get("url", "")) for a in r.json().get("articles", [])]
