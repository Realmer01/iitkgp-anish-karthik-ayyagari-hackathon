"""Typed records that flow through RiskPulse.

    connector --RawDocument--> RiskEngine --RiskSignal--> bus / JSONL / SQLite / API --> subscribers

`RiskSignal` is the engine's public, versioned output contract. Downstream modules depend on
this file only - never on engine internals.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"


def utcnow() -> datetime:
    return datetime.now(UTC)


def as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


class SourceType(str, Enum):
    NEWS = "news"
    SOCIAL = "social"


class EventType(str, Enum):
    GEOPOLITICAL = "Geopolitical"
    MACROECONOMIC = "Macroeconomic"
    CREDIT_EVENT = "Credit Event"
    MERGER_ACQUISITION = "Merger & Acquisition"
    PRODUCT_LAUNCH = "Product Launch"
    EARNINGS = "Earnings"
    LEGAL_REGULATORY = "Legal & Regulatory"
    LEADERSHIP_CHANGE = "Leadership Change"
    OPERATIONAL_RISK = "Operational Risk"
    ANALYST_RATING = "Analyst Rating"
    CAPITAL_MARKETS = "Capital Markets"
    GENERAL_NEWS = "General News"


EVENT_TYPES: list[str] = [e.value for e in EventType]


class RawDocument(BaseModel):
    """A unit of unstructured text as delivered by a source connector."""

    id: str = ""
    source: str                      # connector id: gdelt, yahoo_rss, google_news, mastodon, social_corpus ...
    source_type: SourceType
    publisher: str = ""              # outlet / domain for news, platform for social
    author: str = ""                 # account handle for social posts
    published_at: datetime = Field(default_factory=utcnow)
    title: str = ""
    text: str = ""
    url: str = ""
    lang: str = "en"
    raw_meta: dict = Field(default_factory=dict)

    @field_validator("published_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return as_utc(v)

    def model_post_init(self, __context) -> None:
        if not self.id:
            # publisher + timestamp included: the same headline from another outlet, or republished days
            # later, is a different document (otherwise their signal ids would collide)
            key = (f"{self.source}|{self.publisher}|{self.author}|{self.published_at.isoformat()}|{self.url}|"
                   f"{self.title}|{self.text[:200]}")
            self.id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    @property
    def content(self) -> str:
        if self.title and self.text and not self.text.startswith(self.title):
            return f"{self.title}. {self.text}"
        return self.title or self.text

    @property
    def origin(self) -> str:
        """Who said it - used for duplicate detection and cross-source confirmation."""
        return self.author or self.publisher or self.source


class Entity(BaseModel):
    type: Literal["company", "sector", "market"]
    id: str                          # ticker, GICS sector name, or "MARKET"
    name: str
    ticker: str | None = None
    sector: str | None = None
    country: str | None = None
    matched: str = ""                # surface form that triggered the link


class ImpactDrivers(BaseModel):
    """Inputs and per-feature contributions behind an impact score (model-risk explainability)."""

    features: dict[str, float]       # each feature in [0, 1]
    contributions: dict[str, float]  # weight * feature, in logit units
    logit: float
    cues: list[str] = Field(default_factory=list)


class RiskSignal(BaseModel):
    signal_id: str
    schema_version: str = SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=utcnow)
    published_at: datetime
    source: str
    source_type: SourceType
    publisher: str = ""
    doc_ids: list[str]
    story_id: str
    url: str = ""
    headline: str

    entity: Entity
    regions: list[str] = Field(default_factory=list)

    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: Literal["positive", "neutral", "negative"]
    sentiment_confidence: float = Field(ge=0.0, le=1.0)

    event_type: str
    event_confidence: float = Field(ge=0.0, le=1.0)
    event_probs: dict[str, float] = Field(default_factory=dict)
    event_rule: str | None = None    # high-precision rule that fired, if any

    impact_score: float = Field(ge=1.0, le=10.0)
    impact_drivers: ImpactDrivers
    mention_count: int = 1           # documents in the story so far
    sources: list[str] = Field(default_factory=list)  # distinct publishers confirming the story

    model_tier: Literal["full", "lite"]
    model_version: str = ""
    latency_ms: float = 0.0

    @field_validator("event_type")
    @classmethod
    def _known_event(cls, v: str) -> str:
        if v not in EVENT_TYPES:
            raise ValueError(f"unknown event type {v!r}")
        return v
