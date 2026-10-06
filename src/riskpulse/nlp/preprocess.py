"""Normaliser: cleaning, social-text handling, language filter and story clustering."""
from __future__ import annotations

import hashlib
import html
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

URL_RX = re.compile(r"https?://\S+|www\.\S+")
TAG_RX = re.compile(r"<[^>]+>")
WS_RX = re.compile(r"\s+")
SPLIT_TAG_RX = re.compile(r"(#)\s+(?=\w)")             # "# Finance" (Mastodon HTML) -> "#Finance"
HASHTAG_TAIL_RX = re.compile(r"(\s#\w+)+\s*$")         # trailing hashtag clouds carry no meaning
CASHTAG_RX = re.compile(r"\$([A-Za-z]{1,5})\b")
STOPWORDS = {"the", "a", "an", "and", "of", "to", "in", "on", "for", "is", "are", "with", "as", "at",
             "by", "from", "its", "it", "after", "over", "says", "new", "be", "has", "have", "will"}

# Social-media conventions rewritten into plain financial English that FinBERT understands.
SLANG = {  # order matters: specific phrases before generic ones
    r"(bears|shorts|short sellers) (got |getting |are )?(rekt|wrecked|crushed)": "the stock rallied strongly",
    r"crushed (it|earnings|estimates)": "beat expectations strongly",
    r"to the moon": "set for a strong rally", r"mooning": "rallying strongly", r"stonks?": "stocks",
    r"bag ?holders?": "investors stuck with losses", r"rekt": "suffered heavy losses", r"tendies": "profits",
    r"diamond hands": "holding with conviction", r"paper hands": "panic selling", r"btfd|buy the dip": "buying the dip",
    r"ath": "all-time high", r"yolo": "high-risk bet", r"fud": "fear and doubt", r"hodl": "hold",
}
SLANG_RX = [(re.compile(rf"(?<!\w){k}(?!\w)", re.I), v) for k, v in SLANG.items()]
EMOJI = {"🚀": " rally ", "📈": " rising ", "📉": " falling ", "🐂": " bullish ", "🐻": " bearish ",
         "💎": " conviction ", "🔥": " strong ", "💀": " disaster ", "😭": " painful ", "🤡": " foolish ",
         "💸": " losing money ", "💰": " profit ", "⚠️": " warning ", "🩸": " heavy losses ", "🟢": " up ", "🔴": " down "}


def clean(text: str) -> str:
    text = html.unescape(text or "")
    text = TAG_RX.sub(" ", text)
    text = URL_RX.sub(" ", text)
    text = SPLIT_TAG_RX.sub(r"\1", text)
    text = HASHTAG_TAIL_RX.sub("", text)
    text = CASHTAG_RX.sub(lambda m: "$" + m.group(1).upper(), text)
    return WS_RX.sub(" ", text).strip(" -|")


def normalize_social(text: str) -> str:
    """Rewrite slang and emojis into words a news-trained model can read."""
    for e, word in EMOJI.items():
        text = text.replace(e, word)
    for rx, repl in SLANG_RX:
        text = rx.sub(repl, text)
    return WS_RX.sub(" ", text).strip()


def strip_publisher(title: str, publisher: str) -> str:
    """Google News titles end in ' - Publisher'; drop it so it does not leak into the model."""
    if publisher and title.endswith(f" - {publisher}"):
        return title[: -len(publisher) - 3]
    return title


def is_english(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters or sum(ch.isascii() for ch in letters) / len(letters) < 0.9:
        return False
    words = re.findall(r"[a-z]+", text.lower())
    return len(words) < 6 or any(w in STOPWORDS for w in words)


def fingerprint(text: str) -> str:
    """Exact-duplicate key: case/punctuation-insensitive hash."""
    return hashlib.md5(re.sub(r"[^a-z0-9]+", " ", text.lower()).strip().encode()).hexdigest()


@dataclass
class Story:
    story_id: str
    centroid: np.ndarray
    first_seen: datetime
    last_seen: datetime
    sources: list[str] = field(default_factory=list)
    mentions: int = 1


@dataclass
class Assignment:
    story_id: str
    mention_count: int
    sources: list[str]
    duplicate: bool
    novel: bool


class StoryClusterer:
    """Online clustering of documents into stories over a sliding window.

    * exact duplicate (same fingerprint)                         -> dropped
    * near-duplicate (cosine >= dedup, same origin)              -> dropped (re-posts, syndication)
    * same story (cosine >= story threshold)                     -> merged: mention_count += 1,
                                                                    origin added to `sources`
    * otherwise                                                  -> new story (novel = True)
    Cross-source confirmation (len(sources)) and velocity feed the impact score.
    """

    def __init__(self, dedup_threshold: float = 0.92, story_threshold: float = 0.72,
                 window_hours: float = 24.0, max_stories: int = 5000):
        self.dedup, self.story_threshold = dedup_threshold, story_threshold
        self.window = timedelta(hours=window_hours)
        self.stories: deque[Story] = deque(maxlen=max_stories)
        self._fingerprints: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def _evict(self, now: datetime) -> None:
        while self.stories and now - self.stories[0].last_seen > self.window:
            self.stories.popleft()
        if len(self._fingerprints) > 100_000:
            cutoff = now - self.window
            self._fingerprints = {k: t for k, t in self._fingerprints.items() if t >= cutoff}

    def assign(self, emb: np.ndarray, origin: str, ts: datetime, text: str) -> Assignment:
        with self._lock:
            self._evict(ts)
            fp = fingerprint(text)
            if fp in self._fingerprints and ts - self._fingerprints[fp] <= self.window:
                return Assignment("", 0, [], duplicate=True, novel=False)
            self._fingerprints[fp] = ts
            best, sim = None, -1.0
            if self.stories:
                sims = np.stack([s.centroid for s in self.stories]) @ emb
                i = int(np.argmax(sims))
                best, sim = self.stories[i], float(sims[i])
            if best is not None and sim >= self.story_threshold:
                if sim >= self.dedup and origin in best.sources:
                    return Assignment(best.story_id, best.mentions, list(best.sources), duplicate=True, novel=False)
                best.mentions += 1
                best.centroid = best.centroid + (emb - best.centroid) / best.mentions
                best.centroid /= np.linalg.norm(best.centroid) + 1e-9
                if origin not in best.sources:
                    best.sources.append(origin)
                best.last_seen = max(best.last_seen, ts)
                return Assignment(best.story_id, best.mentions, list(best.sources), duplicate=False, novel=False)
            story = Story(f"S{uuid.uuid4().hex[:10]}", emb.copy(), ts, ts, [origin])
            self.stories.append(story)
            return Assignment(story.story_id, 1, [origin], duplicate=False, novel=True)
