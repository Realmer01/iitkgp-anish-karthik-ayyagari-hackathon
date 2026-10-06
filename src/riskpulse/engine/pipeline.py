"""RiskEngine - the unified AI/NLP risk engine (the "central processing unit").

    RawDocument batch
      -> normalise: clean, social slang/emoji, language + length filter
      -> entity linking (company / sector / market)                 no entity -> dropped as noise
      -> encode once: embeddings shared by clustering, events and the sentiment head
      -> story clustering: exact + near-duplicate collapse, mention count, confirming sources
      -> sentiment (document level; sentence level per company for multi-company texts)
      -> event classification (+ high-precision rules)
      -> impact score (+ drivers)
      -> one RiskSignal per (document, entity)
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import Counter, deque

import numpy as np

from riskpulse import __version__, config
from riskpulse.nlp import entities as ent
from riskpulse.nlp import impact
from riskpulse.nlp.embedding import LiteEmbedder, make_embedder
from riskpulse.nlp.events import EventClassifier
from riskpulse.nlp.preprocess import StoryClusterer, clean, is_english, normalize_social
from riskpulse.nlp.sentiment import (
    LiteSentiment,
    make_sentiment,
    policy_polarity,
    split_clauses,
    split_sentences,
    summarise,
)
from riskpulse.nlp.tier import resolve_tier
from riskpulse.schema import RawDocument, RiskSignal, SourceType

log = logging.getLogger("riskpulse.engine")
STORY_THRESHOLDS = {"full": (0.92, 0.72), "lite": (0.85, 0.55)}   # (dedup, same-story) cosine per vector space
LONG_TEXT_CHARS = 280


class RiskEngine:
    def __init__(self, tier: str | None = None):
        t0 = time.perf_counter()
        self.tier = resolve_tier(tier)
        try:
            self.embedder = make_embedder(self.tier)
            self.sentiment = make_sentiment(self.tier)
        except Exception as exc:  # offline with an empty model cache, broken torch install, ...
            log.warning("full tier failed to load (%s) - falling back to lite", exc)
            self.tier, self.embedder, self.sentiment = "lite", LiteEmbedder(), LiteSentiment()
        self.events = EventClassifier(self.tier)
        self.stories = self._new_clusterer()
        self.min_chars = int(config.get("engine.min_chars", 20))
        self.min_words = int(config.get("engine.min_words", 4))
        self.model_version = (f"riskpulse-{__version__}|{self.tier}|{self.sentiment.version}|"
                              f"{self.events.version}|impact-{impact.weights_source()}")
        self.stats: Counter = Counter()
        self.latencies: deque[float] = deque(maxlen=1000)
        self.started = time.time()
        self._lock = threading.Lock()
        log.info("RiskEngine ready in %.1fs: %s", time.perf_counter() - t0, self.model_version)

    def _new_clusterer(self) -> StoryClusterer:
        dedup, story = STORY_THRESHOLDS[self.tier]
        return StoryClusterer(dedup_threshold=dedup, story_threshold=story,
                              window_hours=float(config.get("engine.story_window_hours", 24)))

    # ------------------------------------------------------------------ public API
    def process(self, docs: list[RawDocument]) -> list[RiskSignal]:
        if not docs:
            return []
        with self._lock:
            return self._process(docs, self.stories)

    def analyze_text(self, text: str, source_type: str = "news", publisher: str = "analyst input") -> list[RiskSignal]:
        """Score one ad-hoc text (API /analyze, dashboard) without touching the live story state."""
        doc = RawDocument(source="manual", source_type=SourceType(source_type), publisher=publisher, text=text)
        with self._lock:
            return self._process([doc], self._new_clusterer(), count=False)

    def status(self) -> dict:
        lat = list(self.latencies)
        up = max(time.time() - self.started, 1e-9)
        return {"model_tier": self.tier, "model_version": self.model_version, **dict(self.stats),
                "docs_per_second": round(self.stats["docs_in"] / up, 3),
                "latency_ms_p50": round(float(np.percentile(lat, 50)), 1) if lat else None,
                "latency_ms_p95": round(float(np.percentile(lat, 95)), 1) if lat else None}

    # ------------------------------------------------------------------ pipeline
    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(texts)

    def _sentiment(self, texts: list[str], types: list[str], emb: np.ndarray) -> np.ndarray:
        p = self.sentiment.probs(texts, types, emb)
        for i, (t, st) in enumerate(zip(texts, types)):
            chunks = split_sentences(t) if len(t) > LONG_TEXT_CHARS else [t]
            if len(chunks) > 1:
                pc = self.sentiment.probs(chunks, [st] * len(chunks), self._encode(chunks))
                w = np.array([len(c) for c in chunks], dtype=float)
                p[i] = (pc * w[:, None]).sum(axis=0) / w.sum()
        return p

    def _entity_sentiment(self, multi: list[tuple[int, str, str, list]]) -> dict[tuple[int, str], np.ndarray]:
        """Sentence-level sentiment for texts naming several companies ("Apple beats, Boeing slumps")."""
        jobs = [(i, s, st, [e.id for e in ents if ent.mention_spans(s, e)])
                for i, text, st, ents in multi for s in split_clauses(text)]
        jobs = [j for j in jobs if j[3]]
        if not jobs:
            return {}
        sents = [s for _, s, _, _ in jobs]
        probs = self.sentiment.probs(sents, [st for _, _, st, _ in jobs], self._encode(sents))
        acc: dict[tuple[int, str], list[np.ndarray]] = {}
        for (i, _, _, hit), p in zip(jobs, probs):
            for eid in hit:
                acc.setdefault((i, eid), []).append(p)
        return {k: np.mean(v, axis=0) for k, v in acc.items()}

    def _process(self, docs: list[RawDocument], stories: StoryClusterer, count: bool = True) -> list[RiskSignal]:
        t0 = time.perf_counter()
        stats = self.stats if count else Counter()
        stats["docs_in"] += len(docs)
        kept, shown, model_texts, links = [], [], [], []
        for d in docs:
            text = clean(d.content)
            if len(text) < self.min_chars or len(text.split()) < self.min_words or not is_english(text):
                stats["dropped_language_or_length"] += 1
                continue
            entities = ent.link(text)
            if not entities:
                stats["dropped_no_entity"] += 1
                continue
            kept.append(d)
            shown.append(text)
            model_texts.append(normalize_social(text) if d.source_type == SourceType.SOCIAL else text)
            links.append(entities)
        if not kept:
            return []

        emb = self._encode(model_texts)
        assignments = [stories.assign(e, d.origin, d.published_at, t) for d, e, t in zip(kept, emb, shown)]
        keep = [i for i, a in enumerate(assignments) if not a.duplicate]
        stats["dropped_duplicate"] += len(kept) - len(keep)
        if not keep:
            return []
        kept, shown, model_texts, links, assignments = ([seq[i] for i in keep]
                                                        for seq in (kept, shown, model_texts, links, assignments))
        emb = emb[keep]
        types = [d.source_type.value for d in kept]

        doc_p = self._sentiment(model_texts, types, emb)
        multi = [(i, t, st, [e for e in es if e.type == "company"])
                 for i, (t, st, es) in enumerate(zip(model_texts, types, links))
                 if sum(e.type == "company" for e in es) > 1]
        ent_p = self._entity_sentiment(multi) if multi else {}
        events = self.events.predict(model_texts, emb)
        for i, ev in enumerate(events):
            if ev["event_type"] == "Macroeconomic":
                doc_p[i] = policy_polarity(model_texts[i], doc_p[i])

        per_doc_ms = (time.perf_counter() - t0) * 1000 / len(kept)
        out: list[RiskSignal] = []
        for i, (d, text, entities, a) in enumerate(zip(kept, shown, links, assignments)):
            ev, regions = events[i], ent.regions(text)
            for e in entities:
                s_score, s_label, s_conf = summarise(ent_p.get((i, e.id), doc_p[i]))
                score, drivers = impact.score(
                    event_type=ev["event_type"], event_conf=ev["event_confidence"], sentiment=s_score,
                    sentiment_conf=s_conf, text=text, entity=e, source_type=types[i], publisher=d.publisher,
                    n_sources=len(a.sources), mentions=a.mention_count, novel=a.novel)
                out.append(RiskSignal(
                    signal_id=hashlib.sha1(f"{d.id}|{e.id}".encode()).hexdigest()[:16],
                    published_at=d.published_at, source=d.source, source_type=d.source_type, publisher=d.origin,
                    doc_ids=[d.id], story_id=a.story_id, url=d.url, headline=text[:300], entity=e, regions=regions,
                    sentiment_score=s_score, sentiment_label=s_label, sentiment_confidence=s_conf,
                    impact_score=score, impact_drivers=drivers, mention_count=a.mention_count, sources=a.sources,
                    model_tier=self.tier, model_version=self.model_version, latency_ms=round(per_doc_ms, 1), **ev))
        stats["signals_out"] += len(out)
        if count:
            self.latencies.append(per_doc_ms)
        return out
