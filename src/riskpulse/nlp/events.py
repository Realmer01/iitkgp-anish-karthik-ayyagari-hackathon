"""Event Classification into the 12-class taxonomy (config/taxonomy.yaml).

full : average of two classifiers - logistic regression on bge-small embeddings (the vectors story
       clustering already computes) and a TF-IDF word/bi-gram logistic regression
lite : the TF-IDF logistic regression alone
Both are trained by scripts/train_models.py on the weakly-supervised tweet corpus plus the dev half of the
gold news headlines (domain adaptation), and ensembled with a
small set of high-precision rules (lexicons.OVERRIDES): when a rule fires ("files for Chapter 11",
"agrees to acquire"), its class wins. `KeywordBaseline` is the naive comparison used in evaluation.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np

from riskpulse.config import MODELS
from riskpulse.nlp import lexicons as lx
from riskpulse.schema import EVENT_TYPES

log = logging.getLogger("riskpulse.events")

KEYWORD_VOTES = [
    ("Credit Event", lx.CREDIT_RX), ("Geopolitical", lx.GEOPOLITICAL_RX), ("Operational Risk", lx.OPERATIONAL_RX),
    ("Merger & Acquisition", lx.MNA_RX), ("Legal & Regulatory", lx.LEGAL_RX), ("Leadership Change", lx.LEADERSHIP_RX),
    ("Product Launch", lx.PRODUCT_RX), ("Earnings", lx.EARNINGS_RX), ("Macroeconomic", lx.MACRO_RX),
]


def rule_override(text: str) -> str | None:
    for event, rx in lx.OVERRIDES:
        if rx.search(text):
            return event
    return None


class KeywordBaseline:
    """Naive approach: count keyword hits per class; no hits -> General News."""

    def probs(self, texts: list[str]) -> np.ndarray:
        out = np.full((len(texts), len(EVENT_TYPES)), 0.01)
        for row, t in zip(out, texts):
            for event, rx in KEYWORD_VOTES:
                row[EVENT_TYPES.index(event)] += len(rx.findall(t))
            if row.max() <= 0.01:
                row[EVENT_TYPES.index("General News")] = 1.0
        return out / out.sum(axis=1, keepdims=True)


def _aligned(model, raw: np.ndarray) -> np.ndarray:
    """Re-order a classifier's probability columns into EVENT_TYPES order."""
    out = np.zeros((raw.shape[0], len(EVENT_TYPES)))
    for j, c in enumerate(model.classes_):
        out[:, EVENT_TYPES.index(c)] = raw[:, j]
    return out


class EventClassifier:
    """Full tier: average of a TF-IDF and an embedding classifier (an ensemble); lite tier: TF-IDF only."""

    def __init__(self, tier: str):
        self.tier = tier
        path = MODELS / f"events_{tier}.joblib"
        self.model, self.parts = None, []
        if path.exists():
            bundle = joblib.load(path)
            if bundle.get("kind") == "ensemble":
                tfidf = bundle.get("tfidf") or joblib.load(MODELS / bundle["tfidf_from"])["model"]
                self.parts = [("text", tfidf), ("embedding", bundle["emb"])]
            else:
                self.model = bundle["model"]
                self.parts = [("embedding" if tier == "full" else "text", self.model)]
        else:
            log.warning("no trained event model at %s - using keyword baseline", path)
        self._fallback = KeywordBaseline()
        self.version = {2: f"ensemble-{tier}", 1: f"lr-{tier}"}.get(len(self.parts), "keywords")

    def model_probs(self, texts: list[str], embeddings: np.ndarray | None) -> np.ndarray:
        usable = [(kind, m) for kind, m in self.parts if kind == "text" or embeddings is not None]
        if not usable:
            return self._fallback.probs(texts)
        probs = [_aligned(m, m.predict_proba(texts if kind == "text" else embeddings)) for kind, m in usable]
        return np.mean(probs, axis=0)

    def predict(self, texts: list[str], embeddings: np.ndarray | None = None) -> list[dict]:
        probs = self.model_probs(texts, embeddings) if texts else np.zeros((0, len(EVENT_TYPES)))
        out = []
        for text, p in zip(texts, probs):
            best = int(np.argmax(p))
            label, conf, rule = EVENT_TYPES[best], float(p[best]), rule_override(text)
            if rule is not None:
                label, conf = rule, max(0.9, float(p[EVENT_TYPES.index(rule)]))
            out.append({"event_type": label, "event_confidence": round(min(conf, 1.0), 4), "event_rule": rule,
                        "event_probs": {e: round(float(x), 4) for e, x in zip(EVENT_TYPES, p) if x >= 0.005}})
        return out
