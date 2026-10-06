"""Sentiment Score in [-1, 1] = P(positive) - P(negative); confidence = max class probability.

full : FinBERT (ProsusAI/finbert, finance-domain BERT) blended with a "social head" - logistic
       regression on bge-small embeddings trained on 9.5k labelled finance tweets, which knows
       cashtag/slang conventions FinBERT never saw. Blend weight per source type is tuned on a dev
       split by scripts/train_models.py:  p = w[type] * p_finbert + (1 - w[type]) * p_head
lite : VADER plus a hand-built finance lexicon (VADER alone thinks "downgrade" is neutral).
       compound c maps to probabilities with p_pos - p_neg = c; the neutral band width is tuned on
       the training tweets.
Long texts are chunked by sentence and the chunk probabilities averaged, weighted by length.
"""
from __future__ import annotations

import logging
import re

import joblib
import numpy as np

from riskpulse import config
from riskpulse.config import MODELS

log = logging.getLogger("riskpulse.sentiment")
LABELS = ("negative", "neutral", "positive")          # canonical column order of every probability matrix
SENT_SPLIT_RX = re.compile(r"(?<=[.!?;])\s+(?=[A-Z$])|\s+[-|]\s+|\s*\n+\s*")

# VADER valence scale is roughly [-4, 4]. Terms chosen for financial headlines and posts.
FINANCE_LEXICON = {
    "beat": 1.6, "beats": 1.6, "miss": -1.6, "misses": -1.8, "missed": -1.6, "upgrade": 1.8, "upgrades": 1.8,
    "upgraded": 1.8, "downgrade": -1.9, "downgrades": -1.9, "downgraded": -1.9, "outperform": 1.6,
    "underperform": -1.6, "overweight": 1.0, "underweight": -1.0, "bullish": 2.2, "bearish": -2.2,
    "surge": 1.9, "surges": 1.9, "surged": 1.9, "soar": 2.1, "soars": 2.1, "soared": 2.1, "jump": 1.5,
    "jumps": 1.5, "jumped": 1.5, "rally": 1.7, "rallies": 1.7, "rallied": 1.7, "gain": 1.2, "gains": 1.3,
    "rise": 1.1, "rises": 1.1, "rose": 1.1, "climb": 1.1, "climbs": 1.2, "boost": 1.3,
    "boosts": 1.3, "profit": 1.2, "profits": 1.2, "growth": 1.0, "dividend": 0.7, "buyback": 1.1,
    "plunge": -2.4, "plunges": -2.4, "plunged": -2.4, "plummet": -2.5, "plummets": -2.5, "tumble": -2.0,
    "tumbles": -2.0, "slump": -2.0, "slumps": -2.0, "sink": -1.6, "sinks": -1.6, "slide": -1.5, "slides": -1.5,
    "fall": -1.3, "falls": -1.3, "fell": -1.3, "drop": -1.3, "drops": -1.4, "dropped": -1.3, "decline": -1.4,
    "declines": -1.4, "loss": -1.5, "losses": -1.6, "selloff": -2.1, "sell-off": -2.1, "crash": -2.7,
    "default": -2.6, "defaults": -2.6, "bankruptcy": -3.0, "bankrupt": -3.0, "insolvency": -2.8,
    "lawsuit": -1.6, "sued": -1.6, "probe": -1.3, "fraud": -2.8, "fined": -1.8, "recall": -1.7,
    "recalls": -1.7, "layoffs": -1.8, "warning": -1.5, "warns": -1.5, "sanctions": -1.6,
    "tariffs": -1.3, "war": -2.6, "recession": -2.3, "inflation": -0.9, "volatility": -0.8,
    "headwinds": -1.4, "tailwinds": 1.3, "outage": -1.8, "hack": -2.0, "breach": -2.0, "halts": -1.5,
}


CLAUSE_SPLIT_RX = re.compile(r"[,;]?\s+\b(?:while|whereas|but|however|although|even as)\b\s+|;\s+", re.I)


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT_RX.split(text) if p and len(p.strip()) > 3]
    return parts or [text]


def split_clauses(text: str) -> list[str]:
    """Sentences, further split at contrastive conjunctions - for company-level sentiment."""
    parts = [c.strip() for s in split_sentences(text) for c in CLAUSE_SPLIT_RX.split(s) if c and len(c.strip()) > 3]
    return parts or [text]


def policy_polarity(text: str, p: np.ndarray) -> np.ndarray:
    """Read monetary-policy news from the equity investor's side: models trained on corporate news
    see "raises rates" as positive, but tightening is negative for risk assets (and easing positive).
    Swaps the positive/negative mass when the wording is unambiguously hawkish or dovish."""
    from riskpulse.nlp.lexicons import DOVISH_RX, HAWKISH_RX

    hawk, dove = bool(HAWKISH_RX.search(text)), bool(DOVISH_RX.search(text))
    if hawk != dove and ((hawk and p[2] > p[0]) or (dove and p[0] > p[2])):
        p = p.copy()
        p[0], p[2] = p[2], p[0]
    return p


def summarise(p: np.ndarray) -> tuple[float, str, float]:
    return (round(float(np.clip(p[2] - p[0], -1, 1)), 4), LABELS[int(np.argmax(p))], round(float(np.max(p)), 4))


class LiteSentiment:
    tier = "lite"
    version = "vader+finlex"

    def __init__(self, neutral_weight: float | None = None):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.vader = SentimentIntensityAnalyzer()
        self.vader.lexicon.update(FINANCE_LEXICON)
        bundle = MODELS / "sentiment_lite.joblib"
        self.neutral_weight = neutral_weight if neutral_weight is not None else (
            joblib.load(bundle)["neutral_weight"] if bundle.exists() else 0.35)

    def compound(self, texts: list[str]) -> np.ndarray:
        return np.array([self.vader.polarity_scores(t)["compound"] for t in texts])

    def probs(self, texts: list[str], source_types: list[str] | None = None, embeddings=None) -> np.ndarray:
        c = self.compound(texts)
        m = np.abs(c)
        neu = (1 - m) * self.neutral_weight
        pos, neg = np.where(c > 0, m, 0.0), np.where(c < 0, m, 0.0)
        p = np.stack([neg, neu, pos], axis=1)
        return p / p.sum(axis=1, keepdims=True).clip(min=1e-9) if len(texts) else np.zeros((0, 3))


class FullSentiment:
    tier = "full"

    def __init__(self, model_id: str | None = None, batch_size: int | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_id = model_id or config.get("engine.sentiment_model", "ProsusAI/finbert")
        self.batch_size = batch_size or int(config.get("engine.batch_size", 32))
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).eval()
        id2label = {i: lab.lower() for i, lab in self.model.config.id2label.items()}
        self._order = [next(i for i, lab in id2label.items() if lab == want) for want in LABELS]
        self._torch = torch
        self.head, self.blend = None, {"news": 1.0, "social": 1.0}
        head = MODELS / "sentiment_head.joblib"
        if head.exists():
            bundle = joblib.load(head)
            self.head, self.blend = bundle["model"], bundle["blend"]
            self._head_order = [list(bundle["classes"]).index(lab) for lab in LABELS]
        self.version = "finbert" + ("+social-head" if self.head is not None else "")

    def finbert(self, texts: list[str]) -> np.ndarray:
        out = []
        with self._torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True, max_length=128,
                               return_tensors="pt")
                out.append(self._torch.softmax(self.model(**enc).logits, dim=-1).numpy()[:, self._order])
        return np.vstack(out) if out else np.zeros((0, 3))

    def head_probs(self, embeddings: np.ndarray) -> np.ndarray:
        return self.head.predict_proba(embeddings)[:, self._head_order]

    def probs(self, texts: list[str], source_types: list[str], embeddings: np.ndarray | None = None) -> np.ndarray:
        p = self.finbert(texts)
        if self.head is not None and embeddings is not None and len(texts):
            w = np.array([self.blend.get(st, 1.0) for st in source_types])[:, None]
            p = w * p + (1 - w) * self.head_probs(embeddings)
        return p


def make_sentiment(tier: str):
    return FullSentiment() if tier == "full" else LiteSentiment()


def chunked_probs(model, text: str, source_type: str, embed) -> np.ndarray:
    """Length-weighted average over sentence chunks, for texts longer than a headline."""
    chunks = split_sentences(text)
    if len(chunks) == 1:
        return model.probs([text], [source_type], embed([text]))[0]
    p = model.probs(chunks, [source_type] * len(chunks), embed(chunks))
    w = np.array([len(c) for c in chunks], dtype=float)
    return (p * w[:, None]).sum(axis=0) / w.sum()
