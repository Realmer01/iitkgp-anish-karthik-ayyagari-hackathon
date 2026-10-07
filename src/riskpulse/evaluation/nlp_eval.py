"""Sentiment and event-classification evaluation against naive baselines.

Datasets
  tweets_valid   zeroshot finance-tweet validation split (never used in training or tuning)
  phrasebank     Financial PhraseBank AllAgree news sentences (FinBERT was trained on it: leaky for FinBERT)
  gold_news      400 real archive headlines, labelled blind to model output (out-of-domain for every model)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from riskpulse.config import LABELED, PROCESSED, RAW
from riskpulse.nlp.events import EventClassifier, KeywordBaseline, rule_override
from riskpulse.nlp.preprocess import clean, normalize_social
from riskpulse.nlp.sentiment import LABELS, LiteSentiment, policy_polarity
from riskpulse.schema import EVENT_TYPES

TWEET_LABELS = {0: "negative", 1: "positive", 2: "neutral"}


def _scores(y, p) -> dict:
    return {"n": int(len(y)), "accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro", labels=sorted(set(y)))), 4)}


def sentiment_datasets() -> dict[str, tuple[list[str], list[str], str]]:
    tw = pd.read_csv(RAW / "twitter_sentiment_valid.csv")
    pb = pd.read_csv(RAW / "phrasebank_allagree.csv")
    gold = pd.read_csv(LABELED / "gold_sample.csv").merge(pd.read_csv(LABELED / "gold_labels.csv"), on="gold_id")
    return {"tweets_valid": ([normalize_social(clean(t)) for t in tw["text"]], tw["label"].map(TWEET_LABELS).tolist(), "social"),
            "phrasebank": ([clean(t) for t in pb["text"]], pb["label"].tolist(), "news"),
            "gold_news": ([clean(t) for t in gold["text"]], gold["sentiment"].tolist(), "news")}


def vader_naive(texts: list[str]) -> list[str]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    v = SentimentIntensityAnalyzer()
    c = np.array([v.polarity_scores(t)["compound"] for t in texts])
    return np.where(c >= 0.05, "positive", np.where(c <= -0.05, "negative", "neutral")).tolist()


def evaluate_sentiment(full: bool) -> dict:
    rows, preds = [], {}
    lite = LiteSentiment()
    models = {"VADER (naive baseline)": lambda x, st: vader_naive(x),
              "Lite: VADER + finance lexicon": lambda x, st: [LABELS[i] for i in lite.probs(x).argmax(1)]}
    if full:
        from riskpulse.nlp.embedding import FullEmbedder
        from riskpulse.nlp.sentiment import FullSentiment
        emb, fs = FullEmbedder(), FullSentiment()
        ev = EventClassifier("full")

        def blend(x, st):
            e = emb.encode(x)
            p = fs.probs(x, [st] * len(x), e)
            if st == "news":   # the engine's monetary-policy rule fires on Macroeconomic events
                for i, out in enumerate(ev.predict(x, e)):
                    if out["event_type"] == "Macroeconomic":
                        p[i] = policy_polarity(x[i], p[i])
            return [LABELS[i] for i in p.argmax(1)]
        models.update({"FinBERT": lambda x, st: [LABELS[i] for i in fs.finbert(x).argmax(1)],
                       "Social head (bge-small + LR)": lambda x, st: [LABELS[i] for i in fs.head_probs(emb.encode(x)).argmax(1)],
                       "RiskPulse full (blend + policy rule)": blend})
    for ds, (texts, y, st) in sentiment_datasets().items():
        for name, fn in models.items():
            p = fn(texts, st)
            preds[(ds, name)] = p
            rows.append({"dataset": ds, "model": name, **_scores(y, p)})
    best = "RiskPulse full (blend + policy rule)" if full else "Lite: VADER + finance lexicon"
    y_gold = sentiment_datasets()["gold_news"][1]
    cm = confusion_matrix(y_gold, preds[("gold_news", best)], labels=list(LABELS))
    return {"table": rows, "confusion": {"model": best, "dataset": "gold_news", "labels": list(LABELS),
                                         "matrix": cm.tolist()},
            "note": "PhraseBank is leaky for FinBERT (its training data). gold_news = 400 real headlines labelled blind "
                    "to model output: the fairest test. Macro-F1 over negative / neutral / positive."}


def evaluate_events(full: bool) -> dict:
    valid = pd.read_csv(PROCESSED / "events_valid.csv")
    gold = pd.read_csv(LABELED / "gold_sample.csv").merge(pd.read_csv(LABELED / "gold_labels.csv"), on="gold_id")
    data = {"tweets_valid (weak labels)": (valid["text"].tolist(), valid["event_type"].tolist()),
            "gold_news": ([clean(t) for t in gold["text"]], gold["event_type"].tolist())}
    kb = KeywordBaseline()

    def keyword(x, e=None):
        return [EVENT_TYPES[i] for i in kb.probs(x).argmax(1)]

    def with_rules(base):
        return lambda x, e=None: [rule_override(t) or b for t, b in zip(x, base(x, e))]

    models = {"Keyword baseline (naive)": keyword}
    lite = EventClassifier("lite")
    models["Lite: TF-IDF + LR + rules"] = lambda x, e=None: [o["event_type"] for o in lite.predict(x)]
    encoders = {}
    if full:
        from riskpulse.nlp.embedding import FullEmbedder
        enc, fullc = FullEmbedder(), EventClassifier("full")
        encoders = {ds: enc.encode(x) for ds, (x, _) in data.items()}
        models["Full: embeddings + LR (no rules)"] = lambda x, e: [EVENT_TYPES[i] for i in fullc.model_probs(x, e).argmax(1)]
        models["RiskPulse full: embeddings + LR + rules"] = lambda x, e: [o["event_type"] for o in fullc.predict(x, e)]
    rows, gold_preds = [], {}
    for ds, (x, y) in data.items():
        for name, fn in models.items():
            p = fn(x, encoders.get(ds))
            rows.append({"dataset": ds, "model": name, **_scores(y, p)})
            if ds == "gold_news":
                gold_preds[name] = p
    best = "RiskPulse full: embeddings + LR + rules" if full else "Lite: TF-IDF + LR + rules"
    y = data["gold_news"][1]
    classes = [c for c in EVENT_TYPES if c in set(y)]
    per_class = []
    f_best = f1_score(y, gold_preds[best], labels=classes, average=None)
    f_kw = f1_score(y, gold_preds["Keyword baseline (naive)"], labels=classes, average=None)
    for c, a, b in zip(classes, f_best, f_kw):
        per_class.append({"class": c, "support": int(sum(v == c for v in y)), "hybrid_f1": round(float(a), 3),
                          "keyword_f1": round(float(b), 3)})
    cm = confusion_matrix(y, gold_preds[best], labels=classes)
    return {"table": rows, "per_class": per_class,
            "confusion": {"model": best, "dataset": "gold_news", "labels": classes, "matrix": cm.tolist()},
            "gold_predictions": {k: v for k, v in gold_preds.items()},
            "note": "tweets_valid labels come from the same weak-supervision mapping as training (optimistic). "
                    "gold_news uses human-style labels on real headlines (out-of-domain). Classes with support < 10 "
                    "have unstable F1."}
