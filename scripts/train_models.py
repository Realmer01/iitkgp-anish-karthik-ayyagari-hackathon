"""Train the engine's small model artefacts (seconds to minutes on a laptop CPU).

    python scripts/train_models.py            # lite + full
    python scripts/train_models.py --lite     # lite only (no torch needed)

Writes to models/:
  sentiment_lite.joblib  neutral-band width for VADER+lexicon, tuned on a dev split of the training tweets
  events_lite.joblib     TF-IDF + logistic regression event classifier
  events_full.joblib     logistic regression on bge-small embeddings
  sentiment_head.joblib  social sentiment head (LR on embeddings) + FinBERT/head blend weights
Only TRAIN files are used here. Validation files are reserved for scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from riskpulse.config import MODELS, PROCESSED, RAW
from riskpulse.nlp.preprocess import clean, normalize_social
from riskpulse.nlp.sentiment import LABELS, LiteSentiment

SEED = 42
TWEET_LABELS = {0: "negative", 1: "positive", 2: "neutral"}   # zeroshot sentiment corpus encoding


def tweets_train() -> pd.DataFrame:
    df = pd.read_csv(RAW / "twitter_sentiment_train.csv")
    df["text"] = df["text"].map(clean).map(normalize_social)
    df["label"] = df["label"].map(TWEET_LABELS)
    return df


def train_lite(events: pd.DataFrame, tweets: pd.DataFrame) -> None:
    tr, dev = train_test_split(tweets, test_size=0.15, random_state=SEED, stratify=tweets["label"])
    model = LiteSentiment(neutral_weight=1.0)
    best = max(((f1_score(dev["label"], [LABELS[i] for i in LiteSentiment(neutral_weight=w).probs(dev["text"].tolist()).argmax(1)],
                          average="macro"), w) for w in np.arange(0.05, 1.01, 0.05)), key=lambda x: x[0])
    joblib.dump({"neutral_weight": round(float(best[1]), 2), "dev_macro_f1": round(best[0], 4)},
                MODELS / "sentiment_lite.joblib")
    print(f"  sentiment_lite: neutral_weight={best[1]:.2f} dev macro-F1={best[0]:.3f}")
    del model

    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True,
                                         stop_words="english"),
                         LogisticRegression(C=6.0, max_iter=3000, class_weight="balanced"))
    pipe.fit(events["text"], events["event_type"])
    joblib.dump({"model": pipe, "classes": list(pipe.classes_), "trained_on": len(events)}, MODELS / "events_lite.joblib",
                compress=3)
    print(f"  events_lite: {len(events)} examples, {len(pipe.classes_)} classes")


def train_full(events: pd.DataFrame, tweets: pd.DataFrame) -> None:
    from riskpulse.nlp.embedding import FullEmbedder
    from riskpulse.nlp.sentiment import FullSentiment

    emb = FullEmbedder()
    t0 = time.time()
    xe = emb.encode(events["text"].tolist())
    clf = LogisticRegression(C=4.0, max_iter=4000, class_weight="balanced").fit(xe, events["event_type"])
    joblib.dump({"model": clf, "classes": list(clf.classes_), "embedding_model": emb.model_id,
                 "trained_on": len(events)}, MODELS / "events_full.joblib", compress=3)
    print(f"  events_full: {len(events)} examples embedded in {time.time() - t0:.0f}s")

    tr, dev = train_test_split(tweets, test_size=0.15, random_state=SEED, stratify=tweets["label"])
    xtr, xdev = emb.encode(tr["text"].tolist()), emb.encode(dev["text"].tolist())
    head = LogisticRegression(C=4.0, max_iter=4000, class_weight="balanced").fit(xtr, tr["label"])
    order = [list(head.classes_).index(lab) for lab in LABELS]
    p_head = head.predict_proba(xdev)[:, order]
    fin = FullSentiment()
    fin.head = None
    p_fin = fin.finbert(dev["text"].tolist())
    scores = []
    for w in np.arange(0.0, 1.01, 0.1):
        pred = [LABELS[i] for i in (w * p_fin + (1 - w) * p_head).argmax(1)]
        scores.append((f1_score(dev["label"], pred, average="macro"), round(float(w), 1)))
    best_f1, w_social = max(scores)
    # News blend: FinBERT is trained on financial news, so news keeps a FinBERT-heavy weight;
    # scripts/evaluate.py reports both components and the blend on the held-out news gold set.
    blend = {"social": w_social, "news": 0.8}
    head_full = LogisticRegression(C=4.0, max_iter=4000, class_weight="balanced").fit(
        np.vstack([xtr, xdev]), pd.concat([tr["label"], dev["label"]]))
    joblib.dump({"model": head_full, "classes": list(head_full.classes_), "blend": blend,
                 "dev_curve": scores, "embedding_model": emb.model_id}, MODELS / "sentiment_head.joblib", compress=3)
    print(f"  sentiment_head: blend={blend} dev macro-F1={best_f1:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", action="store_true")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    events = pd.read_csv(PROCESSED / "events_train.csv")
    tweets = tweets_train()
    print("Training lite tier:")
    train_lite(events, tweets)
    if not args.lite:
        print("Training full tier:")
        train_full(events, tweets)


if __name__ == "__main__":
    main()
