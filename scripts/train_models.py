"""Train the engine's small model artefacts (seconds to minutes on a laptop CPU).

    python scripts/train_models.py            # lite + full
    python scripts/train_models.py --lite     # lite only (no torch needed)

Writes to models/:
  sentiment_lite.joblib  neutral-band width for VADER + lexicon, tuned on a dev split of the training tweets
  events_lite.joblib     TF-IDF + logistic regression event classifier
  events_full.joblib     ensemble of TF-IDF LR and bge-small-embedding LR (probabilities averaged)
  sentiment_head.joblib  social sentiment head (LR on embeddings) + FinBERT/head blend weight per source type

Domain adaptation. The training corpus is tweets; the engine mostly reads news headlines. The gold set of
400 real headlines is split 50/50 (seeded, stratified, data/labeled/gold_split.csv). Only the DEV half is
used here: its event labels are added to training with weight 15, and it tunes the news sentiment blend.
Those choices (model family, weight, blend) were selected by 5-fold cross-validation inside the dev half.
The TEST half is never used for training or tuning; scripts/evaluate (run.py evaluate) reports on it.
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import _bootstrap  # noqa: F401
from riskpulse.config import LABELED, MODELS, PROCESSED, RAW
from riskpulse.nlp.preprocess import clean, normalize_social
from riskpulse.nlp.sentiment import LABELS, LiteSentiment

SEED = 42
GOLD_SEED = 2026
IN_DOMAIN_WEIGHT = 15.0          # chosen by dev-half CV among {0, 5, 15}
TWEET_LABELS = {0: "negative", 1: "positive", 2: "neutral"}   # zeroshot sentiment corpus encoding


def tweets_train() -> pd.DataFrame:
    df = pd.read_csv(RAW / "twitter_sentiment_train.csv")
    df["text"] = df["text"].map(clean).map(normalize_social)
    df["label"] = df["label"].map(TWEET_LABELS)
    return df


def gold_split() -> pd.DataFrame:
    """Seeded, event-stratified 50/50 dev/test split of the gold headlines (persisted for transparency)."""
    gold = pd.read_csv(LABELED / "gold_sample.csv").merge(pd.read_csv(LABELED / "gold_labels.csv"), on="gold_id")
    gold["text"] = gold["text"].map(clean)
    strat = gold["event_type"].where(gold["event_type"].map(gold["event_type"].value_counts()) >= 5, "rare")
    dev, _ = train_test_split(gold, test_size=0.5, random_state=GOLD_SEED, stratify=strat)
    gold["split"] = np.where(gold["gold_id"].isin(dev["gold_id"]), "dev", "test")
    gold[["gold_id", "split"]].to_csv(LABELED / "gold_split.csv", index=False, lineterminator="\n")
    return gold


def event_training(events: pd.DataFrame, gold: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray]:
    dev = gold[gold["split"] == "dev"]
    texts = events["text"].tolist() + dev["text"].tolist()
    labels = events["event_type"].tolist() + dev["event_type"].tolist()
    weights = np.r_[np.ones(len(events)), np.full(len(dev), IN_DOMAIN_WEIGHT)]
    return texts, labels, weights


def tfidf_lr():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True,
                                         stop_words="english"),
                         LogisticRegression(C=6.0, max_iter=3000, class_weight="balanced"))


def train_lite(events: pd.DataFrame, tweets: pd.DataFrame, gold: pd.DataFrame) -> None:
    _, dev = train_test_split(tweets, test_size=0.15, random_state=SEED, stratify=tweets["label"])
    best = max(((f1_score(dev["label"], [LABELS[i] for i in LiteSentiment(neutral_weight=w).probs(dev["text"].tolist()).argmax(1)],
                          average="macro"), w) for w in np.arange(0.05, 1.01, 0.05)), key=lambda x: x[0])
    joblib.dump({"neutral_weight": round(float(best[1]), 2), "dev_macro_f1": round(best[0], 4)},
                MODELS / "sentiment_lite.joblib")
    print(f"  sentiment_lite: neutral_weight={best[1]:.2f} dev macro-F1={best[0]:.3f}")

    texts, labels, weights = event_training(events, gold)
    pipe = tfidf_lr()
    pipe.fit(texts, labels, logisticregression__sample_weight=weights)
    joblib.dump({"model": pipe, "classes": list(pipe.classes_), "trained_on": len(texts),
                 "in_domain_weight": IN_DOMAIN_WEIGHT}, MODELS / "events_lite.joblib", compress=3)
    print(f"  events_lite: {len(texts)} examples ({(gold['split'] == 'dev').sum()} in-domain), "
          f"{len(pipe.classes_)} classes")


def train_full(events: pd.DataFrame, tweets: pd.DataFrame, gold: pd.DataFrame) -> None:
    from riskpulse.nlp.embedding import FullEmbedder
    from riskpulse.nlp.sentiment import FullSentiment

    emb = FullEmbedder()
    t0 = time.time()
    texts, labels, weights = event_training(events, gold)
    xe = emb.encode(texts)
    emb_clf = LogisticRegression(C=4.0, max_iter=4000, class_weight="balanced").fit(xe, labels, sample_weight=weights)
    # The TF-IDF half of the ensemble is the lite-tier model (identical training data), stored once.
    joblib.dump({"kind": "ensemble", "tfidf_from": "events_lite.joblib", "emb": emb_clf, "classes": list(emb_clf.classes_),
                 "embedding_model": emb.model_id, "trained_on": len(texts), "in_domain_weight": IN_DOMAIN_WEIGHT},
                MODELS / "events_full.joblib", compress=3)
    print(f"  events_full: ensemble of TF-IDF and embedding LR, {len(texts)} examples, {time.time() - t0:.0f}s")

    tr, dev = train_test_split(tweets, test_size=0.15, random_state=SEED, stratify=tweets["label"])
    xtr, xdev = emb.encode(tr["text"].tolist()), emb.encode(dev["text"].tolist())
    head = LogisticRegression(C=4.0, max_iter=4000, class_weight="balanced").fit(xtr, tr["label"])
    order = [list(head.classes_).index(lab) for lab in LABELS]
    fin = FullSentiment()
    fin.head = None

    def best_blend(p_fin, p_head, y):
        scores = [(f1_score(y, [LABELS[i] for i in (w * p_fin + (1 - w) * p_head).argmax(1)], average="macro"),
                   round(float(w), 1)) for w in np.arange(0.0, 1.01, 0.1)]
        return max(scores), scores

    (f1_social, w_social), social_curve = best_blend(fin.finbert(dev["text"].tolist()), head.predict_proba(xdev)[:, order],
                                                     dev["label"])
    news_dev = gold[gold["split"] == "dev"]
    (f1_news, w_news), news_curve = best_blend(fin.finbert(news_dev["text"].tolist()),
                                               head.predict_proba(emb.encode(news_dev["text"].tolist()))[:, order],
                                               news_dev["sentiment"])
    blend = {"social": w_social, "news": w_news}
    head_full = LogisticRegression(C=4.0, max_iter=4000, class_weight="balanced").fit(
        np.vstack([xtr, xdev]), pd.concat([tr["label"], dev["label"]]))
    joblib.dump({"model": head_full, "classes": list(head_full.classes_), "blend": blend,
                 "dev_curves": {"social": social_curve, "news": news_curve}, "embedding_model": emb.model_id},
                MODELS / "sentiment_head.joblib", compress=3)
    print(f"  sentiment_head: blend={blend} (dev macro-F1 social {f1_social:.3f}, news {f1_news:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", action="store_true")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    events = pd.read_csv(PROCESSED / "events_train.csv")
    tweets = tweets_train()
    gold = gold_split()
    print(f"Gold split: {gold['split'].value_counts().to_dict()} (test half is never used here)")
    print("Training lite tier:")
    train_lite(events, tweets, gold)
    if not args.lite:
        print("Training full tier:")
        train_full(events, tweets, gold)


if __name__ == "__main__":
    main()
