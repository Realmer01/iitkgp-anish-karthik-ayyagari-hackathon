"""Build the event-classification training set by weak supervision.

The public zeroshot/twitter-financial-news-topic corpus labels ~21k finance tweets with 20
*topics* (Analyst Update, Fed | Central Banks, Company | Product News, ...). A risk engine needs
*risk events* instead (Credit Event, Geopolitical, Operational Risk, ...). We bridge the two with
labelling functions: every topic has a default target event, and high-precision keyword
functions re-route a tweet when its text clearly describes a different risk event (a
"Company | Product News" tweet about a ransomware attack becomes Operational Risk).

Rare classes are topped up with data/raw/synthetic_seed_events.csv - hand-written headlines
about *fictional* companies, flagged is_synthetic=1, used for training only (never evaluation).

Output: data/processed/events_{train,valid}.csv with columns text, event_type, topic, is_synthetic
"""
from __future__ import annotations

import re

import pandas as pd

import _bootstrap  # noqa: F401
from riskpulse.config import PROCESSED, RAW
from riskpulse.nlp import lexicons as lx
from riskpulse.nlp.preprocess import clean

TOPICS = ["Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt",
          "Dividend", "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion",
          "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets",
          "Politics", "Personnel Change", "Stock Commentary", "Stock Movement"]

DEFAULT = {
    "Analyst Update": "Analyst Rating", "Fed | Central Banks": "Macroeconomic",
    "Company | Product News": "General News", "Treasuries | Corporate Debt": "Macroeconomic",
    "Dividend": "Capital Markets", "Earnings": "Earnings", "Energy | Oil": "Macroeconomic",
    "Financials": "General News", "Currencies": "Macroeconomic", "General News | Opinion": "General News",
    "Gold | Metals | Materials": "Macroeconomic", "IPO": "Capital Markets",
    "Legal | Regulation": "Legal & Regulatory", "M&A | Investments": "Merger & Acquisition",
    "Macro": "Macroeconomic", "Markets": "General News", "Politics": "Geopolitical",
    "Personnel Change": "Leadership Change", "Stock Commentary": "General News",
    "Stock Movement": "General News",
}
STRONG_GEO = re.compile(r"\b(war|invasion|invade[sd]?|sanctions?|embargo|missiles?|military|troops|tariffs?|"
                        r"trade war|coup|nato|ceasefire|export controls?|nuclear|terror\w*|airstrikes?|"
                        r"geopolitic\w*|blockade|retaliat\w*)\b", re.I)
STRONG_CREDIT = re.compile(r"\b(default\w*|bankrupt\w*|chapter 11|insolven\w*|junk|downgrad\w*|restructur\w*|"
                           r"receivership|bank run|bailout|distress\w*|liquidity crisis|missed (a )?payment|"
                           r"credit ?watch|spreads? wid\w*|loan losses|bad loans|collapse[sd]?)\b", re.I)


def relabel(topic: str, text: str) -> str:
    """Labelling functions, applied in priority order."""
    if topic in ("Company | Product News", "Financials", "Treasuries | Corporate Debt", "Markets",
                 "General News | Opinion", "Stock Movement") and STRONG_CREDIT.search(text):
        return "Credit Event"
    if topic in ("General News | Opinion", "Macro", "Energy | Oil", "Markets", "Company | Product News",
                 "Gold | Metals | Materials", "Currencies") and STRONG_GEO.search(text):
        return "Geopolitical"
    if topic == "Company | Product News":
        if lx.OVERRIDES[3][1].search(text) or lx.OPERATIONAL_RX.search(text):
            return "Operational Risk"
        if lx.PRODUCT_RX.search(text):
            return "Product Launch"
        if lx.LEGAL_RX.search(text):
            return "Legal & Regulatory"
        if lx.MNA_RX.search(text):
            return "Merger & Acquisition"
    if topic in ("Stock Movement", "Stock Commentary") and lx.EARNINGS_RX.search(text) and \
            re.search(r"\b(results|earnings|eps|revenue|guidance|quarter)\b", text, re.I):
        return "Earnings"
    return DEFAULT[topic]


def build(split: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"twitter_topic_{split}.csv")
    df["topic"] = df["label"].map(dict(enumerate(TOPICS)))
    df["text"] = df["text"].map(clean)
    df = df[df["text"].str.len() >= 15].drop_duplicates("text")
    df["event_type"] = [relabel(t, x) for t, x in zip(df["topic"], df["text"])]
    df["is_synthetic"] = 0
    return df[["text", "event_type", "topic", "is_synthetic"]]


def main() -> None:
    out = PROCESSED
    out.mkdir(parents=True, exist_ok=True)
    train, valid = build("train"), build("valid")
    seed = pd.read_csv(RAW / "synthetic_seed_events.csv")
    seed["topic"], seed["is_synthetic"] = "synthetic seed", 1
    train = pd.concat([train, seed[train.columns]], ignore_index=True)
    train.to_csv(out / "events_train.csv", index=False)
    valid.to_csv(out / "events_valid.csv", index=False)
    print("train", len(train), train["event_type"].value_counts().to_dict())
    print("valid", len(valid), valid["event_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
