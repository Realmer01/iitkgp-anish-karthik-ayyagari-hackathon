"""Draw the evaluation gold-set sample of REAL news headlines (seeded, reproducible).

    python scripts/build_gold_sample.py      -> data/labeled/gold_sample.csv

Design: 250 headlines uniformly at random from archive headlines the engine links to an entity, plus
150 targeted at rare risk classes (20-25 per cue lexicon) so per-class scores are measurable. The
training data are tweets, so news headlines are an out-of-domain test. Labels are assigned blind to
model predictions in data/labeled/gold_labels.csv (see data/README.md for the labelling guide).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import pandas as pd

from riskpulse.config import LABELED, RAW
from riskpulse.nlp import lexicons as lx
from riskpulse.nlp.entities import link
from riskpulse.nlp.preprocess import clean, is_english

SEED = 2026
TARGETS = {"credit": lx.CREDIT_RX, "operational": lx.OPERATIONAL_RX, "geopolitical": lx.GEOPOLITICAL_RX,
           "mna": lx.MNA_RX, "legal": lx.LEGAL_RX, "leadership": lx.LEADERSHIP_RX, "product": lx.PRODUCT_RX}


def main() -> None:
    df = pd.read_csv(RAW / "news_history.csv.gz").drop_duplicates("title")
    df["text"] = df["title"].map(clean)
    df = df[df["text"].str.split().str.len().ge(5) & df["text"].map(is_english)]
    df = df[[bool(link(t)) for t in df["text"]]]
    rand = df.sample(250, random_state=SEED)
    rest = df.drop(rand.index)
    picks = []
    for i, (name, rx) in enumerate(TARGETS.items()):
        pool = rest[rest["text"].str.contains(rx) & ~rest.index.isin([p for g in picks for p in g.index])]
        picks.append(pool.sample(min(len(pool), 22), random_state=SEED + i).assign(stratum=f"target:{name}"))
    sample = pd.concat([rand.assign(stratum="random"), *picks]).head(400)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle so strata are blind
    sample.insert(0, "gold_id", [f"G{i + 1:03d}" for i in range(len(sample))])
    LABELED.mkdir(parents=True, exist_ok=True)
    sample[["gold_id", "published_at", "publisher", "text", "stratum"]].to_csv(LABELED / "gold_sample.csv", index=False)
    print(len(sample), sample["stratum"].value_counts().to_dict())


if __name__ == "__main__":
    main()
