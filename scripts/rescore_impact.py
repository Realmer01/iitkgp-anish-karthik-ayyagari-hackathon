"""Re-score impact for precomputed signals from their stored features (no NLP re-run needed).

    python scripts/rescore_impact.py [--tier full]

* recomputes the causal `attention` feature (entity signals in the trailing 24h, in emission order -
  exactly what the engine computes online);
* recomputes impact with the current weights (expert, or calibrated if models/impact_calibrated.json
  exists and config/impact.yaml has use_calibrated: true);
* rewrites data/output/signals_history_<tier>.csv.gz and the demo JSONL (matched by signal_id).
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict, deque

import _bootstrap  # noqa: F401
import pandas as pd

from riskpulse.config import OUTPUT
from riskpulse.nlp import impact
from riskpulse.schema import RiskSignal


def add_attention(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["published_at"], utc=True, format="ISO8601")
    windows: dict[str, deque] = defaultdict(deque)
    counts = []
    for entity, t in zip(df["entity_id"], ts):
        q = windows[entity]
        while q and t - q[0] > pd.Timedelta(hours=24):
            q.popleft()
        q.append(t)
        counts.append(len(q))
    df["entity_signals_24h"] = counts
    df["f_attention"] = [round(impact.attention(n), 4) for n in counts]
    return df


def rescore(df: pd.DataFrame) -> pd.DataFrame:
    feats = df[[f"f_{f}" for f in impact.FEATURES]].rename(columns=lambda c: c[2:]).to_dict("records")
    df["impact"] = [impact.score_features(f)[0] for f in feats]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="full", choices=["full", "lite"])
    args = ap.parse_args()
    path = OUTPUT / f"signals_history_{args.tier}.csv.gz"
    df = pd.read_csv(path)              # file order = engine emission order
    df = rescore(add_attention(df))
    df.to_csv(path, index=False, compression="gzip")
    print(f"{path.name}: {len(df):,} signals rescored ({impact.weights_source()} weights); "
          f"impact>7: {(df['impact'] > 7).mean():.2%}")
    if args.tier == "full" and (OUTPUT / "demo_signals.jsonl.gz").exists():
        by_id = df.set_index("signal_id")
        out = []
        with gzip.open(OUTPUT / "demo_signals.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                s = RiskSignal.model_validate_json(line)
                if s.signal_id in by_id.index:
                    f = dict(s.impact_drivers.features)
                    f["attention"] = float(by_id.loc[s.signal_id, "f_attention"])
                    score, contrib, logit = impact.score_features(f)
                    s = s.model_copy(update={"impact_score": score, "impact_drivers": s.impact_drivers.model_copy(
                        update={"features": f, "contributions": contrib, "logit": logit})})
                out.append(s.model_dump_json())
        with gzip.open(OUTPUT / "demo_signals.jsonl.gz", "wt", encoding="utf-8") as fh:
            fh.writelines(line + "\n" for line in out)
        print(f"demo_signals.jsonl.gz: {len(out):,} signals rescored")
    print(json.dumps({"weights": impact.model_params()[1], "intercept": impact.model_params()[0]}))


if __name__ == "__main__":
    main()
