"""Run the RiskEngine over the historical headline archive, in time order, exactly as replay would.

    python scripts/precompute_signals.py --tier full     (~40 min CPU)   python ... --tier lite  (~1 min)

Outputs (data/output/):
  signals_history_<tier>.csv.gz  compact table of every signal + impact features (backtest, calibration)
  demo_signals.jsonl.gz          full RiskSignal JSON for the demo window (offline `run.py demo`)
"""
from __future__ import annotations

import argparse
import gzip
import time

import pandas as pd

import _bootstrap  # noqa: F401
from riskpulse import config
from riskpulse.config import OUTPUT, RAW
from riskpulse.engine import RiskEngine
from riskpulse.nlp.impact import FEATURES
from riskpulse.schema import RawDocument, SourceType


def load_history() -> list[RawDocument]:
    frames = [pd.read_csv(RAW / "news_history.csv.gz").assign(source="google_news")]
    if (RAW / "gdelt_headlines.csv.gz").exists():
        frames.append(pd.read_csv(RAW / "gdelt_headlines.csv.gz").assign(source="gdelt"))
    df = pd.concat(frames, ignore_index=True).dropna(subset=["title"])
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, format="ISO8601")
    df = df.sort_values("published_at").drop_duplicates(["title", "publisher"])
    # Google News gives most items a date only (stamped midnight US Pacific = 07:00/08:00 UTC).
    day_only = (df["source"] == "google_news") & df["published_at"].dt.strftime("%M:%S").eq("00:00") \
        & df["published_at"].dt.hour.isin([7, 8])
    return [RawDocument(source=r.source, source_type=SourceType.NEWS, publisher=str(r.publisher),
                        title=str(r.title), published_at=r.published_at.to_pydatetime(),
                        raw_meta={"time_precision": "day" if d else "exact"})
            for r, d in zip(df.itertuples(), day_only)]


def compact(s) -> dict:
    row = {"published_at": s.published_at.isoformat(), "signal_id": s.signal_id, "story_id": s.story_id,
           "source": s.source, "publisher": s.publisher, "entity_type": s.entity.type, "entity_id": s.entity.id,
           "sector": s.entity.sector or "", "event_type": s.event_type, "event_conf": s.event_confidence,
           "event_rule": s.event_rule or "", "sentiment": s.sentiment_score, "sent_conf": s.sentiment_confidence,
           "impact": s.impact_score, "mention_count": s.mention_count, "n_sources": len(s.sources),
           "regions": "|".join(s.regions), "headline": s.headline[:120]}
    row.update({f"f_{k}": s.impact_drivers.features[k] for k in FEATURES})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="full", choices=["full", "lite"])
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    docs = load_history()
    precision = {d.id: d.raw_meta["time_precision"] for d in docs}
    engine = RiskEngine(args.tier)
    demo_lo, demo_hi = (pd.Timestamp(config.get(f"demo.{k}")) for k in ("start", "end"))
    rows, demo, t0 = [], [], time.time()
    for i in range(0, len(docs), args.batch):
        for s in engine.process(docs[i:i + args.batch]):
            rows.append({**compact(s), "time_precision": precision[s.doc_ids[0]]})
            if demo_lo <= pd.Timestamp(s.published_at) < demo_hi:
                demo.append(s.model_dump_json())
        if (i // args.batch) % 100 == 0:
            print(f"  {i:,}/{len(docs):,} docs, {len(rows):,} signals, {time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT / f"signals_history_{engine.tier}.csv.gz", index=False, compression="gzip")
    if engine.tier == "full" or not (OUTPUT / "demo_signals.jsonl.gz").exists():
        with gzip.open(OUTPUT / "demo_signals.jsonl.gz", "wt", encoding="utf-8") as fh:
            fh.writelines(line + "\n" for line in demo)
    print(f"{len(docs):,} docs -> {len(out):,} signals ({engine.tier}) in {time.time() - t0:.0f}s; "
          f"{len(demo):,} demo-window signals; stats {dict(engine.stats)}")


if __name__ == "__main__":
    main()
