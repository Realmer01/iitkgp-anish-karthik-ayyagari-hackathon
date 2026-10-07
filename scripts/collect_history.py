"""Collect two years of real, dated headlines for backtesting and impact validation.

Google News RSS supports date-bounded search (after:/before:), so for every week we query
  * each index constituent  ("Nvidia stock", "JPMorgan stock", ...)
  * a set of macro / geopolitical / credit themes
and keep publisher, timestamp and headline (no article bodies). Resumable: finished
(query, week) pairs are skipped on re-run. Output: data/raw/news_history.csv.gz

    python scripts/collect_history.py --weeks 104
"""
from __future__ import annotations

import argparse
import random
import time
import zlib
from datetime import date, timedelta

import pandas as pd

import _bootstrap  # noqa: F401
from riskpulse.config import RAW
from riskpulse.ingestion.news import GoogleNewsSource
from riskpulse.universe import companies

OUT = RAW / "news_history.csv.gz"
TICKER_QUERIES = {"AAPL": "Apple stock", "MSFT": "Microsoft stock", "NVDA": "Nvidia stock",
                  "GOOGL": "Alphabet OR Google stock", "META": "Meta stock", "AMZN": "Amazon stock",
                  "TSLA": "Tesla stock", "JPM": "JPMorgan stock", "GS": '"Goldman Sachs" stock',
                  "XOM": '"Exxon Mobil" OR ExxonMobil stock', "JNJ": '"Johnson & Johnson" stock',
                  "PFE": "Pfizer stock", "BA": "Boeing stock", "CAT": "Caterpillar stock",
                  "WMT": "Walmart stock", "KO": '"Coca-Cola" stock'}
THEMES = {"MACRO_FED": "federal reserve interest rates", "MACRO_ECON": "inflation recession economy",
          "GEO": "tariffs OR sanctions OR war markets", "CREDIT": "credit downgrade OR default OR bankruptcy",
          "MARKET": "stock market selloff OR rally", "ENERGY": "oil prices OPEC"}
PER_QUERY_CAP = {"ticker": 40, "theme": 50}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--end", type=str, default=date.today().isoformat())
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    done = set(zip(existing.get("query_key", []), existing.get("week", [])))

    end = date.fromisoformat(args.end)
    weeks = [end - timedelta(days=7 * (i + 1)) for i in range(args.weeks)]
    jobs = [(c.ticker, TICKER_QUERIES[c.ticker], "ticker") for c in companies()]
    jobs += [(k, q, "theme") for k, q in THEMES.items()]

    source = GoogleNewsSource()
    rows, n = [], 0
    todo = [(w, j) for w in weeks for j in jobs if (j[0], w.isoformat()) not in done]
    print(f"{len(todo)} queries to run ({len(done)} already cached)")
    for w, (key, q, kind) in todo:
        query = f"{q} after:{w.isoformat()} before:{(w + timedelta(days=7)).isoformat()}"
        for attempt in range(4):
            try:
                docs = source.search(query)
                break
            except Exception as exc:
                time.sleep(10 * (attempt + 1))
                print(f"  retry {key} {w}: {exc}")
        else:
            continue
        docs = [d for d in docs if w <= d.published_at.date() <= w + timedelta(days=7)]
        random.Random(zlib.crc32(f"{key}|{w}".encode())).shuffle(docs)  # reproducible sampling
        for d in docs[:PER_QUERY_CAP[kind]]:
            rows.append({"query_key": key, "query_kind": kind, "week": w.isoformat(),
                         "published_at": d.published_at.isoformat(), "publisher": d.publisher, "title": d.title})
        n += 1
        if n % 50 == 0:
            _flush(existing, rows)
            print(f"  {n}/{len(todo)} queries, {len(rows)} new headlines")
        time.sleep(1.2 + random.random())
    _flush(existing, rows)
    print("done")


def _flush(existing: pd.DataFrame, rows: list[dict]) -> None:
    df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    df = df.drop_duplicates(["query_key", "title", "published_at"])
    df.sort_values("published_at").to_csv(OUT, index=False, compression="gzip")


if __name__ == "__main__":
    main()
