"""Cache recent GDELT DOC 2.0 headlines for the universe and macro themes.

The DOC API only searches a rolling ~3-month window and allows one request per 5 seconds, so the
script walks that window in slices with polite spacing, exponential backoff on throttling and
connection resets, and resumes from finished (query, slice) pairs on re-run.
Output: data/raw/gdelt_headlines.csv.gz        python scripts/fetch_gdelt.py --days 90
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

import _bootstrap  # noqa: F401
import pandas as pd
import requests

from riskpulse.config import RAW
from riskpulse.ingestion.news import GDELT_URL, parse_gdelt_articles, short_name
from riskpulse.universe import companies

THEMES = {"MACRO": '(inflation OR "interest rates" OR "central bank" OR recession)',
          "GEO": "(sanctions OR tariffs OR invasion OR missile OR ceasefire)",
          "CREDIT": '(bankruptcy OR default OR downgrade OR "credit rating")'}
OUT = RAW / "gdelt_headlines.csv.gz"


def query_for(name: str) -> str:
    short = short_name(name).replace(".com", "")
    # quote multi-word or punctuated names: GDELT reads a bare hyphen ("Coca-Cola") as NOT
    return short if short.isalnum() else f'"{short}"'


def get(session: requests.Session, params: dict, spacing: float) -> list[dict] | None:
    for attempt in range(6):
        time.sleep(spacing * (1.6 ** attempt))
        try:
            r = session.get(GDELT_URL, params=params, timeout=40)
        except requests.RequestException as exc:
            print(f"    network error: {exc.__class__.__name__}")
            continue
        if r.status_code == 200 and r.text.lstrip().startswith("{"):
            return r.json().get("articles", [])
        if r.status_code == 200:          # GDELT answers query errors with a 200 text message
            print(f"    query rejected: {r.text[:70]!r}")
            return []
        print(f"    throttled ({r.status_code})")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--slice", type=int, default=15)
    ap.add_argument("--spacing", type=float, default=8.0)
    args = ap.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (RiskPulse research prototype)"
    prior = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["query_key", "slice"])
    done = set(zip(prior["query_key"], prior["slice"]))
    rows = prior.to_dict("records")
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    names = [query_for(c.name) for c in companies()]
    # GDELT throttles hard per IP, so companies are grouped into OR-queries (fewer, larger requests).
    jobs = [(f"COMPANIES_{i // 6 + 1}", f"({' OR '.join(names[i:i + 6])})") for i in range(0, len(names), 6)]
    jobs += list(THEMES.items())
    for d in range(0, args.days, args.slice):
        hi, lo = end - timedelta(days=d), end - timedelta(days=min(d + args.slice, args.days))
        tag = lo.strftime("%Y-%m-%d")
        for key, q in jobs:
            if (key, tag) in done:
                continue
            arts = get(session, {"query": f"{q} sourcelang:english", "mode": "artlist", "format": "json",
                                 "maxrecords": 100, "sort": "hybridrel", "startdatetime": lo.strftime("%Y%m%d%H%M%S"),
                                 "enddatetime": hi.strftime("%Y%m%d%H%M%S")}, args.spacing)
            if arts is None:
                print(f"  {tag} {key}: gave up for now (re-run to resume)")
                continue
            for doc in parse_gdelt_articles(arts):
                rows.append({"query_key": key, "slice": tag, "published_at": doc.published_at.isoformat(),
                             "publisher": doc.publisher, "title": doc.title, "country": doc.raw_meta.get("country", "")})
            if not arts:  # remember empty slices too, so they are not re-queried
                rows.append({"query_key": key, "slice": tag})
            print(f"  {tag} {key}: {len(arts)}", flush=True)
        df = pd.DataFrame(rows).drop_duplicates()
        df.sort_values("published_at", na_position="last").to_csv(OUT, index=False, compression="gzip")
    print(f"saved {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
