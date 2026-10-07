"""Download the public datasets RiskPulse uses into data/ (outputs are committed; re-run to refresh).

    python scripts/fetch_data.py            # corpora + prices
    python scripts/fetch_data.py --prices   # market data only

Sources (all free, no API key, no proprietary data):
  * zeroshot/twitter-financial-news-topic      (MIT)  labelled finance tweets, 20 topics
  * zeroshot/twitter-financial-news-sentiment  (MIT)  labelled finance tweets, 3 sentiments
  * Financial PhraseBank v1.0, Malo et al. 2014 (CC BY-NC-SA 3.0)  labelled news sentences
  * Yahoo Finance via yfinance: daily (2y) and hourly (~730d) bars for the universe and benchmarks
"""
from __future__ import annotations

import argparse
import io
import zipfile

import pandas as pd
import requests

import _bootstrap  # noqa: F401
from riskpulse.config import PRICES, RAW
from riskpulse.universe import tickers

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"
HF_FILES = {
    "twitter_topic_train.csv": ("zeroshot/twitter-financial-news-topic", "topic_train.csv"),
    "twitter_topic_valid.csv": ("zeroshot/twitter-financial-news-topic", "topic_valid.csv"),
    "twitter_sentiment_train.csv": ("zeroshot/twitter-financial-news-sentiment", "sent_train.csv"),
    "twitter_sentiment_valid.csv": ("zeroshot/twitter-financial-news-sentiment", "sent_valid.csv"),
}
PHRASEBANK = ("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip")
BENCHMARKS = ["SPY", "^TNX", "^VIX", "CL=F", "HYG", "LQD"]  # S&P 500, UST 10y, VIX, WTI, HY / IG credit


def fetch_corpora() -> None:
    for out, (repo, file) in HF_FILES.items():
        r = requests.get(HF.format(repo=repo, file=file), timeout=60)
        r.raise_for_status()
        (RAW / out).write_bytes(r.content)
        print(f"  {out}: {len(pd.read_csv(RAW / out)):,} rows")
    r = requests.get(HF.format(repo=PHRASEBANK[0], file=PHRASEBANK[1]), timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith("Sentences_AllAgree.txt"))
        lines = zf.read(name).decode("latin-1").strip().splitlines()
    df = pd.DataFrame([ln.rsplit("@", 1) for ln in lines if "@" in ln], columns=["text", "label"])
    df.to_csv(RAW / "phrasebank_allagree.csv", index=False)
    print(f"  phrasebank_allagree.csv: {len(df):,} rows")


def fetch_prices() -> None:
    import yfinance as yf

    symbols = tickers() + BENCHMARKS
    daily = yf.download(symbols, period="2y", interval="1d", auto_adjust=True, progress=False)["Close"]
    daily.index = pd.to_datetime(daily.index).date
    daily.index.name = "date"
    daily.dropna(how="all").round(4).to_csv(PRICES / "prices_daily.csv")
    print(f"  prices_daily.csv: {daily.shape[0]} days x {daily.shape[1]} series")

    # Open prices let the backtest earn open-to-open returns: weights chosen at a bar's start never earn
    # an overnight gap that happened before they could have been traded.
    raw = yf.download(tickers() + ["SPY"], period="730d", interval="1h", auto_adjust=True, progress=False)
    for field in ("Open", "Close"):
        bars = raw[field].copy()
        bars.index = pd.to_datetime(bars.index, utc=True)
        bars.index.name = "bar_start_utc"
        bars.dropna(how="all").round(3).to_csv(PRICES / f"prices_hourly_{field.lower()}.csv.gz", compression="gzip")
    print(f"  prices_hourly_{{open,close}}.csv.gz: {raw.shape[0]} bars ({raw.index.min()} .. {raw.index.max()})")
    opens = yf.download(tickers() + ["SPY"], period="2y", interval="1d", auto_adjust=True, progress=False)["Open"]
    opens.index = pd.to_datetime(opens.index).date
    opens.index.name = "date"
    opens.dropna(how="all").round(4).to_csv(PRICES / "prices_daily_open.csv")

    caps = []
    for t in tickers():
        info = yf.Ticker(t).fast_info
        caps.append({"ticker": t, "market_cap_usd": float(info["market_cap"]), "shares": float(info["shares"])})
    pd.DataFrame(caps).to_csv(PRICES / "market_caps.csv", index=False)
    print(f"  market_caps.csv: {len(caps)} tickers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", action="store_true", help="only refresh market data")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    PRICES.mkdir(parents=True, exist_ok=True)
    if not args.prices:
        print("Labelled NLP corpora:")
        fetch_corpora()
    print("Market data:")
    fetch_prices()


if __name__ == "__main__":
    main()
