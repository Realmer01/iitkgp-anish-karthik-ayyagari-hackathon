"""Impact score validation and calibration by event study.

For every (company, event day D):
  abnormal return   AR_t = r_stock,t - r_SPY,t                       (market-adjusted model, beta = 1)
  event window      |CAR[D, D+1]|  close-to-close                     (date-only news: the reaction may
                                                                       land on D or D+1)
  forward window    |AR| from open(D+1) to close(D+2)                 strictly after the news is available
Scores compared: the impact score vs |sentiment| alone (the naive baseline the brief asks for) and
news volume. Calibration fits non-negative logistic weights on the first 60% of days to predict a
top-quintile move and tests on the last 40% (walk-forward, no peeking).
"""
from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from riskpulse.config import MODELS, OUTPUT, PRICES
from riskpulse.nlp.impact import FEATURES, model_params


def _trading_days(px: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(px.index)


def event_day(ts: pd.Timestamp, precision: str, days: pd.DatetimeIndex) -> pd.Timestamp | None:
    if precision == "day":
        d = pd.Timestamp(ts.tz_convert("America/Los_Angeles").date())
    else:
        et = ts.tz_convert("America/New_York")
        d = pd.Timestamp(et.date()) + (timedelta(days=1) if et.hour >= 16 else timedelta(0))
    i = days.searchsorted(d)
    return days[i] if i < len(days) else None


def ticker_days(tier: str = "full") -> pd.DataFrame:
    close = pd.read_csv(PRICES / "prices_daily.csv", index_col=0, parse_dates=True)
    opn = pd.read_csv(PRICES / "prices_daily_open.csv", index_col=0, parse_dates=True).reindex(close.index)
    days = _trading_days(close)
    ret = close.pct_change(fill_method=None)
    sig = pd.read_csv(OUTPUT / f"signals_history_{tier}.csv.gz")
    sig = sig[sig["entity_type"] == "company"].copy()
    sig["published_at"] = pd.to_datetime(sig["published_at"], utc=True, format="ISO8601")
    sig["D"] = [event_day(t, p, days) for t, p in zip(sig["published_at"], sig["time_precision"])]
    sig = sig.dropna(subset=["D"])
    sig["abs_sent"] = sig["sentiment"].abs()
    best = sig.sort_values("impact").groupby(["entity_id", "D"]).tail(1).set_index(["entity_id", "D"])
    agg = sig.groupby(["entity_id", "D"]).agg(impact=("impact", "max"), abs_sentiment=("abs_sent", "max"),
                                             volume=("impact", "size"))
    agg = agg.join(best[[f"f_{f}" for f in FEATURES]])
    pos = {d: i for i, d in enumerate(days)}
    rows = []
    for (t, d), r in agg.iterrows():
        i = pos[d]
        if t not in close.columns or i + 2 >= len(days):
            continue
        car = (ret[t].iloc[i] - ret["SPY"].iloc[i]) + (ret[t].iloc[i + 1] - ret["SPY"].iloc[i + 1])
        fwd = (close[t].iloc[i + 2] / opn[t].iloc[i + 1] - 1) - (close["SPY"].iloc[i + 2] / opn["SPY"].iloc[i + 1] - 1)
        if np.isfinite(car) and np.isfinite(fwd):
            rows.append({"ticker": t, "D": d, **r.to_dict(), "abs_car": abs(car), "abs_fwd": abs(fwd)})
    return pd.DataFrame(rows).sort_values("D").reset_index(drop=True)


def _hit_rate(score: pd.Series, move: pd.Series) -> float:
    top = score >= score.quantile(0.9)
    return float((move[top] >= move.quantile(0.9)).mean())


def _deciles(df: pd.DataFrame, col: str) -> pd.Series:
    q = pd.qcut(df[col].rank(method="first"), 10, labels=range(1, 11))
    return df.groupby(q, observed=True)["abs_car"].mean() * 100


def _fit_nonneg_logit(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> tuple[float, np.ndarray]:
    def loss(theta):
        z = theta[0] + X @ theta[1:]
        p = 1 / (1 + np.exp(-z))
        eps = 1e-9
        return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + l2 * np.sum(theta[1:] ** 2)
    theta0 = np.r_[-1.0, np.full(X.shape[1], 0.5)]
    res = minimize(loss, theta0, method="L-BFGS-B", bounds=[(None, None)] + [(0, None)] * X.shape[1])
    return float(res.x[0]), res.x[1:]


def evaluate(tier: str = "full", write_calibration: bool = True) -> dict:
    df = ticker_days(tier)
    table = []
    for name, col in (("Impact score (RiskPulse)", "impact"), ("|Sentiment| only (naive)", "abs_sentiment"),
                      ("News volume", "volume")):
        table.append({"score": name,
                      "spearman_event_window": round(float(spearmanr(df[col], df["abs_car"])[0]), 4),
                      "spearman_forward": round(float(spearmanr(df[col], df["abs_fwd"])[0]), 4),
                      "top_decile_hit_rate": round(_hit_rate(df[col], df["abs_car"]), 4)})
    dec = pd.DataFrame({"decile": range(1, 11), "impact_mean_abs_ar": _deciles(df, "impact").values,
                        "baseline_mean_abs_ar": _deciles(df, "abs_sentiment").values}).round(3)

    # ---- walk-forward calibration of the feature weights
    cut = df["D"].quantile(0.6)
    tr, te = df[df["D"] <= cut], df[df["D"] > cut]
    feats = [f"f_{f}" for f in FEATURES]
    y_tr = (tr["abs_car"] >= tr["abs_car"].quantile(0.8)).astype(float).to_numpy()
    b, w = _fit_nonneg_logit(tr[feats].to_numpy(), y_tr)
    b0, w0, _ = model_params()
    expert_te = te[feats].to_numpy() @ np.array([w0.get(f, 0) for f in FEATURES]) + b0
    calib_te = te[feats].to_numpy() @ w + b
    rho_expert = float(spearmanr(expert_te, te["abs_car"])[0])
    rho_calib = float(spearmanr(calib_te, te["abs_car"])[0])
    rho_base = float(spearmanr(te["abs_sentiment"], te["abs_car"])[0])
    adopt = rho_calib > rho_expert + 0.01
    calib = {"intercept": round(b, 4), "weights": {f: round(float(x), 4) for f, x in zip(FEATURES, w)},
             "target": "top-quintile |CAR[D, D+1]| on the training period", "train_end": str(cut.date()),
             "test_spearman": {"expert": round(rho_expert, 4), "calibrated": round(rho_calib, 4),
                               "abs_sentiment": round(rho_base, 4)}, "adopted": adopt}
    if write_calibration:
        path = MODELS / "impact_calibrated.json"
        if adopt:
            path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        elif path.exists():
            path.unlink()
    return {"n_ticker_days": int(len(df)), "period": [str(df["D"].min().date()), str(df["D"].max().date())],
            "table": table, "deciles": dec.to_dict(orient="records"), "calibration": calib,
            "note": "Market-adjusted abnormal returns from data/prices. Event window [D, D+1] because most archive "
                    "timestamps are date-only. Forward window opens after the news is public (next open). "
                    f"Calibration {'adopted' if adopt else 'not adopted (did not beat the expert weights out of sample)'}."}
