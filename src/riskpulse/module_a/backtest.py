"""Module A backtest on hourly bars with strict no-look-ahead.

Timing rules
  * a signal is usable at `available_at`: its timestamp when the source gives one, otherwise (date-only
    Google News items, stamped midnight US Pacific) the END of that Pacific day, i.e. published_at + 24h;
  * at each bar start the strategy sees only signals with available_at < bar_start, sets weights and
    earns the open-to-open return to the next bar - it never earns a move that happened before it could trade;
  * weights drift with prices between rebalances; costs = one-way turnover x cost_bps.
Benchmarks
  * neutral_policy - the SAME rebalancer with kappa = 0 (no sentiment): isolates the value of the signal
    from side effects of the drift band / caps;
  * static_cap and equal_weight - reset to their weights at every day's first bar.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd

from riskpulse.aggregate import SECTOR_SPILLOVER, SentimentBook, evidence
from riskpulse.config import OUTPUT, PRICES
from riskpulse.module_a.policy import PolicyParams, base_weights, bounds_np, rebalance_np, target_np
from riskpulse.universe import sector_members, tickers


def load_signals(tier: str = "full") -> pd.DataFrame:
    df = pd.read_csv(OUTPUT / f"signals_history_{tier}.csv.gz")
    df = df[df["entity_type"].isin(["company", "sector"])].copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, format="ISO8601")
    day = df.get("time_precision", pd.Series("exact", index=df.index)).eq("day").to_numpy()
    df["available_at"] = df["published_at"] + pd.to_timedelta(np.where(day, 24 * 60, 1), unit="min")
    return df.sort_values("available_at").reset_index(drop=True)


def load_bars() -> pd.DataFrame:
    px = pd.read_csv(PRICES / "prices_hourly_open.csv.gz", index_col=0)
    px.index = pd.to_datetime(px.index, utc=True)
    return px[tickers()].ffill().dropna()


def signal_events(sig: pd.DataFrame, min_conf: float) -> list[tuple]:
    sectors, out = sector_members(), []
    for r in sig[sig["sent_conf"] >= min_conf].itertuples():
        e = evidence(r.sent_conf, r.f_credibility, r.impact, r.mention_count)
        targets = [r.entity_id] if r.entity_type == "company" else sectors.get(r.entity_id, [])
        mult = 1.0 if r.entity_type == "company" else SECTOR_SPILLOVER
        out += [(r.available_at, t, r.sentiment, e * mult, r.headline) for t in targets]
    return out


def _simulate(rets: pd.DataFrame, events: list[tuple], p: PolicyParams, half_life: float | None):
    base = base_weights(p).reindex(rets.columns).to_numpy()
    lo, hi = bounds_np(base, p)
    book = SentimentBook(half_life_hours=half_life)
    order = {t: i for i, t in enumerate(book.tickers)}
    perm = [order[c] for c in rets.columns]
    w, j = base.copy(), 0
    n = len(rets)
    port, turn = np.zeros(n), np.zeros(n)
    W, S = np.zeros((n, len(base))), np.zeros((n, len(base)))
    for i, (t, r) in enumerate(zip(rets.index, rets.to_numpy())):
        while j < len(events) and events[j][0] < t:               # strictly before the bar starts
            ts, tick, s, e, _ = events[j]
            book.add(tick, s, e, ts.to_pydatetime())
            j += 1
        sent = np.asarray(book.vector(t.to_pydatetime()))[perm]
        w, to = rebalance_np(w, target_np(base, sent, p, lo, hi), p, lo, hi, base)
        cost = to * p.cost_bps / 1e4
        gross = float(w @ r)
        port[i], turn[i], W[i], S[i] = gross - cost, to, w, sent
        w = w * (1 + r) / (1 + gross)                              # drift with prices
    return port, turn, W, S


def _daily_reset(rets: pd.DataFrame, w0: np.ndarray) -> np.ndarray:
    first = ~rets.index.normalize().duplicated()
    out, w = np.zeros(len(rets)), w0.copy()
    for i, (r, fb) in enumerate(zip(rets.to_numpy(), first)):
        if fb:
            w = w0.copy()
        out[i] = float(w @ r)
        w = w * (1 + r) / (1 + out[i])
    return out


def run(signals: pd.DataFrame, bars: pd.DataFrame, params: PolicyParams | None = None,
        half_life_hours: float | None = None, start: str = "2024-10-15", benchmarks: bool = True) -> dict:
    p = params or PolicyParams.from_config()
    bars = bars[bars.index >= pd.Timestamp(start, tz="UTC")]
    rets = (bars.shift(-1) / bars - 1).iloc[:-1].fillna(0.0)
    events = signal_events(signals, SentimentBook().min_conf)
    port, turn, W, S = _simulate(rets, events, p, half_life_hours)
    out = pd.DataFrame({"strategy": port, "turnover": turn}, index=rets.index)
    if benchmarks:
        out["neutral_policy"], out["neutral_turnover"], *_ = _simulate(rets, [], replace(p, kappa=0.0), half_life_hours)
        base = base_weights(p).reindex(rets.columns).to_numpy()
        out["static_cap"] = _daily_reset(rets, base)
        out["equal_weight"] = _daily_reset(rets, np.full(len(base), 1 / len(base)))
    return {"returns": out, "weights": pd.DataFrame(W, index=rets.index, columns=rets.columns),
            "sentiment": pd.DataFrame(S, index=rets.index, columns=rets.columns), "events": events}


def metrics(ret: pd.Series, bench: pd.Series | None = None, turnover: pd.Series | None = None) -> dict:
    years = (ret.index[-1] - ret.index[0]).days / 365.25
    per_year = len(ret) / years
    eq = (1 + ret).cumprod()
    m = {"cumulative_return": float(eq.iloc[-1] - 1), "annual_return": float(eq.iloc[-1] ** (1 / years) - 1),
         "annual_volatility": float(ret.std() * np.sqrt(per_year)), "max_drawdown": float((eq / eq.cummax() - 1).min())}
    m["sharpe_rf0"] = m["annual_return"] / m["annual_volatility"] if m["annual_volatility"] else 0.0
    if turnover is not None:
        m["annual_turnover"] = float(turnover.sum() / years)
    if bench is not None:
        active = ret - bench
        daily = (1 + active).groupby(active.index.normalize()).prod() - 1
        te = float(active.std() * np.sqrt(per_year))
        m.update({"active_return_annual": float(active.mean() * per_year), "tracking_error": te,
                  "information_ratio": float(active.mean() * per_year / te) if te else 0.0,
                  "daily_hit_rate": float((daily[daily != 0] > 0).mean()) if (daily != 0).any() else 0.0})
    return m


def information_coefficient(res: dict, bars: pd.DataFrame) -> dict:
    """Daily cross-sectional Spearman IC: sentiment at the day's first bar vs open-to-next-open return."""
    sent = res["sentiment"]
    first = sent.groupby(sent.index.normalize()).head(1)
    opens = bars.groupby(bars.index.normalize()).head(1)
    opens.index = opens.index.normalize()
    fwd = (opens.shift(-1) / opens - 1)[sent.columns]
    ics = []
    for t, row in first.iterrows():
        d = t.normalize()
        if d in fwd.index and not fwd.loc[d].isna().any() and np.abs(row.values).sum() > 0:
            ic = pd.Series(row.values).rank().corr(pd.Series(fwd.loc[d].values).rank())
            if not np.isnan(ic):
                ics.append(ic)
    ics = np.asarray(ics)
    return {"days": int(len(ics)), "mean_ic": float(ics.mean()) if len(ics) else 0.0,
            "ic_t_stat": float(ics.mean() / ics.std() * np.sqrt(len(ics))) if len(ics) > 2 and ics.std() else 0.0,
            "share_positive_days": float((ics > 0).mean()) if len(ics) else 0.0}


def sensitivity(signals: pd.DataFrame, bars: pd.DataFrame, kappas=(0.4, 0.8, 1.2, 2.0),
                half_lives=(4, 12, 24, 72)) -> pd.DataFrame:
    rows = []
    neutral = run(signals, bars)["returns"]["neutral_policy"]
    for k in kappas:
        for hl in half_lives:
            r = run(signals, bars, PolicyParams.from_config(kappa=k), half_life_hours=hl, benchmarks=False)["returns"]
            m = metrics(r["strategy"], neutral, r["turnover"])
            rows.append({"kappa": k, "half_life_hours": hl, **{x: round(m[x], 4) for x in (
                "active_return_annual", "information_ratio", "annual_turnover", "max_drawdown")}})
    return pd.DataFrame(rows)


def window(res: dict, start: str, end: str) -> dict:
    r = res["returns"]
    sel = r[(r.index >= pd.Timestamp(start, tz="UTC")) & (r.index < pd.Timestamp(end, tz="UTC") + timedelta(days=1))]
    cols = [c for c in ("strategy", "neutral_policy", "static_cap", "equal_weight") if c in sel]
    return {k: float((1 + sel[k]).prod() - 1) for k in cols}
