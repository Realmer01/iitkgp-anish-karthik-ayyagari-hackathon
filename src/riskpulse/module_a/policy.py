"""Module A weighting policy - pure functions shared by the live rebalancer and the backtest.

    target_i = b_i * exp(kappa * s_i) / sum_j b_j * exp(kappa * s_j)        sentiment tilt
    subject to   max(min_w, b_i - max_active) <= w_i <= min(max_w, b_i + max_active),  sum w = 1, w >= 0
    trade only if some |target_i - current_i| > min(drift_band, drift_band_rel * b_i)   no churn on noise
    one-way turnover per rebalance <= max_turnover; cost = turnover x cost_bps

b_i are capped market-cap base weights (the neutral index), s_i in [-1, 1] the decayed sentiment.
kappa = 0.8: a maximally positive name gets e^0.8 = 2.2x the weight of a neutral one, before limits.
The numerical core works on numpy arrays (the backtest calls it ~3,500 times per run); thin pandas
wrappers keep the tickers attached for the live module.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from riskpulse import config
from riskpulse.universe import companies


@dataclass(frozen=True)
class PolicyParams:
    kappa: float = 0.8
    min_weight: float = 0.005
    max_weight: float = 0.18
    max_active_weight: float = 0.04
    drift_band: float = 0.005
    drift_band_rel: float = 0.25
    max_turnover: float = 0.10
    cost_bps: float = 5.0
    base_cap: float = 0.15
    base_floor: float = 0.01
    base_weighting: str = "capped_market_cap"

    @classmethod
    def from_config(cls, **over) -> PolicyParams:
        c = config.get("module_a", {})
        p = cls(kappa=c.get("kappa", 0.8), min_weight=c.get("min_weight", 0.005), max_weight=c.get("max_weight", 0.18),
                max_active_weight=c.get("max_active_weight", 0.04), drift_band=c.get("drift_band", 0.005),
                drift_band_rel=c.get("drift_band_relative", 0.25),
                max_turnover=c.get("max_turnover_per_rebalance", 0.10), cost_bps=c.get("transaction_cost_bps", 5),
                base_cap=c.get("base_cap", 0.15), base_floor=c.get("base_floor", 0.01),
                base_weighting=c.get("base_weighting", "capped_market_cap"))
        return replace(p, **over)


# ---------------------------------------------------------------------------- numpy core
def project_np(raw: np.ndarray, lo: np.ndarray, hi: np.ndarray, iters: int = 60) -> np.ndarray:
    """w = clip(lambda * raw, lo, hi) with lambda found by bisection so that sum w = 1.

    Multiplicative scaling keeps the ranking of raw weights: a name pushed to its floor by negative
    sentiment stays there instead of receiving redistributed weight."""
    raw = np.maximum(raw, 1e-12)
    if lo.sum() >= 1:
        return lo / lo.sum()
    a, b = 0.0, 1.0
    while np.clip(raw * b, lo, hi).sum() < 1 and b < 1e12:
        b *= 2
    for _ in range(iters):
        mid = (a + b) / 2
        if np.clip(raw * mid, lo, hi).sum() < 1:
            a = mid
        else:
            b = mid
    w = np.clip(raw * b, lo, hi)
    return w / w.sum()


def bounds_np(base: np.ndarray, p: PolicyParams) -> tuple[np.ndarray, np.ndarray]:
    lo = np.minimum(np.maximum(p.min_weight, base - p.max_active_weight), base)
    hi = np.maximum(np.minimum(p.max_weight, base + p.max_active_weight), base)
    return lo, hi


def target_np(base: np.ndarray, sentiment: np.ndarray, p: PolicyParams, lo, hi) -> np.ndarray:
    return project_np(base * np.exp(p.kappa * np.clip(sentiment, -1, 1)), lo, hi)


def rebalance_np(current: np.ndarray, target: np.ndarray, p: PolicyParams, lo, hi,
                 base: np.ndarray) -> tuple[np.ndarray, float]:
    """Tolerance band is absolute for large names and relative for small ones (a 1% name can move 25bp)."""
    gap = target - current
    if (np.abs(gap) <= np.minimum(p.drift_band, p.drift_band_rel * base)).all():
        return current, 0.0
    turnover = np.abs(gap).sum() / 2
    step = gap * min(1.0, p.max_turnover / turnover) if turnover > 0 else gap
    new = project_np(current + step, lo, hi)
    return new, float(np.abs(new - current).sum() / 2)


# ---------------------------------------------------------------------------- pandas wrappers
def base_weights(params: PolicyParams | None = None) -> pd.Series:
    params = params or PolicyParams.from_config()
    cos = companies()
    idx = [c.ticker for c in cos]
    if params.base_weighting == "equal":
        return pd.Series(1 / len(idx), index=idx)
    caps = np.array([c.market_cap_bn for c in cos], dtype=float)
    n = len(idx)
    w = project_np(caps / caps.sum(), np.full(n, min(params.base_floor, 1 / n)), np.full(n, max(params.base_cap, 1 / n)))
    return pd.Series(w, index=idx)


def bounds(base: pd.Series, params: PolicyParams) -> tuple[pd.Series, pd.Series]:
    lo, hi = bounds_np(base.values, params)
    return pd.Series(lo, index=base.index), pd.Series(hi, index=base.index)


def target_weights(base: pd.Series, sentiment: pd.Series, params: PolicyParams) -> pd.Series:
    lo, hi = bounds_np(base.values, params)
    s = sentiment.reindex(base.index).fillna(0.0).values
    return pd.Series(target_np(base.values, s, params, lo, hi), index=base.index)


def rebalance(current: pd.Series, target: pd.Series, base: pd.Series, params: PolicyParams) -> tuple[pd.Series, float]:
    """Returns (new weights, one-way turnover). No trade unless drift exceeds the band."""
    lo, hi = bounds_np(base.values, params)
    new, to = rebalance_np(current.values, target.reindex(base.index).values, params, lo, hi, base.values)
    return pd.Series(new, index=base.index), to
