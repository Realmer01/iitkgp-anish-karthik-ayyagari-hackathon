"""Module B portfolio: trade blotter -> positions with risk analytics.

Per asset class (all values in USD unless noted):
  Loan        EAD = drawn + CCF x undrawn (CCF 40% committed lines, 20% trade finance);
              12m expected loss EL = EAD x PD x LGD (LGD 35% secured / 45% unsecured); value = drawn - EL
  Bond        price from annual cash flows at yield = base rate + rating spread; modified duration
              and convexity from the same cash flows
  IRS         signed DV01 = notional x annuity x 1bp (+ for pay-fixed: gains when rates rise)
  CDS         CS01 = notional x risky annuity x 1bp (bought protection gains when spreads widen)
  FX forward  foreign-currency notional, value change = -N x FX move (bank sells foreign forward)
  Equity      market value; TRS = delta-one exposure; index puts priced with Black-Scholes
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from riskpulse.config import PORTFOLIO, PRICES

AS_OF = date(2026, 10, 6)
CCF = {"Revolving Credit Facility": 0.40, "Term Loan": 0.40, "Trade Finance": 0.20}
LGD = {"Senior Secured": 0.35, "Senior Unsecured": 0.45}
CDS_RECOVERY = 0.40
INDEX_RATING = {"I-CDX-IG": "A", "I-CDX-HY": "BB"}   # proxies for index spread level and shock


@dataclass(frozen=True)
class Market:
    rates: dict[str, float]
    fx: dict[str, float]
    ratings: pd.DataFrame
    vol: float

    def spread(self, rating: str) -> float:
        return float(self.ratings.loc[rating, "spread_bp"]) / 1e4

    def pd(self, rating: str) -> float:
        return float(self.ratings.loc[rating, "pd_1y"])


@lru_cache(maxsize=1)
def market() -> Market:
    mp = pd.read_csv(PORTFOLIO / "market_params.csv").set_index("currency")
    ratings = pd.read_csv(PORTFOLIO / "rating_params.csv").set_index("rating")
    vix = pd.read_csv(PRICES / "prices_daily.csv", index_col=0)["^VIX"].dropna().iloc[-1]
    return Market(mp["base_rate"].to_dict(), mp["usd_per_unit"].to_dict(), ratings, float(vix) / 100)


def years_to(maturity: str) -> float:
    if not isinstance(maturity, str) or not maturity:
        return 0.0
    return max((date.fromisoformat(maturity) - AS_OF).days / 365.25, 1 / 365)


def bond_analytics(coupon: float, y: float, t: float) -> tuple[float, float, float]:
    """Price (per 100), modified duration and convexity for annual coupons."""
    times = np.arange(t, 0, -1.0)[::-1]
    cfs = np.full(len(times), coupon * 100)
    cfs[-1] += 100
    disc = (1 + y) ** -times
    price = float((cfs * disc).sum())
    mod_dur = float((times * cfs * disc).sum() / price / (1 + y))
    convexity = float((times * (times + 1) * cfs * disc).sum() / price / (1 + y) ** 2)
    return price, mod_dur, convexity


def annuity(r: float, t: float) -> float:
    times = np.arange(t, 0, -1.0)
    return float(((1 + r) ** -times).sum()) if len(times) else t


def risky_annuity(r: float, spread: float, t: float) -> float:
    h = spread / (1 - CDS_RECOVERY)
    return (1 - math.exp(-(r + h) * t)) / (r + h)


def bs_put(s: float, k: float, t: float, r: float, vol: float) -> tuple[float, float, float]:
    """Black-Scholes put price, delta and gamma (spot normalised to 1)."""
    from scipy.stats import norm

    d1 = (math.log(s / k) + (r + vol ** 2 / 2) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    price = k * math.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
    return price, norm.cdf(d1) - 1, norm.pdf(d1) / (s * vol * math.sqrt(t))


def positions(blotter: pd.DataFrame | None = None) -> pd.DataFrame:
    tx = blotter if blotter is not None else pd.read_csv(PORTFOLIO / "transactions.csv")
    cp = pd.read_csv(PORTFOLIO / "counterparties.csv").set_index("cpty_id")
    m = market()
    rows = []
    for pid, g in tx.groupby("position_id", sort=False):
        f = g.iloc[0]
        fx, ccy = m.fx[f.currency], f.currency
        base = dict(position_id=pid, instrument=f.instrument, asset_class=f.asset_class, side=f.side,
                    cpty_id=f.cpty_id, counterparty=f.counterparty, sector=f.sector, country=f.country,
                    region=f.region, rating=f.rating, currency=ccy, fx=fx, maturity=f.maturity,
                    years=years_to(f.maturity), underlying=f.underlying if isinstance(f.underlying, str) else "",
                    ticker="", notional=0.0, ead=0.0, pd=0.0, lgd=0.0, el=0.0, mod_dur=0.0, convexity=0.0,
                    dv01=0.0, cs01=0.0, delta=0.0, gamma=0.0, value=0.0)
        if f.asset_class == "Loan":
            commit = g.loc[g.side == "Commitment", "notional"].sum()
            drawn = g.loc[g.side.isin(["Drawdown", "Repayment"]), "notional"].sum()
            ead = drawn + CCF[f.instrument] * max(commit - drawn, 0)
            lgd = LGD.get(f.seniority, 0.45)
            el = ead * m.pd(f.rating) * lgd
            base.update(notional=drawn * fx, ead=ead * fx, pd=m.pd(f.rating), lgd=lgd, el=el * fx,
                        value=(drawn - el) * fx, commitment=commit * fx, coupon=f.coupon_or_rate)
        elif f.asset_class == "Bond":
            face = g["notional"].sum()
            sovereign = f.instrument == "Sovereign Bond"
            y = m.rates[ccy] + (0.0 if sovereign else m.spread(f.rating))
            price, dur, conv = bond_analytics(f.coupon_or_rate, y, base["years"])
            base.update(notional=face * fx, value=face * price / 100 * fx, mod_dur=dur, convexity=conv,
                        price=price, yld=y, pd=m.pd(f.rating), coupon=f.coupon_or_rate,
                        ticker=cp.loc[f.cpty_id, "ticker"] if isinstance(cp.loc[f.cpty_id, "ticker"], str) else "")
        elif f.asset_class == "Equity":
            shares = g["notional"].sum()
            base.update(notional=shares * f.price, value=shares * f.price, delta=shares * f.price, ticker=f.underlying)
        else:
            n, r, t = float(f.notional), m.rates[ccy], base["years"]
            if f.instrument == "Interest Rate Swap":
                sign = 1.0 if f.side == "Pay Fixed" else -1.0
                a = annuity(r, t)
                base.update(notional=n * fx, dv01=sign * n * a * 1e-4 * fx, value=sign * (r - f.coupon_or_rate) * a * n * fx)
            elif f.instrument.startswith("CDS") or f.instrument == "Credit Default Swap":
                ref = f.underlying
                rating = INDEX_RATING.get(ref) or cp.loc[ref, "rating"]
                s = m.spread(rating)
                ra = risky_annuity(r, s, t)
                tick = cp.loc[ref, "ticker"] if isinstance(cp.loc[ref, "ticker"], str) else ""
                base.update(notional=n, cs01=n * ra * 1e-4, value=(s - f.coupon_or_rate) * ra * n, rating=rating,
                            ref_entity=ref, ticker=tick, sector=cp.loc[ref, "sector"] if ref in cp.index else f.sector)
            elif f.instrument == "FX Forward":
                base.update(notional=n * fx, delta=-n * fx)
            elif f.instrument == "Equity Total Return Swap":
                base.update(notional=n, delta=n, ticker=f.underlying)
            elif f.instrument == "Equity Index Put Option":
                p, d, gm = bs_put(1.0, f.coupon_or_rate, t, r, m.vol)
                base.update(notional=n, value=n * p, delta=n * d, gamma=n * gm, strike=f.coupon_or_rate)
            elif f.instrument == "Equity Index Future":
                base.update(notional=n, delta=-n)
            elif f.instrument == "Commodity Swap":
                base.update(notional=n, delta=-n)
        rows.append(base)
    df = pd.DataFrame(rows)
    return df.fillna({"ticker": "", "ref_entity": ""})


def summary(pos: pd.DataFrame) -> dict:
    return {"positions": int(len(pos)), "value_usd": float(pos["value"].sum()),
            "by_asset_class": pos.groupby("asset_class")["value"].sum().round(0).to_dict(),
            "loan_ead_usd": float(pos["ead"].sum()), "expected_loss_usd": float(pos["el"].sum()),
            "bond_dv01_usd": float((pos["value"] * pos["mod_dur"] * 1e-4).sum()),
            "swap_dv01_usd": float(pos["dv01"].sum()), "cds_cs01_usd": float(pos["cs01"].sum())}
