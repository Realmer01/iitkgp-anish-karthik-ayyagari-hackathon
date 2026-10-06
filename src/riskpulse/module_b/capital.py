"""Simplified capital proxy - clearly NOT a regulatory calculation.

RWA = credit RWA (loans: EAD x Basel-style standardised risk weight by rating; bonds: value x corporate /
sovereign weight) + equity RWA (250%) + counterparty RWA for derivatives (1.4 x (max(MTM, 0) + add-on) x
dealer weight, add-ons in the spirit of SA-CCR). Under stress, PDs migrate ratings (nearest rating by PD,
in log space) so risk weights rise; losses reduce CET1 one-for-one (pre-tax, conservative).
"""
from __future__ import annotations

import math

import pandas as pd

from riskpulse import config
from riskpulse.module_b.portfolio import market

ADD_ON = {"Interest Rate Swap": 0.005, "FX Forward": 0.04, "Credit Default Swap": 0.05, "CDS Index": 0.05,
          "Equity Total Return Swap": 0.32, "Equity Index Put Option": 0.20, "Equity Index Future": 0.20,
          "Commodity Swap": 0.18}
DEALER_RW = 0.50


def rating_for_pd(pd_value: float) -> str:
    table = market().ratings["pd_1y"]
    return min(table.index, key=lambda r: abs(math.log(max(pd_value, 1e-6)) - math.log(table[r])))


def rwa(pos: pd.DataFrame, stressed: bool = False) -> float:
    ratings = market().ratings
    total = 0.0
    for p in pos.itertuples():
        rating = rating_for_pd(p.stressed_pd) if stressed and p.asset_class in ("Loan", "Bond") else p.rating
        if p.asset_class == "Loan":
            total += p.ead * ratings.loc[rating, "rw_corporate"]
        elif p.asset_class == "Bond":
            col = "rw_sovereign" if p.instrument == "Sovereign Bond" else "rw_corporate"
            value = p.value_after if stressed else p.value
            total += max(value, 0) * ratings.loc[rating, col]
        elif p.asset_class == "Equity":
            total += max(p.value_after if stressed else p.value, 0) * 2.5
        else:
            mtm = p.value_after if stressed else p.value
            years = max(p.years, 1.0) if p.instrument == "Interest Rate Swap" else 1.0
            total += 1.4 * (max(mtm, 0) + abs(p.notional) * ADD_ON.get(p.instrument, 0.05) * years) * DEALER_RW
    return total


def capital_impact(res: pd.DataFrame) -> dict:
    base_ratio = float(config.get("module_b.baseline_cet1_ratio", 0.13))
    minimum = float(config.get("module_b.cet1_minimum_ratio", 0.105))
    rwa0, rwa1 = rwa(res), rwa(res, stressed=True)
    cet1_0 = base_ratio * rwa0
    cet1_1 = cet1_0 + float(res["pnl_total"].sum())
    return {"rwa_before": rwa0, "rwa_after": rwa1, "cet1_before": cet1_0, "cet1_after": cet1_1,
            "cet1_ratio_before": cet1_0 / rwa0, "cet1_ratio_after": cet1_1 / rwa1, "minimum_ratio": minimum,
            "headroom_bp_after": (cet1_1 / rwa1 - minimum) * 1e4, "breach": cet1_1 / rwa1 < minimum,
            "note": "simplified standardised-style proxy, not a regulatory calculation"}
