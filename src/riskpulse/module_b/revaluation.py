"""Revalue every position under a ShockSet (sensitivity-based, with full repricing for options).

  bonds        dP/P = -D_mod * dy + 0.5 * C * dy^2,  dy = rate shift + spread shift
  loans        dEL  = EAD * (PD' - PD) * LGD  (impairment through P&L); PD' = min(1, PD * multiplier)
  IRS          dV   = DV01 * d(rate) in bp           CDS  dV = CS01 * d(spread) in bp
  FX forwards  dV   = -N * dFX                       non-USD assets also carry FX translation P&L
  equity/TRS   dV   = delta * d(equity)              index puts: Black-Scholes repricing, vol up in sell-offs
  futures, commodity swaps: delta-one
P&L is split into rates / spread / credit / equity / fx / commodity components for attribution.
"""
from __future__ import annotations

import pandas as pd

from riskpulse.module_b.portfolio import bs_put, market
from riskpulse.module_b.scenarios import ShockSet
from riskpulse.universe import by_ticker

COMPONENTS = ("pnl_rates", "pnl_spread", "pnl_credit", "pnl_equity", "pnl_fx", "pnl_commodity")
VOL_UP_PER_EQUITY_DOWN = 1.5   # +1.5 vol points per 1% equity fall (VIX ~21 -> ~45 on the Apr-2025 -12% move)


def _sector_of(ticker: str, default: str) -> str:
    c = by_ticker().get(ticker)
    return c.sector if c else default


def revalue(pos: pd.DataFrame, s: ShockSet) -> pd.DataFrame:
    m = market()
    out = []
    for p in pos.itertuples():
        r = dict.fromkeys(COMPONENTS, 0.0)
        r["stressed_pd"] = p.pd
        fxs = s.fx_pct(p.currency)
        if p.asset_class == "Bond":
            sovereign = p.instrument == "Sovereign Bond"
            dr = s.rate_bp(p.currency) / 1e4
            ds = s.spread_bp(p.rating, p.sector, p.region, p.ticker, p.country, sovereign) / 1e4
            local = p.value / p.fx
            r["pnl_rates"] = local * (-p.mod_dur * dr) * p.fx
            total = local * (-p.mod_dur * (dr + ds) + 0.5 * p.convexity * (dr + ds) ** 2) * p.fx
            r["pnl_spread"] = total - r["pnl_rates"]
            r["pnl_fx"] = (p.value + total) * fxs
            r["stressed_pd"] = min(1.0, p.pd * s.pd_multiplier(p.sector, p.region, p.ticker))
        elif p.asset_class == "Loan":
            pd_new = min(1.0, p.pd * s.pd_multiplier(p.sector, p.region, p.ticker))
            r["stressed_pd"] = pd_new
            r["pnl_credit"] = -p.ead * (pd_new - p.pd) * p.lgd
            r["pnl_fx"] = (p.value + r["pnl_credit"]) * fxs
        elif p.asset_class == "Equity":
            r["pnl_equity"] = p.value * s.equity_pct(p.sector, p.ticker)
        elif p.instrument == "Interest Rate Swap":
            r["pnl_rates"] = p.dv01 * s.rate_bp(p.currency)
        elif p.cs01:
            r["pnl_spread"] = p.cs01 * s.spread_bp(p.rating, p.sector, "North America", p.ticker)
        elif p.instrument == "FX Forward":
            r["pnl_fx"] = p.delta * fxs
        elif p.instrument == "Equity Total Return Swap":
            r["pnl_equity"] = p.delta * s.equity_pct(_sector_of(p.ticker, p.sector), p.ticker)
        elif p.instrument == "Equity Index Put Option":
            eq = s.index_equity_pct
            vol = m.vol + VOL_UP_PER_EQUITY_DOWN * max(0.0, -eq)
            r0 = m.rates["USD"]
            before = bs_put(1.0, p.strike, p.years, r0, m.vol)[0]
            after = bs_put(1.0 + eq, p.strike, p.years, r0, vol)[0]
            r["pnl_equity"] = p.notional * (after - before)
        elif p.instrument == "Equity Index Future":
            r["pnl_equity"] = p.delta * s.index_equity_pct
        elif p.instrument == "Commodity Swap":
            r["pnl_commodity"] = p.delta * s.oil_pct
        r["pnl_total"] = sum(r[c] for c in COMPONENTS)
        out.append(r)
    res = pd.concat([pos.reset_index(drop=True), pd.DataFrame(out)], axis=1)
    res["value_after"] = res["value"] + res["pnl_total"]
    res["delta_el"] = -res["pnl_credit"]
    return res
