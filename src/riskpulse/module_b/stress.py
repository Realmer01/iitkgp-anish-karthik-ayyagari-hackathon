"""One stress-test run: shocks -> revaluation -> aggregation, capital and the naive comparison."""
from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from functools import lru_cache

import numpy as np
import pandas as pd

from riskpulse.module_b import scenarios
from riskpulse.module_b.capital import capital_impact
from riskpulse.module_b.portfolio import positions
from riskpulse.module_b.revaluation import COMPONENTS, revalue


@lru_cache(maxsize=1)
def book() -> pd.DataFrame:
    return positions()


def plain(obj):
    """Recursively convert numpy / pandas scalars so results serialise to JSON anywhere."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _group(res: pd.DataFrame, col: str) -> dict[str, float]:
    return res.groupby(col)["pnl_total"].sum().round(0).sort_values().to_dict()


def run(shocks: scenarios.ShockSet, trigger: dict | None = None, now: datetime | None = None,
        pos: pd.DataFrame | None = None, with_naive: bool = True) -> dict:
    t0 = time.perf_counter()
    pos = book() if pos is None else pos
    res = revalue(pos, shocks)
    before, pnl = float(res["value"].sum()), float(res["pnl_total"].sum())
    top = res.reindex(res["pnl_total"].sort_values().index).head(10)
    result = {
        "run_id": uuid.uuid4().hex[:12], "ts": (now or datetime.now(UTC)).isoformat(),
        "trigger": trigger or {"type": "manual"}, "scenario_key": shocks.key, "scenario": shocks.name,
        "analogue": shocks.analogue, "scale": round(shocks.scale, 3), "shocks": shocks.describe(),
        "value_before": before, "value_after": before + pnl, "pnl": pnl, "pnl_pct": pnl / before,
        "delta_expected_loss": float(res["delta_el"].sum()),
        "by_component": {c.replace("pnl_", ""): round(float(res[c].sum()), 0) for c in COMPONENTS},
        "by_asset_class": _group(res, "asset_class"), "by_sector": _group(res, "sector"),
        "by_region": _group(res, "region"), "by_country": _group(res, "country"), "by_rating": _group(res, "rating"),
        "sector_x_asset": res.pivot_table(index="sector", columns="asset_class", values="pnl_total", aggfunc="sum",
                                          fill_value=0).round(0).to_dict(orient="index"),
        "top_contributors": [{"position_id": r.position_id, "instrument": r.instrument, "counterparty": r.counterparty,
                              "sector": r.sector, "rating": r.rating, "pnl": round(float(r.pnl_total), 0)}
                             for r in top.itertuples() if r.pnl_total < 0],
        "capital": capital_impact(res),
    }
    if with_naive:
        naive = run(scenarios.naive(), pos=pos, with_naive=False)
        result["naive"] = {k: naive[k] for k in ("scenario", "pnl", "pnl_pct", "by_asset_class", "by_component")}
        result["naive"]["cet1_ratio_after"] = naive["capital"]["cet1_ratio_after"]
    result["compute_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return plain(result)


def what_if(event_type: str, impact: float, headline: str = "", ticker: str = "", sector: str = "",
            regions: list[str] | None = None) -> dict:
    if ticker and not sector:
        from riskpulse.universe import by_ticker
        sector = by_ticker()[ticker].sector if ticker in by_ticker() else ""
    entity_type = "company" if ticker else ("sector" if sector else "market")
    shocks = scenarios.build(event_type, impact, headline, entity_type, ticker, sector, regions or [])
    return run(shocks, trigger={"type": "what-if", "event_type": event_type, "impact": impact, "headline": headline,
                                "ticker": ticker, "sector": sector, "regions": regions or []})
