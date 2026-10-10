"""Engine throughput / latency, Module A backtest and Module B scenario evaluation."""
from __future__ import annotations

import time
from datetime import timedelta

import numpy as np
import pandas as pd

from riskpulse.config import LABELED, RAW
from riskpulse.module_a import backtest as bt
from riskpulse.module_b import scenarios, stress
from riskpulse.schema import RawDocument, SourceType


# ---------------------------------------------------------------------------- engine
def engine_performance(tiers: list[str]) -> dict:
    from riskpulse.engine import RiskEngine

    heads = pd.read_csv(RAW / "news_history.csv.gz").drop_duplicates("title").sample(512, random_state=7)["title"].tolist()
    gold = pd.read_csv(LABELED / "gold_sample.csv")["text"].head(60).tolist()
    rows = []
    for tier in tiers:
        t0 = time.perf_counter()
        eng = RiskEngine(tier)
        load = time.perf_counter() - t0
        docs = [RawDocument(source="bench", source_type=SourceType.NEWS, publisher="bench", title=h) for h in heads]
        t0 = time.perf_counter()
        for i in range(0, len(docs), 64):
            eng.process(docs[i:i + 64])
        batch_s = time.perf_counter() - t0
        single = []
        for g in gold:
            t1 = time.perf_counter()
            eng.analyze_text(g)
            single.append((time.perf_counter() - t1) * 1000)
        rows.append({"tier": tier, "model_load_s": round(load, 2), "batch_docs_per_s": round(len(docs) / batch_s, 1),
                     "single_doc_ms_p50": round(float(np.percentile(single, 50)), 1),
                     "single_doc_ms_p95": round(float(np.percentile(single, 95)), 1)})
    return {"tiers": rows, "note": "CPU only (laptop, 12 logical cores). 512 archive headlines in batches of 64; "
                                   "single-document latency over 60 gold headlines."}


# ---------------------------------------------------------------------------- module A
def module_a(quick: bool = False) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    bars = bt.load_bars()
    sig = bt.load_signals("full")
    res = bt.run(sig, bars)
    r = res["returns"]
    table = []
    for name, col, bench in (("RiskPulse sentiment tilt (FinBERT tier)", "strategy", "neutral_policy"),
                             ("Neutral policy (same rules, kappa = 0)", "neutral_policy", None),
                             ("Static capped cap-weight", "static_cap", None), ("Equal weight", "equal_weight", None)):
        m = bt.metrics(r[col], r[bench] if bench else None, r["turnover"] if col == "strategy" else None)
        table.append({"portfolio": name, **{k: round(v, 4) for k, v in m.items()}})
    lite = bt.run(bt.load_signals("lite"), bars, benchmarks=False)["returns"]
    m_lite = bt.metrics(lite["strategy"], r["neutral_policy"], lite["turnover"])
    table.append({"portfolio": "Naive: lexicon sentiment tilt (lite tier)", **{k: round(v, 4) for k, v in m_lite.items()}})
    ic = bt.information_coefficient(res, bars)

    rng = np.random.default_rng(0)
    placebo = []
    comp = sig["entity_type"] == "company"
    for _ in range(3):
        s = sig.copy()
        names = s.loc[comp, "entity_id"].unique()
        s.loc[comp, "entity_id"] = s.loc[comp, "entity_id"].map(dict(zip(names, rng.permutation(names))))
        rp = bt.run(s, bars, benchmarks=False)["returns"]
        placebo.append(bt.metrics(rp["strategy"], r["neutral_policy"])["active_return_annual"])
    delayed = sig.copy()
    delayed["available_at"] = delayed["available_at"] + pd.Timedelta(days=1)
    rd = bt.run(delayed.sort_values("available_at"), bars, benchmarks=False)["returns"]

    equity = (1 + r[["strategy", "neutral_policy", "static_cap", "equal_weight"]]).cumprod()
    equity = equity.groupby(equity.index.normalize()).last()
    equity.columns = ["Sentiment tilt", "Neutral policy", "Static cap-weight", "Equal weight"]
    sens = pd.DataFrame() if quick else bt.sensitivity(sig, bars)
    out = {"table": table, "ic": ic,
           "placebo_ticker_shuffle_active_return": [round(x, 4) for x in placebo],
           "delay_plus_1d_active_return": round(bt.metrics(rd["strategy"], r["neutral_policy"])["active_return_annual"], 4),
           "april_2025_window": {k: round(v, 4) for k, v in bt.window(res, "2025-04-01", "2025-04-14").items()},
           "stress_days": bt.stress_days(res),
           "note": "Hourly open-to-open returns, Oct 2024 - Oct 2026, 5 bp costs, strict no-look-ahead (date-only news "
                   "usable only after its day ends). Active figures are vs the neutral policy. With correct timing the "
                   "sentiment tilt shows no reliable edge: day-stamped news on mega caps is priced by the next open."}
    return out, equity, sens


# ---------------------------------------------------------------------------- module B
SCENARIOS = [("Geopolitical", "US imposes sweeping tariffs on Chinese imports; trade war escalates", "", ["China"]),
             ("Geopolitical", "Russia invades; EU and US impose sanctions", "", ["Russia/CIS", "Europe"]),
             ("Macroeconomic", "Fed raises rates by 75 basis points and signals more hikes", "", []),
             ("Macroeconomic", "Recession fears as jobless claims surge", "", []),
             ("Credit Event", "Boeing downgraded to junk amid default fears", "BA", []),
             ("Operational Risk", "Global IT outage halts airlines and banks", "", [])]


def module_b(engine=None) -> dict:
    rows = []
    for ev, head, ticker, regions in SCENARIOS:
        r = stress.what_if(ev, 8.5, head, ticker, regions=regions)
        n = r["naive"]
        rows.append({"scenario": r["scenario"], "pnl_musd": r["pnl"] / 1e6, "naive_pnl_musd": n["pnl"] / 1e6,
                     "rates_pnl_musd": r["by_component"]["rates"] / 1e6,
                     "naive_rates_pnl_musd": n["by_component"]["rates"] / 1e6,
                     "credit_pnl_musd": r["by_component"]["credit"] / 1e6,
                     "cet1_after_pct": r["capital"]["cet1_ratio_after"] * 100, "compute_ms": r["compute_ms"]})
    sign_errors = sum(np.sign(x["rates_pnl_musd"]) != np.sign(x["naive_rates_pnl_musd"]) for x in rows)
    latency = None
    if engine is not None:
        t0 = time.perf_counter()
        sig = engine.analyze_text("Russia launches missile strikes on Ukraine; US and EU impose sweeping new sanctions")[0]
        sh = scenarios.build(sig.event_type, sig.impact_score, sig.headline, sig.entity.type, sig.entity.ticker or "",
                             sig.entity.sector or "", sig.regions)
        stress.run(sh)
        latency = round(time.perf_counter() - t0, 3)
    return {"scenarios": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()} for x in rows],
            "rates_sign_errors_naive": int(sign_errors), "headline_to_stress_result_s": latency,
            "baseline_cet1_pct": 13.0, "minimum_cet1_pct": 10.5,
            "note": "Impact 8.5 for every scenario. Naive = rates +200bp, spreads +100bp, equities -10% for any event. "
                    "Latency = engine scoring of one headline + full portfolio revaluation on a laptop CPU; the comparison "
                    "is an assumed daily (overnight) batch stress run, i.e. up to ~24h from news to numbers."}


def replay_triggers() -> dict:
    """Feed the April 2025 demo signals through the Module B monitor on a simulated clock."""
    import gzip

    from riskpulse.config import OUTPUT
    from riskpulse.module_b.monitor import StressMonitor
    from riskpulse.schema import RiskSignal

    with gzip.open(OUTPUT / "demo_signals.jsonl.gz", "rt", encoding="utf-8") as fh:
        # time order, and within one timestamp highest impact first - exactly how the live monitor takes a batch
        sigs = sorted((RiskSignal.model_validate_json(x) for x in fh if x.strip()),
                      key=lambda s: (s.published_at, -s.impact_score))
    clock = {"t": sigs[0].published_at}
    mon = StressMonitor(now=lambda: clock["t"])
    for s in sigs:
        clock["t"] = s.published_at + timedelta(seconds=1)
        if mon.wants(s):
            mon.consider(s)
    runs = mon.runs[::-1]

    def brief(r: dict) -> dict:
        return {"time": r["ts"][:16], "headline": r["trigger"]["headline"], "impact": r["trigger"]["impact_score"],
                "scenario": r["scenario"], "pnl_musd": round(r["pnl"] / 1e6, 1),
                "cet1_after_pct": round(r["capital"]["cet1_ratio_after"] * 100, 2)}
    return {"signals": len(sigs), "eligible": sum(mon.wants(s) for s in sigs), "runs": len(runs),
            "suppressed": mon.suppressed, "first_run": brief(runs[0]) if runs else None,
            "highest_impact_run": brief(max(runs, key=lambda r: r["trigger"]["impact_score"])) if runs else None}
