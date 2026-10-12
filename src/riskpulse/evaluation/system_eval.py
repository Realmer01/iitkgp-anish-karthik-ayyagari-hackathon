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


def _archive_signals() -> pd.DataFrame:
    from riskpulse.config import OUTPUT
    s = pd.read_csv(OUTPUT / "signals_history_full.csv.gz", keep_default_na=False)
    s["published_at"] = pd.to_datetime(s["published_at"], utc=True, format="ISO8601")
    s["impact"] = s["impact"].astype(float)
    return s.sort_values(["published_at", "impact"], ascending=[True, False], kind="stable").reset_index(drop=True)


def _monitor_decisions(sig: pd.DataFrame) -> tuple[list[dict], dict, int]:
    """Replay archive signals through the real StressMonitor; the revaluation is stubbed (decisions only)."""
    from types import SimpleNamespace
    from unittest import mock

    from riskpulse.module_b.monitor import StressMonitor

    clock = {"t": sig["published_at"].iloc[0].to_pydatetime()}
    mon = StressMonitor(now=lambda: clock["t"])
    runs, wanted = [], 0
    stub = lambda shocks, trigger=None, now=None, **kw: {"run_id": "eval", "pnl": 0.0, "scenario": shocks.name,  # noqa: E731
                                                          "trigger": trigger, "ts": now.isoformat()}
    with mock.patch("riskpulse.module_b.monitor.stress.run", side_effect=stub):
        for r in sig.itertuples():
            t = r.published_at.to_pydatetime()
            s = SimpleNamespace(
                signal_id=r.signal_id, story_id=r.story_id, headline=r.headline, event_type=r.event_type,
                impact_score=float(r.impact), event_confidence=float(r.event_conf), sentiment_score=float(r.sentiment),
                entity=SimpleNamespace(type=r.entity_type, id=r.entity_id, ticker=r.entity_id if r.entity_type == "company"
                                       else None, sector=r.sector or None),
                regions=[x for x in str(r.regions).split("|") if x], publisher=r.publisher, published_at=t,
                generated_at=t)
            clock["t"] = t + timedelta(seconds=1)
            if mon.wants(s):
                wanted += 1
                res = mon.consider(s)
                if res is not None:
                    runs.append({"published_at": r.published_at, "precision": r.time_precision, "scenario": res["scenario"],
                                 "impact": float(r.impact), "headline": r.headline})
    return runs, dict(mon.suppressed), wanted


def trigger_validation(n_perm: int = 10_000) -> dict:
    """Does the automatic stress trigger fire on days the market really moves? (test fixed before computing it)

    * Every full-tier archive signal (Oct 2024 - Oct 2026) goes through the real StressMonitor in time order,
      highest impact first within a timestamp (same rules as live: per-type thresholds, confidence, adverse tone,
      reported-event framing, story dedup, news-time cooldown). Only the decisions matter, so the portfolio
      revaluation is stubbed out. Archive headlines are stored truncated to 120 characters.
    * Trigger day = the trading day D a run's news belongs to (event_day, as in the impact event study);
      market move = |SPY return on D + return on D+1|, the window of the market-level impact test.
    * Primary: mean move on trigger days / mean move on all other days, with a one-sided permutation p-value
      (n_perm random sets of the same number of trading days, seed 0). Secondary: share of trigger days in the
      top quintile of moves (20% by chance). Naive comparator: the same number of days with the most
      market-wide headlines. Robustness: the same test without the April 2025 tariff shock (31 Mar - 15 Apr).
    """
    from riskpulse.config import PRICES
    from riskpulse.evaluation.impact_eval import event_day

    sig = _archive_signals()
    runs, suppressed, wanted = _monitor_decisions(sig)
    close = pd.read_csv(PRICES / "prices_daily.csv", index_col=0, parse_dates=True)
    days = pd.DatetimeIndex(close.index)
    r = close["SPY"].pct_change(fill_method=None)
    move = (r + r.shift(-1)).abs().dropna()
    move = move[move.index > days[0]]
    trig = pd.Series([event_day(x["published_at"], x["precision"], days) for x in runs]).dropna()
    trig_days = sorted(set(trig) & set(move.index))
    mkt = sig[sig["entity_type"] == "market"]
    vol = pd.Series([event_day(t, p, days) for t, p in zip(mkt["published_at"], mkt["time_precision"])]).value_counts()
    vol = vol[vol.index.isin(move.index)]
    rng = np.random.default_rng(0)
    top_q = move.quantile(0.8)

    def stats(sel: list, universe: pd.Series) -> dict:
        sel = [d for d in sel if d in universe.index]
        inside, outside = universe.loc[sel], universe.drop(sel)
        ratio = float(inside.mean() / outside.mean())
        k, vals = len(sel), universe.to_numpy()
        perm = np.array([vals[rng.choice(len(vals), k, replace=False)].mean() for _ in range(n_perm)])
        rest = (vals.sum() - perm * k) / (len(vals) - k)
        p = float((np.sum(perm / rest >= ratio) + 1) / (n_perm + 1))
        return {"days": k, "mean_move_pct": round(float(inside.mean()) * 100, 3),
                "other_days_mean_move_pct": round(float(outside.mean()) * 100, 3), "ratio": round(ratio, 3),
                "p_value": round(p, 4), "top_quintile_share": round(float((inside >= top_q).mean()), 3)}

    shock = (move.index >= "2025-03-31") & (move.index <= "2025-04-15")
    naive = vol.sort_values(ascending=False, kind="stable").index[:len(trig_days)].tolist()
    ex = move[~shock]
    return {"runs": len(runs), "stress_eligible_signals": wanted, "suppressed": suppressed,
            "trading_days": int(len(move)),
            "trigger": stats(trig_days, move), "naive_busiest_news_days": stats(naive, move),
            "trigger_ex_april_2025": stats([d for d in trig_days if d in ex.index], ex),
            "note": "Market move = |SPY return over [D, D+1]|. Ratio = mean move on the selected days / mean on all "
                    "other days; p = one-sided permutation p-value. Naive = the same number of days with the most "
                    "market-wide headlines. Top-quintile share is 0.20 by chance."}


def _move_and_days():
    from riskpulse.config import PRICES
    close = pd.read_csv(PRICES / "prices_daily.csv", index_col=0, parse_dates=True)
    days = pd.DatetimeIndex(close.index)
    r = close["SPY"].pct_change(fill_method=None)
    move = (r + r.shift(-1)).abs().dropna()
    return move[move.index > days[0]], days


def _perm_test(sel: list, universe: pd.Series, rng, n_perm: int) -> dict:
    sel = [d for d in sel if d in universe.index]
    inside, outside = universe.loc[sel], universe.drop(sel)
    if not len(sel) or not len(outside):
        return {"days": len(sel)}
    ratio = float(inside.mean() / outside.mean())
    k, vals = len(sel), universe.to_numpy()
    perm = np.array([vals[rng.choice(len(vals), k, replace=False)].mean() for _ in range(n_perm)])
    rest = (vals.sum() - perm * k) / (len(vals) - k)
    return {"days": k, "mean_move_pct": round(float(inside.mean()) * 100, 3),
            "other_days_mean_move_pct": round(float(outside.mean()) * 100, 3), "ratio": round(ratio, 3),
            "p_value": round(float((np.sum(perm / rest >= ratio) + 1) / (n_perm + 1)), 4),
            "top_quintile_share": round(float((inside >= universe.quantile(0.8)).mean()), 3)}


def red_alert_validation(target_share: float = 0.10, n_perm: int = 10_000) -> dict:
    """Escalation tier on top of the stress trigger, calibrated on year 1 and tested on year 2 (fixed in advance).

    * Same replay as trigger_validation. The 491 trading days are split in half: year 1 (calibration) and
      year 2 (test).
    * Red-alert threshold = the lowest impact level such that, on year-1 trading days, red alerts (days with a
      stress run whose triggering signal's impact >= threshold) occur on at most target_share (10%) of days.
      Chosen from alert frequency only - no market outcome is used.
    * Test on year 2: mean |SPY move over [D, D+1]| on red-alert days / on other year-2 days, one-sided
      permutation p-value (seed 1), top-quintile share; reported next to all stress-run days and the same number
      of busiest-news days in year 2.
    """
    from riskpulse.evaluation.impact_eval import event_day

    sig = _archive_signals()
    runs, _, _ = _monitor_decisions(sig)
    move, days = _move_and_days()
    run_days = pd.DataFrame({"D": [event_day(x["published_at"], x["precision"], days) for x in runs],
                             "impact": [x["impact"] for x in runs]}).dropna()
    daily_max = run_days.groupby("D")["impact"].max().reindex(move.index).fillna(0.0)
    half = len(move) // 2
    y1, y2 = move.index[:half], move.index[half:]
    levels = np.round(np.arange(7.0, 10.01, 0.1), 1)
    share = {lvl: float((daily_max.loc[y1] >= lvl).mean()) for lvl in levels}
    threshold = float(min(lvl for lvl in levels if share[lvl] <= target_share))
    red2 = [d for d in y2 if daily_max.loc[d] >= threshold]
    any2 = [d for d in y2 if daily_max.loc[d] > 0]
    sig_mk = sig[sig["entity_type"] == "market"]
    vol = pd.Series([event_day(t, p, days) for t, p in zip(sig_mk["published_at"], sig_mk["time_precision"])]).value_counts()
    vol2 = vol[vol.index.isin(y2)].sort_values(ascending=False, kind="stable")
    rng = np.random.default_rng(1)
    m2 = move.loc[y2]
    return {"threshold": threshold, "target_share": target_share,
            "calibration_period": [str(y1[0].date()), str(y1[-1].date())],
            "test_period": [str(y2[0].date()), str(y2[-1].date())],
            "year1_red_day_share": round(share[threshold], 3),
            "year2_red_day_share": round(len(red2) / len(y2), 3),
            "test_red_alert_days": _perm_test(red2, m2, rng, n_perm),
            "test_all_stress_run_days": _perm_test(any2, m2, rng, n_perm),
            "test_naive_busiest_news_days": _perm_test(vol2.index[:len(red2)].tolist(), m2, rng, n_perm),
            "note": "Red alert = a stress run whose trigger impact >= threshold; threshold set on year 1 so red "
                    "alerts fall on <= 10% of trading days (frequency only, no market outcomes). Tested on year 2."}


def triage_funnel() -> dict:
    """From raw headlines to stress runs over the whole archive (what a risk desk would have to look at)."""
    from riskpulse.ingestion.base import excluded_publisher

    frames = [pd.read_csv(RAW / "news_history.csv.gz"), pd.read_csv(RAW / "gdelt_headlines.csv.gz")]
    raw = pd.concat(frames, ignore_index=True).dropna(subset=["title"])
    raw = raw[~raw["publisher"].map(excluded_publisher)].drop_duplicates(["title", "publisher"])
    sig = _archive_signals()
    runs, suppressed, wanted = _monitor_decisions(sig)
    span = max((sig["published_at"].max() - sig["published_at"].min()).days, 1)
    stages = [("headlines", len(raw)), ("signals (linked to a company, sector or market)", len(sig)),
              ("impact above 7", int((sig["impact"] > 7).sum())),
              ("stress-eligible and confident", wanted), ("stress tests run", len(runs))]
    return {"days": span, "stages": [{"stage": n, "count": int(c), "per_day": round(c / span, 2)} for n, c in stages],
            "suppressed_by_rules": suppressed,
            "market_wide_share_above_7": round(float((sig.loc[sig["entity_type"] == "market", "impact"] > 7).mean()), 3)}
