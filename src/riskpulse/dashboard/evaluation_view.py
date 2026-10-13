"""Tab 4 - model evaluation. Every number here is read from reports/ (written by `python run.py evaluate`)."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from riskpulse.config import REPORTS
from riskpulse.dashboard import charts

COLUMNS = {"dataset": "Dataset", "model": "Model", "n": "n", "accuracy": "Accuracy", "macro_f1": "Macro-F1",
           "score": "Score", "spearman_event_window": "Spearman, event window", "spearman_forward": "Spearman, forward",
           "top_decile_hit_rate": "Top-decile hit rate", "spearman_all_days": "Spearman, all days",
           "spearman_test_period": "Spearman, test period", "tier": "Tier", "model_load_s": "Model load (s)",
           "batch_docs_per_s": "Batch docs / s", "single_doc_ms_p50": "One doc, p50 (ms)",
           "single_doc_ms_p95": "One doc, p95 (ms)", "scenario": "Scenario", "pnl_musd": "P&L ($m)",
           "naive_pnl_musd": "Naive P&L ($m)", "rates_pnl_musd": "Rates P&L ($m)",
           "naive_rates_pnl_musd": "Naive rates P&L ($m)", "credit_pnl_musd": "Loan credit P&L ($m)",
           "cet1_after_pct": "CET1 after (%)", "compute_ms": "Compute (ms)"}
DATASETS = {"gold_news": "Gold news (200 held-out headlines)", "tweets_valid": "Finance tweets (validation)",
            "phrasebank": "PhraseBank (leaky for FinBERT)",
            "tweets_valid (weak labels)": "Finance tweets (validation, weak labels)"}


def _table(rows: list[dict], index: str | list[str], fmt: str = "{:.3f}") -> None:
    df = pd.DataFrame(rows)
    if "dataset" in df:
        df["dataset"] = df["dataset"].map(lambda d: DATASETS.get(d, d))
    idx = [index] if isinstance(index, str) else index
    df = df.set_index(idx).rename(columns=COLUMNS).rename_axis([COLUMNS.get(i, i) for i in idx])
    st.dataframe(df.style.format(fmt, subset=df.select_dtypes("float").columns)
                 .format("{:,}", subset=df.select_dtypes("integer").columns), use_container_width=True)


def _trigger_section(b: dict) -> None:
    tv, ra, fun = b.get("trigger_validation"), b.get("red_alert"), b.get("triage_funnel")
    if not (tv and ra):
        return
    st.markdown("### Does the stress trigger fire on days that matter?")
    st.caption("A two-year test fixed in code before it was run: every archive signal goes through the live "
               "trigger rules, and trigger days are compared with the S&P 500's actual move over [D, D+1].")
    t, red, naive = tv["trigger"], ra["test_red_alert_days"], ra["test_naive_busiest_news_days"]
    k = st.columns(4)                   # captions, not metric deltas: a delta would draw an up arrow
    k[0].metric("Stress runs, two years", f"{tv['runs']:,}")
    k[0].caption(f"on {t['days'] / tv['trading_days']:.0%} of trading days")
    k[1].metric("Trigger days vs other days", f"{t['ratio']:.2f}x",
                help="mean |S&P 500 move| on trigger days / on all other days")
    k[1].caption(f"p = {t['p_value']:.2f} (not significant)" if t["p_value"] >= 0.05 else f"p = {t['p_value']:.3f}")
    k[2].metric("Red-alert days, out of sample", f"{red['ratio']:.2f}x",
                help=f"threshold impact >= {ra['threshold']:.1f}, set on {ra['calibration_period'][0]} to "
                     f"{ra['calibration_period'][1]} from alert frequency only; tested from {ra['test_period'][0]}")
    k[2].caption(f"p = {red['p_value']:.3f}, {red['days']} days, impact >= {ra['threshold']:.1f}")
    k[3].metric("Busiest news days (naive)", f"{naive['ratio']:.2f}x")
    k[3].caption(f"p = {naive['p_value']:.2f}, same number of days")
    st.caption(f"{red['top_quintile_share']:.0%} of red-alert days were among the most volatile fifth of days (20% by "
               f"chance). Red alerts fired on {ra['year2_red_day_share']:.0%} of test-year days against a "
               f"{ra['target_share']:.0%} target, so the threshold needs regular recalibration.")
    if fun:
        stages = pd.DataFrame(fun["stages"]).rename(columns={"stage": "Stage", "count": "Count", "per_day": "Per day"})
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(charts.bars_h(dict(zip(stages["Stage"], stages["Per day"])),
                                          "Triage: from headlines to stress tests (per day, two-year archive)",
                                          fmt=",.1f", height=300), use_container_width=True)
        with c2:
            st.dataframe(stages.set_index("Stage").style.format({"Count": "{:,}", "Per day": "{:,.2f}"}),
                         use_container_width=True)


def render() -> None:
    path = REPORTS / "metrics.json"
    if not path.exists():
        st.info("No evaluation report yet - run `python run.py evaluate`.")
        return
    m = json.loads(path.read_text(encoding="utf-8"))
    st.caption(f"Generated {m.get('generated_at', '')[:16].replace('T', ' ')} UTC by `python run.py evaluate` · "
               "source: reports/metrics.json")

    if s := m.get("sentiment"):
        st.markdown("### Sentiment score vs naive baseline")
        _table(s["table"], ["dataset", "model"])
        st.caption(s.get("note", ""))
        if c := s.get("confusion"):
            st.plotly_chart(charts.heatmap_counts(c["matrix"], c["labels"], c["labels"],
                                                  f"Confusion matrix - {c['model']}, gold news (200 held-out)",
                                                  height=360), use_container_width=True)
    if e := m.get("events"):
        st.markdown("### Event classification vs keyword-only baseline")
        _table(e["table"], ["dataset", "model"])
        st.caption(e.get("note", ""))
        if pc := e.get("per_class"):
            df = pd.DataFrame(pc)
            st.plotly_chart(charts.grouped_bars(df["class"].tolist(), {"RiskPulse hybrid": df["hybrid_f1"].tolist(),
                                                                       "Keyword baseline": df["keyword_f1"].tolist()},
                                                "Per-class F1 on the held-out gold set", fmt=".2f", height=400,
                                                ytitle="F1"),
                            use_container_width=True)
    if i := m.get("impact"):
        st.markdown("### Impact score vs realised market moves (event study)")
        _table(i["table"], "score")
        st.caption(i.get("note", ""))
        if ml := i.get("market_level"):
            st.markdown("**Market-wide signals vs the S&P 500's own move** (the scope Module B stress tests use)")
            _table(ml["table"], "score")
        if cal := i.get("calibration"):
            t = cal["test_spearman"]
            st.caption(f"Walk-forward calibration (train to {cal['train_end']}): out-of-sample Spearman expert "
                       f"{t['expert']:.3f} vs calibrated {t['calibrated']:.3f} vs |sentiment| {t['abs_sentiment']:.3f} - "
                       + ("calibrated weights adopted." if cal["adopted"] else
                          "not adopted (pre-registered rule: must beat expert by > 0.01)."))
        if d := i.get("deciles"):
            df = pd.DataFrame(d)
            st.plotly_chart(charts.grouped_bars([str(x) for x in df["decile"]],
                                                {"Impact score": df["impact_mean_abs_ar"].tolist(),
                                                 "|Sentiment| baseline": df["baseline_mean_abs_ar"].tolist()},
                                                "Mean |abnormal return| by score decile", fmt=".2f",
                                                xtitle="score decile (1 = lowest)", ytitle="mean |abnormal return|, %"),
                            use_container_width=True)
    if b := m.get("module_b"):
        _trigger_section(b)
    if g := m.get("engine"):
        st.markdown("### Engine throughput and latency")
        _table(g["tiers"], "tier", "{:,.2f}")
    if b:
        st.markdown("### Module B scenario results")
        _table(b["scenarios"], "scenario", "{:,.2f}")
        st.caption(b.get("note", ""))
