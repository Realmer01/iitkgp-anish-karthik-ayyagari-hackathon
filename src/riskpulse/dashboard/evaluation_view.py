"""Tab 4 - model evaluation. Every number here is read from reports/ (written by `python run.py evaluate`)."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from riskpulse.config import REPORTS
from riskpulse.dashboard import charts


def _table(rows: list[dict], index: str | list[str], fmt: str = "{:.3f}") -> None:
    df = pd.DataFrame(rows).set_index(index)
    st.dataframe(df.style.format(fmt, subset=df.select_dtypes("float").columns)
                 .format("{:,}", subset=df.select_dtypes("integer").columns), use_container_width=True)


def render() -> None:
    path = REPORTS / "metrics.json"
    if not path.exists():
        st.info("No evaluation report yet - run `python run.py evaluate`.")
        return
    m = json.loads(path.read_text(encoding="utf-8"))
    st.caption(f"Generated {m.get('generated_at', '')[:16]} by `python run.py evaluate` · source: reports/metrics.json")

    if s := m.get("sentiment"):
        st.markdown("### Sentiment score vs naive baseline")
        _table(s["table"], ["dataset", "model"])
        st.caption(s.get("note", ""))
        if c := s.get("confusion"):
            st.plotly_chart(charts.heatmap_counts(c["matrix"], c["labels"], c["labels"],
                                                  f"Confusion matrix - {c['model']} on {c['dataset']}", height=330),
                            use_container_width=True)
    if e := m.get("events"):
        st.markdown("### Event classification vs keyword-only baseline")
        _table(e["table"], ["dataset", "model"])
        st.caption(e.get("note", ""))
        if pc := e.get("per_class"):
            df = pd.DataFrame(pc)
            st.plotly_chart(charts.grouped_bars(df["class"].tolist(), {"RiskPulse hybrid": df["hybrid_f1"].tolist(),
                                                                       "Keyword baseline": df["keyword_f1"].tolist()},
                                                "Per-class F1 on the held-out gold set", fmt=".2f", height=380),
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
                                                "Mean |abnormal return| (%) by score decile", fmt=".2f"),
                            use_container_width=True)
    if g := m.get("engine"):
        st.markdown("### Engine throughput and latency")
        _table(g["tiers"], "tier", "{:.2f}")
    if b := m.get("module_b"):
        st.markdown("### Module B scenario results")
        _table(b["scenarios"], "scenario", "{:,.2f}")
        st.caption(b.get("note", ""))
