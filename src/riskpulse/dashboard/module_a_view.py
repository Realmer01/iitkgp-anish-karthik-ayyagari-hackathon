"""Tab 2 - Module A: tactical index rebalancer."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from riskpulse.config import REPORTS
from riskpulse.dashboard import charts
from riskpulse.dashboard.client import get
from riskpulse.dashboard.theme import tint

BACKTEST_COLUMNS = {"cumulative_return": "Cumulative return", "annual_return": "Annual return",
                    "annual_volatility": "Volatility", "max_drawdown": "Max drawdown", "sharpe_rf0": "Sharpe (rf = 0)",
                    "active_return_annual": "Active return / yr", "tracking_error": "Tracking error",
                    "information_ratio": "Information ratio", "daily_hit_rate": "Daily hit rate",
                    "annual_turnover": "Turnover / yr (x)"}
PCT_COLUMNS = {"Cumulative return", "Annual return", "Volatility", "Max drawdown", "Active return / yr",
               "Tracking error", "Daily hit rate"}


def live_module_a(refresh: int) -> None:
    @st.fragment(run_every=refresh)
    def _live() -> None:
        state, hist = get("/module-a/state"), get("/module-a/weights")
        if not state or not hist:
            st.info("Waiting for the API...")
            return
        w = pd.DataFrame(state["weights"]).set_index("ticker")
        k = st.columns(5)
        k[0].metric("Rebalances", f"{state['rebalances']:,}")
        k[1].metric("Cumulative turnover", f"{state['cumulative_turnover']:.1%}")
        k[2].metric("Trading cost", f"{state['cumulative_cost_bps']:.1f} bp")
        top, bottom = w["active_weight"].idxmax(), w["active_weight"].idxmin()
        k[3].metric("Largest overweight", top, f"{w.loc[top, 'active_weight'] * 100:+.2f}pp")
        k[4].metric("Largest underweight", bottom, f"{w.loc[bottom, 'active_weight'] * 100:+.2f}pp")

        h = pd.DataFrame(hist)
        h["ts"] = pd.to_datetime(h["ts"], format="ISO8601", utc=True)      # mixed fractional seconds
        active = h.assign(active=h["weight"] - h["base_weight"]).pivot_table(index="ticker", columns="ts", values="active")
        active = active.reindex(w.index)
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = charts.heatmap_signed(active * 100, "Active weight over time (pp vs capped cap-weight base)",
                                        fmt=".2f", height=430, colorbar_title="pp")
            fig.update_xaxes(type="date", tickformat="%d %b", nticks=8)       # a real time axis for snapshots
            fig.update_traces(hovertemplate="%{y} · %{x|%d %b %H:%M}: %{z:.2f} pp<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.plotly_chart(charts.signed_bars_h((w["active_weight"] * 100).to_dict(), "Current tilt vs base (pp)",
                                                 fmt=".2f", height=430), use_container_width=True)
        moved = active.abs().max(axis=1).sort_values(ascending=False).index[:3].tolist()
        pick = st.multiselect("Tickers to plot (max 4)", list(w.index), default=moved, max_selections=4)
        if pick:
            wide = h.pivot_table(index="ts", columns="ticker", values="weight")[pick] * 100
            base = h[h["ticker"] == pick[0]].set_index("ts")["base_weight"] * 100
            st.plotly_chart(charts.lines(wide, "Index weight over time (%)", yfmt=".2f", reference=base,
                                         reference_name=f"{pick[0]} base weight", ytitle="weight %", xfmt="%d %b"),
                            use_container_width=True)
        log = pd.DataFrame(get("/module-a/log", limit=300) or [])
        st.markdown("##### Rebalance log - every weight change and the headline behind it")
        if log.empty:
            st.caption("No rebalances yet.")
        else:
            view = pd.DataFrame({"time (sim)": pd.to_datetime(log["ts"], format="ISO8601", utc=True).dt.strftime("%Y-%m-%d %H:%M"),
                                 "ticker": log["ticker"],
                                 "weight change (pp)": (log["new_weight"] - log["old_weight"]) * 100,
                                 "sentiment": log["sentiment"], "headline behind it": log["reason"]})
            st.dataframe(view.style.map(lambda v: tint(v, -0.5, 0.5), subset=["weight change (pp)"])
                         .format({"weight change (pp)": "{:+.2f}", "sentiment": "{:+.2f}"}),
                         hide_index=True, use_container_width=True, height=260,
                         column_config={"headline behind it": st.column_config.TextColumn(width="large")})
    _live()


def backtest_section() -> None:
    path = REPORTS / "metrics.json"
    if not path.exists():
        st.info("Run `python run.py evaluate` to generate the historical backtest.")
        return
    m = json.loads(path.read_text(encoding="utf-8")).get("module_a")
    if not m:
        return
    st.markdown("##### Historical backtest (Oct 2024 - Oct 2026, hourly bars, strict no-look-ahead)")
    st.caption(m.get("note", ""))
    curves = REPORTS / "module_a_equity.csv"
    if curves.exists():
        eq = pd.read_csv(curves, index_col=0, parse_dates=True)
        st.plotly_chart(charts.lines(eq, "Growth of $1 (net of 5 bp costs)", yfmt=".2f", xfmt="%b %Y"),
                        use_container_width=True)
    t = pd.DataFrame(m["table"]).set_index("portfolio").rename(columns=BACKTEST_COLUMNS)
    t = t[[c for c in BACKTEST_COLUMNS.values() if c in t.columns]].apply(pd.to_numeric, errors="coerce")
    for c in t.columns:                    # Streamlit shows missing numbers as "None"; benchmarks have no active stats
        t[c] = [("-" if pd.isna(v) else f"{v:.1%}" if c in PCT_COLUMNS else f"{v:.2f}") for v in t[c]]
    st.dataframe(t, use_container_width=True)
    sens = REPORTS / "module_a_sensitivity.csv"
    if sens.exists():
        s = pd.read_csv(sens).pivot_table(index="kappa", columns="half_life_hours", values="active_return_annual") * 100
        s.index = [f"κ = {k:g}" for k in s.index]
        s.columns = [f"{h:g} h" for h in s.columns]
        st.plotly_chart(charts.heatmap_signed(s, "Sensitivity: active return vs neutral policy (% / yr)", fmt=".2f",
                                              height=320, show_values=True, colorbar_title="% / yr",
                                              xtitle="half-life of the sentiment signal",
                                              ytitle="tilt strength"), use_container_width=True)
