"""Tab 3 - Module B: event-driven stress testing of the wholesale banking book."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from riskpulse.dashboard import charts
from riskpulse.dashboard.client import get, post
from riskpulse.dashboard.theme import tint

EVENTS = ["Geopolitical", "Macroeconomic", "Credit Event", "Operational Risk", "Legal & Regulatory",
          "Merger & Acquisition"]
REGIONS = ["North America", "Europe", "China", "Asia-Pacific", "Middle East", "Russia/CIS", "Latin America"]


@st.cache_data(ttl=600, show_spinner=False)
def portfolio() -> dict | None:
    return get("/module-b/portfolio")


def render_run(r: dict) -> None:
    cap = r["capital"]
    t = r.get("trigger", {})
    if t.get("headline"):
        st.markdown(f"**Trigger:** “{t['headline']}” · {t.get('event_type')} · impact {t.get('impact_score', t.get('impact'))}"
                    + (f" · {t.get('publisher')}" if t.get("publisher") else ""))
    st.markdown(f"**Scenario:** {r['scenario']} ×{r['scale']:.2f}  <span class='rp-muted'>analogue: {r['analogue']}</span>",
                unsafe_allow_html=True)
    k = st.columns(5)
    k[0].metric("Value before", f"${r['value_before'] / 1e9:,.3f}bn")
    k[1].metric("Value after", f"${r['value_after'] / 1e9:,.3f}bn", f"{r['pnl_pct']:+.2%}")
    k[2].metric("Stress P&L", f"${r['pnl'] / 1e6:,.1f}m")
    k[3].metric("Δ expected loss", f"${r['delta_expected_loss'] / 1e6:,.1f}m")
    k[4].metric("CET1 ratio", f"{cap['cet1_ratio_after']:.2%}", f"{(cap['cet1_ratio_after'] - cap['cet1_ratio_before']) * 1e4:+.0f} bp")
    if cap["breach"]:
        st.markdown(f"<div class='rp-callout'>⚠ <b>Capital breach</b>: stressed CET1 {cap['cet1_ratio_after']:.2%} is "
                    f"below the {cap['minimum_ratio']:.1%} minimum (simplified proxy).</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='rp-callout ok'>✓ CET1 stays above the {cap['minimum_ratio']:.1%} minimum "
                    f"(headroom {cap['headroom_bp_after']:.0f} bp, simplified proxy).</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.waterfall(r["value_before"], r["by_asset_class"], "Portfolio value: before → after by asset class"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(charts.signed_bars_h(r["by_component"], "P&L by risk factor"), use_container_width=True)
    sx = pd.DataFrame(r["sector_x_asset"]).T.fillna(0)
    c3, c4 = st.columns([3, 2])
    with c3:
        st.plotly_chart(charts.heatmap_signed(sx, "P&L by sector × asset class", height=420), use_container_width=True)
    with c4:
        st.markdown("**Top loss contributors**")
        top = pd.DataFrame(r["top_contributors"])
        if not top.empty:
            st.dataframe(top[["position_id", "counterparty", "instrument", "pnl"]].style
                         .map(lambda v: tint(v, -2e7, 2e7), subset=["pnl"]).format({"pnl": "${:,.0f}"}),
                         hide_index=True, use_container_width=True, height=380)
    if "naive" in r:
        n = r["naive"]
        comps = list(r["by_component"])
        st.plotly_chart(charts.grouped_bars(comps, {"Event-driven scenario": [r["by_component"][c] / 1e6 for c in comps],
                                                    "Naive static shock": [n["by_component"].get(c, 0) / 1e6 for c in comps]},
                                            "Event-driven vs naive static stress (P&L, $m)", fmt=",.1f"),
                        use_container_width=True)
        st.caption(f"Naive approach (rates +200bp, spreads +100bp, equities −10% for every event): P&L "
                   f"${n['pnl'] / 1e6:,.1f}m vs event-driven ${r['pnl'] / 1e6:,.1f}m. The naive shock ignores what the "
                   "event actually does - e.g. it adds rate losses in flight-to-quality events where yields fall.")


def live_module_b(refresh: int) -> None:
    pf = portfolio()
    if pf:
        s = pf["summary"]
        k = st.columns(4)
        k[0].metric("Book value", f"${s['value_usd'] / 1e9:,.2f}bn")
        k[1].metric("Positions", f"{s['positions']:,}")
        k[2].metric("Loan EAD", f"${s['loan_ead_usd'] / 1e9:,.2f}bn")
        k[3].metric("Expected loss (12m)", f"${s['expected_loss_usd'] / 1e6:,.1f}m")
        with st.expander("Portfolio composition"):
            pos = pd.DataFrame(pf["positions"])
            c1, c2 = st.columns(2)
            c1.plotly_chart(charts.bars_h(pos.groupby("asset_class")["value"].sum().to_dict(), "Value by asset class ($)",
                                          fmt="$,.3s", height=260), use_container_width=True)
            c2.plotly_chart(charts.bars_h(pos.groupby("sector")["value"].sum().to_dict(), "Value by sector ($)",
                                          fmt="$,.3s", height=260), use_container_width=True)
            st.dataframe(pos[["position_id", "instrument", "counterparty", "sector", "country", "rating", "value"]],
                         hide_index=True, use_container_width=True, height=240)

    @st.fragment(run_every=refresh)
    def _runs() -> None:
        data = get("/module-b/runs", limit=100) or {}
        runs, mon = data.get("runs", []), data.get("monitor", {})
        sup = mon.get("suppressed", {})
        st.markdown(f"##### Trigger log  <span class='rp-muted'>rule: impact > {mon.get('default_threshold', 7)} "
                    f"(per-type overrides), confidence ≥ {mon.get('min_event_confidence', 0.45)}, adverse tone, a "
                    f"reported event (not commentary); the same scenario re-runs within "
                    f"{mon.get('cooldown_minutes', 360) / 60:.0f} h only if impact rises by "
                    f"{mon.get('escalation_delta', 1.0):.0f}+ · suppressed: {sup.get('commentary', 0)} commentary / "
                    f"speculation, {sup.get('not_adverse', 0)} positive-tone, {sup.get('story_dedup', 0)} repeats "
                    f"of a story, {sup.get('cooldown', 0)} non-escalating</span>", unsafe_allow_html=True)
        if not runs:
            st.info("No stress test triggered yet - waiting for a high-impact event.")
            return
        log = pd.DataFrame([{"time (sim)": r["ts"][:16].replace("T", " "), "event": r["trigger"].get("event_type"),
                             "impact": r["trigger"].get("impact_score"), "scenario": r["scenario"],
                             "P&L $m": r["pnl"] / 1e6, "CET1 after": r["capital"]["cet1_ratio_after"],
                             "headline": r["trigger"].get("headline", "")} for r in runs])
        st.dataframe(log.style.format({"P&L $m": "{:,.1f}", "CET1 after": "{:.2%}", "impact": "{:.1f}"}),
                     hide_index=True, use_container_width=True, height=220)
        idx = st.selectbox("Inspect run", range(len(runs)), format_func=lambda i: f"{log.iloc[i]['time (sim)']} · "
                           f"{log.iloc[i]['scenario']} · {log.iloc[i]['headline'][:70]}")
        render_run(runs[idx])
    _runs()


def what_if_panel() -> None:
    st.markdown("##### What-if: run any scenario manually")
    with st.form("what_if"):
        c1, c2, c3 = st.columns([1.2, 1, 1.4])
        ev = c1.selectbox("Event type", EVENTS)
        imp = c2.slider("Impact score", 1.0, 10.0, 8.5, 0.1)
        ticker = c3.selectbox("Named company (optional)", ["", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
                                                             "JPM", "GS", "XOM", "JNJ", "PFE", "BA", "CAT", "WMT", "KO"])
        headline = st.text_input("Headline cue words (route the scenario)", "New tariffs on Chinese imports escalate trade war")
        regions = st.multiselect("Regions named in the event", REGIONS, default=["China"])
        go_ = st.form_submit_button("Run stress test", type="primary")
    if go_:
        res = post("/module-b/what-if", json={"event_type": ev, "impact": imp, "headline": headline, "ticker": ticker,
                                              "regions": regions})
        if res and "error" not in res:
            render_run(res)
        else:
            st.error(res.get("error") if res else "API unavailable")
