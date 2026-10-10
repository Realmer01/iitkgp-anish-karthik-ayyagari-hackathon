"""Tab 1 - live risk feed and the 'Analyze a headline' playground."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from riskpulse.dashboard import charts
from riskpulse.dashboard.client import get, post
from riskpulse.dashboard.theme import tint

EXAMPLES = {
    "Geopolitical shock (triggers Module B)":
        "Russia launches full-scale invasion of Ukraine; global stocks plunge as US and EU impose sweeping sanctions",
    "Credit event at an index company": "Boeing files for Chapter 11 bankruptcy protection after missing debt payments",
    "Positive product news (lifts weight)": "Nvidia unveils next-generation AI chip as analysts say demand far exceeds supply",
    "Hawkish central bank": "Fed raises interest rates by 75 basis points and signals more hikes to fight inflation",
    "Social post with slang": "$TSLA to the moon 🚀 deliveries crushed estimates, bears rekt",
}


def _signal_table(rows: list[dict]) -> None:
    if not rows:
        st.info("No signals yet - the replay clock is starting.")
        return
    df = pd.DataFrame([{"published (UTC)": r["published_at"][:16].replace("T", " "), "entity": r["entity"]["id"],
                        "sentiment": r["sentiment_score"], "event": r["event_type"], "impact": r["impact_score"],
                        "source": r["publisher"] or r["source"], "headline": r["headline"]} for r in rows])
    styled = df.style.map(lambda v: tint(v), subset=["sentiment"]).format({"sentiment": "{:+.2f}", "impact": "{:.1f}"})
    st.dataframe(styled, hide_index=True, use_container_width=True, height=360, column_config={
        "impact": st.column_config.ProgressColumn("impact (1-10)", min_value=1, max_value=10, format="%.1f"),
        "headline": st.column_config.TextColumn(width="large")})


def analyze_box() -> None:
    st.subheader("Analyze a headline")
    c1, c2 = st.columns([3, 1])
    with c2:
        example = st.selectbox("Example", ["(type your own)", *EXAMPLES])
        source_type = st.radio("Source type", ["news", "social"], horizontal=True,
                               index=1 if example.startswith("Social") else 0)
    with c1:
        text = st.text_area("Headline or post", value=EXAMPLES.get(example, ""), height=90,
                            placeholder="Paste any financial headline or social post...")
    b1, b2, _ = st.columns([1, 1.6, 4])
    preview = b1.button("Analyze", type="secondary", use_container_width=True)
    publish = b2.button("Analyze & publish to modules", type="primary", use_container_width=True)
    if not (preview or publish) or len(text.strip()) < 3:
        return
    before = get("/module-a/state")
    with st.spinner("Scoring with the risk engine..."):
        res = post("/analyze", json={"text": text, "source_type": source_type, "inject": bool(publish)})
    if not res or "error" in res:
        st.error(f"Engine unavailable: {res.get('error') if res else 'no response'}")
        return
    if not res["signals"]:
        st.warning("No company, sector or market entity found - the engine drops unlinked text as noise.")
        return
    for s in res["signals"]:
        _signal_card(s)
    if publish:
        _module_reactions(res, before)


def _signal_card(s: dict) -> None:
    d = s["impact_drivers"]
    st.markdown(f"<span class='rp-chip'>{s['entity']['type']}: <b>{s['entity']['id']}</b></span>"
                + "".join(f"<span class='rp-chip'>{r}</span>" for r in s["regions"])
                + (f"<span class='rp-chip'>rule: {s['event_rule']}</span>" if s.get("event_rule") else "")
                + f"<span class='rp-chip'>{s['model_tier']} tier · {s['latency_ms']:.0f} ms</span>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Sentiment score", f"{s['sentiment_score']:+.2f}", s["sentiment_label"],
              delta_color="normal" if s["sentiment_score"] >= 0 else "inverse")
    m2.metric("Event classification", s["event_type"], f"confidence {s['event_confidence']:.0%}", delta_color="off")
    m3.metric("Impact score", f"{s['impact_score']:.1f} / 10",
              "above stress threshold" if s["impact_score"] > 7 else "below stress threshold", delta_color="off")
    c1, c2 = st.columns([3, 2])
    with c1:
        contrib = {k.replace("_", " "): v for k, v in d["contributions"].items()}
        st.plotly_chart(charts.bars_h(contrib, "Why this impact score: driver contributions (logit units)", fmt=".2f",
                                      height=280), use_container_width=True)
    with c2:
        probs = pd.Series(s["event_probs"]).sort_values(ascending=False).head(4)
        st.markdown("**Event probabilities**")
        st.dataframe(probs.rename("probability").to_frame().style.format("{:.0%}"), use_container_width=True)
        if d["cues"]:
            st.markdown("**Severity cues:** " + ", ".join(f"`{c}`" for c in d["cues"]))


def _module_reactions(res: dict, before: dict | None) -> None:
    runs = res.get("stress_runs") or []
    if runs:
        r = runs[0]
        cap = r["capital"]
        st.markdown(f"<div class='rp-callout'>⚠ <b>Module B stress test triggered</b> - {r['scenario']} "
                    f"(scale {r['scale']:.2f}): portfolio P&L <b>${r['pnl'] / 1e6:,.1f}m</b> ({r['pnl_pct']:.2%}), "
                    f"CET1 {cap['cet1_ratio_before']:.2%} → {cap['cet1_ratio_after']:.2%}. See the Module B tab.</div>",
                    unsafe_allow_html=True)
    else:
        why = res.get("stress_note") or "below the impact threshold, not a stress-eligible event, or suppressed"
        st.markdown(f"<div class='rp-callout ok'>✓ No stress test: {why}.</div>", unsafe_allow_html=True)
    after = get("/module-a/state")
    if before and after:
        b = {w["ticker"]: w["weight"] for w in before["weights"]}
        moved = {w["ticker"]: w["weight"] - b[w["ticker"]] for w in after["weights"] if abs(w["weight"] - b[w["ticker"]]) > 1e-5}
        if moved:
            st.markdown("**Module A reaction (weight change, percentage points):** "
                        + ", ".join(f"{t} {v * 100:+.2f}pp" for t, v in sorted(moved.items(), key=lambda x: x[1])))


def live_feed(refresh: int) -> None:
    @st.fragment(run_every=refresh)
    def _feed() -> None:
        stats = get("/stats") or {}
        rows = get("/signals", limit=400) or []
        eng = stats.get("engine", {})
        runs = (get("/module-b/runs", limit=200) or {}).get("runs", [])
        high = sum(r["impact_score"] > 7 for r in rows)
        k = st.columns(6)
        k[0].metric("Signals stored", f"{stats.get('signals_stored', 0):,}")
        k[1].metric("High-impact (>7)", f"{high:,}", help="among the latest 400 signals")
        k[2].metric("Stress tests run", f"{len(runs):,}")
        k[3].metric("Engine tier", eng.get("model_tier", "loading…"))
        lat = sorted(r["latency_ms"] for r in rows if r.get("latency_ms"))
        k[4].metric("Engine latency p50", f"{lat[len(lat) // 2]:.0f} ms / doc" if lat else "n/a",
                    help="engine processing time per document, measured when each signal was produced")
        now = pd.to_datetime(stats.get("clock", {}).get("now")) if stats.get("clock") else None
        k[5].metric("Clock (UTC)", now.strftime("%d %b %Y %H:%M") if now is not None else "-")
        st.markdown("##### Latest signals")
        _signal_table(rows[:60])
        if rows:
            df = pd.DataFrame([{"event": r["event_type"], "impact": r["impact_score"], "type": r["source_type"]}
                               for r in rows])
            c1, c2 = st.columns(2)
            c1.plotly_chart(charts.bars_h(df["event"].value_counts().to_dict(), "Event mix (latest 400 signals)",
                                          height=330), use_container_width=True)
            c2.plotly_chart(charts.histogram(df["impact"], "Impact score distribution", "impact", height=330),
                            use_container_width=True)
    _feed()
