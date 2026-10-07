"""RiskPulse dashboard (Streamlit). Talks to the API only:  python run.py demo"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from riskpulse import config  # noqa: E402
from riskpulse.dashboard import evaluation_view, feed_view, module_a_view, module_b_view  # noqa: E402
from riskpulse.dashboard.client import API, get, post  # noqa: E402
from riskpulse.dashboard.theme import CSS, install_template  # noqa: E402

st.set_page_config(page_title="RiskPulse - AI/NLP Risk Engine", page_icon="📡", layout="wide")
install_template()
st.markdown(CSS, unsafe_allow_html=True)
REFRESH = int(config.get("dashboard.refresh_seconds", 3))


def sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📡 RiskPulse")
        st.caption("Real-time AI/NLP risk signals → index rebalancing & stress testing")
        health = get("/health")
        if not health:
            st.error(f"API not reachable at {API}. Start it with `python run.py demo`.")
            return
        stats = get("/stats") or {}
        clock = stats.get("clock", {})
        st.markdown(f"**Mode:** `{stats.get('mode')}` · **Engine:** "
                    f"`{(stats.get('engine') or {}).get('model_tier', 'loading')}`"
                    + ("" if health.get("engine_ready") else " ⏳"))
        st.markdown(f"**Clock:** {clock.get('now', '')[:16].replace('T', ' ')} UTC")
        if clock.get("mode") == "replay":
            st.caption(f"Replay window {stats['window'][0][:10]} → {stats['window'][1][:10]}"
                       + (" · finished" if stats.get("replay_finished") else ""))
            c1, c2, c3 = st.columns(3)
            if c1.button("⏸", help="pause", use_container_width=True):
                post("/replay/pause")
            if c2.button("▶", help="resume", use_container_width=True):
                post("/replay/resume")
            if c3.button("⟲", help="restart the replay", use_container_width=True):
                post("/replay/restart")
            speeds = [600, 1800, 3600, 5400, 10800, 21600]
            cur = int(clock.get("speed", 5400))
            sp = st.select_slider("Speed (simulated s per s)", speeds, value=cur if cur in speeds else 5400)
            if sp != cur:
                post("/replay/speed", value=sp)
        st.markdown(f"[API docs]({API}/docs) · [signals stream]({API}/stream)")
        st.caption("Sentiment colours: red = negative, blue = positive (colour-blind safe).")


sidebar()
st.title("RiskPulse · AI/NLP Financial Risk Engine")
tab1, tab2, tab3, tab4 = st.tabs(["Live risk feed", "Module A · Index rebalancer", "Module B · Stress test",
                                  "Model evaluation"])
with tab1:
    feed_view.analyze_box()
    st.divider()
    feed_view.live_feed(REFRESH)
with tab2:
    module_a_view.live_module_a(REFRESH * 2)
    st.divider()
    module_a_view.backtest_section()
with tab3:
    module_b_view.live_module_b(REFRESH * 2)
    st.divider()
    module_b_view.what_if_panel()
with tab4:
    evaluation_view.render()
