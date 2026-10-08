from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from riskpulse.bus import InProcessSubscriber, SignalBus
from riskpulse.module_b import scenarios
from riskpulse.module_b.monitor import StressMonitor
from riskpulse.module_b.portfolio import annuity, bond_analytics, positions
from riskpulse.module_b.revaluation import revalue
from riskpulse.module_b.stress import what_if


def test_bond_price_duration_against_hand_calculation():
    # 2-year 5% annual bond at 5% yield prices at par; Macaulay D = (5/1.05 + 2*105/1.05^2)/100
    price, mod_dur, conv = bond_analytics(0.05, 0.05, 2.0)
    assert price == pytest.approx(100.0, abs=1e-9)
    assert mod_dur == pytest.approx((5 / 1.05 + 2 * 105 / 1.05 ** 2) / 100 / 1.05, rel=1e-9)
    # +100bp: the duration-convexity approximation matches full repricing closely
    exact = bond_analytics(0.05, 0.06, 2.0)[0] / 100 - 1
    approx = -mod_dur * 0.01 + 0.5 * conv * 0.01 ** 2
    assert approx == pytest.approx(exact, abs=2e-5)


def test_annuity_and_impact_scale():
    assert annuity(0.0, 3.0) == pytest.approx(3.0)
    assert scenarios.impact_scale(7) == pytest.approx(0.7) and scenarios.impact_scale(10) == pytest.approx(1.3)
    assert scenarios.impact_scale(1) == pytest.approx(0.1)


def test_routing_by_headline_cues():
    assert scenarios.route("Geopolitical", "New tariffs on Chinese imports")["scenario"] == "trade_war"
    assert scenarios.route("Geopolitical", "Troops cross the border")["scenario"] == "conflict"
    assert scenarios.route("Macroeconomic", "Recession fears grow")["scenario"] == "growth_shock"
    assert scenarios.route("Macroeconomic", "Fed signals more hikes")["scenario"] == "rates_shock"


def test_dv01_sign_and_cds_direction():
    pos = positions()
    up = scenarios.ShockSet("t", "rates up", "", 1.0, {"rates_bp": {"default": 100}})
    res = revalue(pos, up)
    payer = res[(res.instrument == "Interest Rate Swap") & (res.side == "Pay Fixed")]
    receiver = res[(res.instrument == "Interest Rate Swap") & (res.side == "Receive Fixed")]
    assert (payer.pnl_rates > 0).all() and (receiver.pnl_rates < 0).all()
    assert (res[res.asset_class == "Bond"].pnl_rates < 0).all()
    wide = revalue(pos, scenarios.ShockSet("t", "spreads", "", 1.0, {"spreads_bp": dict.fromkeys(
        ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"], 100)}))
    assert (wide[wide.cs01 > 0].pnl_spread > 0).all()     # bought protection gains


def test_stress_run_outputs_and_named_overlay():
    r = what_if("Credit Event", 9.0, "Boeing downgraded to junk", ticker="BA", sector="Industrials")
    assert r["value_after"] == pytest.approx(r["value_before"] + r["pnl"])
    assert r["pnl"] < 0 and r["delta_expected_loss"] >= 0 and r["top_contributors"]
    assert r["capital"]["cet1_ratio_after"] < r["capital"]["cet1_ratio_before"]
    assert "naive" in r and sum(r["by_asset_class"].values()) == pytest.approx(r["pnl"], abs=10)
    s = scenarios.build("Credit Event", 9.0, "", "company", "BA", "Industrials")
    assert s.spread_bp("BBB", "Industrials", "North America", "BA") > s.spread_bp("BBB", "Industrials", "North America")


def test_trigger_threshold_dedup_and_cooldown(make_signal):
    clock = {"t": datetime(2025, 4, 3, tzinfo=UTC)}
    mon = StressMonitor(now=lambda: clock["t"])
    bus = SignalBus()
    mon.attach(InProcessSubscriber(bus))
    geo = dict(entity={"type": "market", "id": "MARKET", "name": "Broad market"}, headline="New tariffs announced")
    bus.publish([make_signal(1, impact_score=6.9, **geo)])                      # below threshold
    assert not mon.runs
    bus.publish([make_signal(2, impact_score=8.2, story_id="A", **geo)])
    assert len(mon.runs) == 1 and mon.runs[0]["scenario_key"] == "trade_war"
    bus.publish([make_signal(3, impact_score=8.5, story_id="A", **geo)])        # same story
    bus.publish([make_signal(4, impact_score=8.5, story_id="B", **geo)])        # same scenario within cooldown
    assert len(mon.runs) == 1 and mon.suppressed == {"story_dedup": 1, "cooldown": 1}
    clock["t"] += timedelta(hours=2)
    bus.publish([make_signal(5, impact_score=9.3, story_id="C", **geo)])        # escalation: >= 8.2 + 1.0
    assert len(mon.runs) == 2
    bus.publish([make_signal(6, impact_score=9.5, story_id="D", **geo)])        # not 1 point above 9.3
    assert len(mon.runs) == 2
    clock["t"] += timedelta(hours=7)                                            # cooldown (6h) has passed
    bus.publish([make_signal(7, impact_score=7.5, story_id="E", **geo)])
    assert len(mon.runs) == 3
    assert isinstance(pd.DataFrame(mon.runs[0]["top_contributors"]), pd.DataFrame)


def test_low_confidence_and_ineligible_events_do_not_trigger(make_signal):
    mon = StressMonitor()
    assert not mon.wants(make_signal(1, impact_score=9.5, event_confidence=0.2))
    assert not mon.wants(make_signal(2, impact_score=9.5, event_type="Product Launch"))
    assert mon.wants(make_signal(3, impact_score=9.5))
