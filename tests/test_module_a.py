from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from riskpulse.aggregate import SentimentBook
from riskpulse.bus import InProcessSubscriber, SignalBus
from riskpulse.module_a import backtest as bt
from riskpulse.module_a.live import LiveRebalancer
from riskpulse.module_a.policy import PolicyParams, base_weights, bounds, rebalance, target_weights
from riskpulse.universe import tickers

P = PolicyParams.from_config()


def test_base_weights_are_capped_market_cap():
    b = base_weights(P)
    assert b.sum() == pytest.approx(1.0) and b.max() <= P.base_cap + 1e-9 and b.min() >= P.base_floor - 1e-9
    assert b["AAPL"] > b["KO"]


@pytest.mark.parametrize("seed", range(5))
def test_target_weights_respect_all_constraints(seed):
    b = base_weights(P)
    s = pd.Series(np.random.default_rng(seed).uniform(-1, 1, len(b)), index=b.index)
    t = target_weights(b, s, P)
    lo, hi = bounds(b, P)
    assert t.sum() == pytest.approx(1.0) and not t.isna().any() and (t >= 0).all()
    assert ((t >= lo - 1e-9) & (t <= hi + 1e-9)).all()
    assert ((t - b).abs() <= P.max_active_weight + 1e-9).all()


def test_tilt_direction():
    b = base_weights(P)
    t = target_weights(b, pd.Series({"MSFT": 0.9, "TSLA": -0.9}), P)
    assert t["MSFT"] > b["MSFT"] and t["TSLA"] < b["TSLA"]


def test_drift_band_and_turnover_cap():
    b = base_weights(P)
    tiny = b.copy()
    tiny["AAPL"] += 0.001
    tiny["MSFT"] -= 0.001
    assert rebalance(b, tiny, b, P)[1] == 0.0                         # inside the band: no trade
    big = target_weights(b, pd.Series(1.0, index=b.index).mul(np.linspace(-1, 1, len(b))), P)
    new, turnover = rebalance(b, big, b, P)
    assert 0 < turnover <= P.max_turnover + 1e-9 and new.sum() == pytest.approx(1.0)


def test_sentiment_book_decay_and_shrinkage():
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    book = SentimentBook(half_life_hours=12, k0=1.0)
    book.add("BA", -1.0, 1.0, t0)
    s0 = book.snapshot(t0).loc["BA", "sentiment"]
    assert s0 == pytest.approx(-0.5)                                  # shrunk: 1 unit of evidence vs k0 = 1
    assert book.snapshot(t0 + timedelta(hours=12)).loc["BA", "sentiment"] > s0   # decays toward 0


def test_backtest_has_no_look_ahead():
    """A signal available at X must not change weights for any bar starting at or before X."""
    idx = pd.date_range("2025-01-06 14:30", periods=40, freq="h", tz="UTC")
    bars = pd.DataFrame(100.0, index=idx, columns=tickers())
    x = idx[20]
    sig = pd.DataFrame([{"available_at": x, "published_at": x, "entity_type": "company", "entity_id": "TSLA",
                         "sentiment": -1.0, "sent_conf": 0.99, "f_credibility": 1.0, "impact": 9.0,
                         "mention_count": 1, "headline": "test"}] * 5)
    res = bt.run(sig, bars, start="2025-01-01", benchmarks=False)
    w = res["weights"]["TSLA"]
    assert (w[w.index <= x] == w.iloc[0]).all()                      # unchanged up to and including X
    assert w[w.index > x].iloc[0] < w.iloc[0]                        # reacts at the first later bar


def test_live_rebalancer_reacts_to_signals(make_signal):
    mod = LiveRebalancer(now=lambda: datetime(2025, 4, 3, 15, tzinfo=timezone.utc))   # replay-clock time
    mod.min_interval = 0
    bus = SignalBus()
    mod.attach(InProcessSubscriber(bus))
    before = mod.current["BA"]
    bus.publish([make_signal(i, sentiment_score=-0.95, sentiment_confidence=0.95) for i in range(5)])
    assert mod.current["BA"] < before and mod.current.sum() == pytest.approx(1.0)
    assert mod.state()["rebalances"] == 1
