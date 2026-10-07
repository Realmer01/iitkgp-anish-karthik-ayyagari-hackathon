import pytest

from riskpulse.nlp import impact
from riskpulse.schema import Entity

BA = Entity(type="company", id="BA", name="Boeing Co.", ticker="BA")
AAPL = Entity(type="company", id="AAPL", name="Apple Inc.", ticker="AAPL")
MARKET = Entity(type="market", id="MARKET", name="Broad market")


def score(**over):
    kw = dict(event_type="General News", event_conf=0.8, sentiment=-0.3, sentiment_conf=0.7,
              text="Shares move in early trading", entity=BA, source_type="news", publisher="Reuters",
              n_sources=1, mentions=1, novel=True)
    kw.update(over)
    return impact.score(**kw)[0]


def test_range_and_rounding():
    for kw in ({}, {"event_type": "Credit Event", "sentiment": -1.0, "sentiment_conf": 1.0,
                    "text": "Bankruptcy and default as shares plunge 40%", "entity": MARKET, "n_sources": 30,
                    "mentions": 50}):
        s = score(**kw)
        assert 1.0 <= s <= 10.0 and round(s, 1) == s


@pytest.mark.parametrize("worse,better", [
    ({"event_type": "Credit Event"}, {"event_type": "General News"}),
    ({"sentiment": -0.9}, {"sentiment": -0.2}),
    ({"sentiment": -0.6}, {"sentiment": 0.6}),                   # bad news weighs more than good news
    ({"text": "Boeing files for bankruptcy"}, {"text": "Boeing shares move"}),
    ({"text": "Shares plunge 25% in a day"}, {"text": "Shares move 1% in a day"}),
    ({"entity": MARKET}, {"entity": BA}),
    ({"entity": AAPL}, {"entity": BA}),                           # larger index weight, more systemic
    ({"n_sources": 6}, {"n_sources": 1}),
    ({"entity_signals_24h": 12}, {"entity_signals_24h": 1}),     # a name in the news cycle moves more
    ({"publisher": "Reuters"}, {"publisher": "random-blog.example"}),
])
def test_monotonic(worse, better):
    assert score(**worse) >= score(**better)


def test_drivers_explain_the_score():
    s, drivers = impact.score(event_type="Geopolitical", event_conf=0.9, sentiment=-0.8, sentiment_conf=0.9,
                              text="New sanctions and tariffs escalate the trade war", entity=MARKET,
                              source_type="news", publisher="Reuters")
    b, w, _ = impact.model_params()
    assert drivers.logit == pytest.approx(b + sum(drivers.contributions.values()), abs=1e-3)
    assert set(drivers.features) == set(impact.FEATURES) and drivers.cues
    assert all(v >= 0 for v in w.values())  # non-negative weights keep the score monotonic
