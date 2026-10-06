"""End-to-end smoke test of the RiskEngine in the lite tier (no torch)."""
import numpy as np
import pytest

from riskpulse.engine import RiskEngine
from riskpulse.nlp.preprocess import normalize_social
from riskpulse.nlp.sentiment import policy_polarity, split_clauses
from riskpulse.schema import RawDocument, RiskSignal


@pytest.fixture(scope="module")
def engine():
    return RiskEngine("lite")


def doc(text, publisher="Reuters", source_type="news"):
    return RawDocument(source="test", source_type=source_type, publisher=publisher, text=text)


def test_end_to_end_signals(engine):
    sig = engine.process([doc("Russia launches missile strikes; US and EU impose sweeping new sanctions"),
                          doc("Boeing shares plunge 9% after FAA grounds 737 MAX jets"),
                          doc("Lovely weather in the park today with the family")])
    assert all(isinstance(s, RiskSignal) for s in sig)
    by_entity = {s.entity.id: s for s in sig}
    assert set(by_entity) == {"MARKET", "BA"}                      # the noise post was dropped
    geo = by_entity["MARKET"]
    assert geo.event_type == "Geopolitical" and geo.sentiment_score < 0 and geo.impact_score > 7
    assert geo.model_tier == "lite" and geo.impact_drivers.contributions
    assert engine.stats["dropped_no_entity"] == 1


def test_duplicates_collapse_and_confirmation_counts(engine):
    a = engine.process([doc("Pfizer wins FDA approval for new RSV vaccine", "Reuters")])
    b = engine.process([doc("Pfizer wins FDA approval for new RSV vaccine", "Reuters")])   # exact repeat
    c = engine.process([doc("FDA approves Pfizer's new RSV vaccine", "CNBC")])            # confirmation
    assert a and not b
    assert c and c[0].story_id == a[0].story_id and c[0].sources == ["Reuters", "CNBC"] and c[0].mention_count == 2


def test_company_level_sentiment_in_mixed_headline(engine):
    sig = {s.entity.id: s for s in engine.process([doc("Apple beats estimates while Boeing slumps on weak deliveries")])}
    assert sig["AAPL"].sentiment_score > sig["BA"].sentiment_score


def test_analyze_text_does_not_touch_story_state(engine):
    n = len(engine.stories.stories)
    assert engine.analyze_text("Fed raises interest rates by 75 basis points")
    assert len(engine.stories.stories) == n


def test_policy_polarity_and_clause_split():
    p = np.array([0.1, 0.1, 0.8])
    assert policy_polarity("Fed raises interest rates again", p)[0] == 0.8
    assert policy_polarity("Fed cuts rates", p)[2] == 0.8
    assert split_clauses("Apple beats estimates while Boeing slumps") == ["Apple beats estimates", "Boeing slumps"]
    assert "rallied" in normalize_social("bears rekt")
