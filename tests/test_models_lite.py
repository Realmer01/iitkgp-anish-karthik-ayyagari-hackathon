"""Sanity checks for the lite-tier models (no torch required)."""
import numpy as np
import pytest

from riskpulse.nlp.events import EventClassifier, KeywordBaseline, rule_override
from riskpulse.nlp.sentiment import LiteSentiment, chunked_probs, summarise
from riskpulse.schema import EVENT_TYPES


@pytest.fixture(scope="module")
def sentiment():
    return LiteSentiment()


@pytest.fixture(scope="module")
def events():
    return EventClassifier("lite")


def test_lite_sentiment_direction(sentiment):
    neg, pos = sentiment.probs(["Boeing shares plunge after deadly crash",
                                "Nvidia beats estimates and shares soar to a record"])
    assert summarise(neg)[1] == "negative" and summarise(pos)[1] == "positive"
    assert summarise(neg)[0] < 0 < summarise(pos)[0]


def test_finance_lexicon_fixes_generic_vader(sentiment):
    assert summarise(sentiment.probs(["Analysts downgrade Pfizer"])[0])[0] < 0


def test_probabilities_are_valid(sentiment):
    p = sentiment.probs(["Fed holds rates", "Stocks crash", "Apple rallies"])
    assert np.allclose(p.sum(axis=1), 1) and (p >= 0).all()


def test_chunked_probs_averages(sentiment):
    p = chunked_probs(sentiment, "Revenue surged. But the outlook was cut and shares plunged.", "news",
                      lambda xs: None)
    assert np.isclose(p.sum(), 1)


@pytest.mark.parametrize("text,expected", [
    ("Northwind Retail files for Chapter 11 bankruptcy protection", "Credit Event"),
    ("Microsoft agrees to acquire a gaming studio for $7 billion", "Merger & Acquisition"),
    ("Fed raises interest rates by 75 basis points to fight inflation", "Macroeconomic"),
    ("Ransomware attack forces pipeline operator to shut down", "Operational Risk"),
])
def test_rules_and_model_agree_on_clear_cases(events, text, expected):
    assert events.predict([text])[0]["event_type"] == expected


def test_event_output_shape(events):
    out = events.predict(["Tesla shares are up 3% in pre-market trading"])[0]
    assert out["event_type"] in EVENT_TYPES and 0 <= out["event_confidence"] <= 1
    assert abs(sum(out["event_probs"].values()) - 1) < 0.05


def test_keyword_baseline_defaults_to_general_news():
    p = KeywordBaseline().probs(["Nothing to see here today"])[0]
    assert EVENT_TYPES[int(np.argmax(p))] == "General News"
    assert rule_override("Nothing to see here") is None
