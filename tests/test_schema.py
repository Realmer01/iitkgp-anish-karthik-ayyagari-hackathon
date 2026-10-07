from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from riskpulse.schema import Entity, ImpactDrivers, RawDocument, RiskSignal, SourceType


def make_signal(**over):
    base = dict(signal_id="s1", published_at=datetime.now(UTC), source="test", source_type="news",
                doc_ids=["d1"], story_id="S1", headline="Boeing shares slump",
                entity=Entity(type="company", id="BA", name="Boeing Co.", ticker="BA"),
                sentiment_score=-0.6, sentiment_label="negative", sentiment_confidence=0.8,
                event_type="Operational Risk", event_confidence=0.7, impact_score=6.2,
                impact_drivers=ImpactDrivers(features={}, contributions={}, logit=0.3), model_tier="lite")
    base.update(over)
    return RiskSignal(**base)


def test_document_id_is_deterministic_and_utc():
    naive = datetime(2026, 1, 1, 12, 0)
    a = RawDocument(source="x", source_type=SourceType.NEWS, title="Fed holds rates", published_at=naive)
    b = RawDocument(source="x", source_type=SourceType.NEWS, title="Fed holds rates", published_at=naive)
    assert a.id == b.id and len(a.id) == 16
    assert a.published_at.tzinfo is not None and a.published_at.utcoffset() == timedelta(0)


def test_content_joins_title_and_text():
    d = RawDocument(source="x", source_type="news", title="Headline", text="Body text")
    assert d.content == "Headline. Body text"
    assert d.origin == "x"


@pytest.mark.parametrize("field,value", [("sentiment_score", 1.5), ("sentiment_score", -1.01),
                                         ("impact_score", 0.5), ("impact_score", 10.5),
                                         ("event_type", "Alien Invasion"), ("sentiment_confidence", 1.2)])
def test_signal_rejects_out_of_range(field, value):
    with pytest.raises(ValidationError):
        make_signal(**{field: value})


def test_signal_round_trips_json():
    s = make_signal()
    again = RiskSignal.model_validate_json(s.model_dump_json())
    assert again == s and again.schema_version == "1.0"
