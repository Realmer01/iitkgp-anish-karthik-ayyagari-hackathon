import os
from datetime import datetime, timezone

import pytest

os.environ.setdefault("RISKPULSE_TIER", "lite")   # the test-suite never needs torch

from riskpulse.schema import Entity, ImpactDrivers, RiskSignal  # noqa: E402


@pytest.fixture
def make_signal():
    def _make(i: int = 0, **over) -> RiskSignal:
        base = dict(signal_id=f"s{i}", published_at=datetime(2025, 4, 3, 14, i % 60, tzinfo=timezone.utc),
                    source="test", source_type="news", publisher="Reuters", doc_ids=[f"d{i}"], story_id=f"S{i}",
                    headline=f"headline {i}", entity=Entity(type="company", id="BA", name="Boeing Co.", ticker="BA",
                                                            sector="Industrials"),
                    sentiment_score=-0.5, sentiment_label="negative", sentiment_confidence=0.8,
                    event_type="Geopolitical", event_confidence=0.9, impact_score=8.0,
                    impact_drivers=ImpactDrivers(features={"credibility": 1.0}, contributions={}, logit=1.0),
                    model_tier="lite")
        base.update(over)
        return RiskSignal(**base)
    return _make
