from datetime import datetime, timedelta, timezone

from riskpulse.clock import ReplayClock
from riskpulse.ingestion.news import parse_date, parse_gdelt_articles
from riskpulse.ingestion.replay import ReplaySource
from riskpulse.schema import RawDocument

T0 = datetime(2025, 4, 2, 20, 0, tzinfo=timezone.utc)


def test_parse_date_formats():
    assert parse_date("Wed, 02 Apr 2025 20:00:00 GMT") == T0
    assert parse_date("20250402T200000Z") == T0
    assert parse_date("2025-04-02T20:00:00Z") == T0
    assert parse_date(None).tzinfo is not None


def test_parse_gdelt_articles():
    docs = parse_gdelt_articles([{"title": "Tariffs hit markets", "domain": "reuters.com",
                                  "seendate": "20250402T200000Z", "language": "English", "sourcecountry": "US"}])
    assert docs[0].publisher == "reuters.com" and docs[0].published_at == T0 and docs[0].lang == "en"


def test_replay_releases_documents_in_simulated_time():
    docs = [RawDocument(source="t", source_type="news", title=f"headline {i}", published_at=T0 + timedelta(hours=i))
            for i in range(5)]
    clock = ReplayClock(start=T0 - timedelta(minutes=1), speed=0)   # frozen clock
    src = ReplaySource(docs, clock)
    assert src.poll() == []
    clock.jump(T0 + timedelta(hours=2, minutes=1))
    assert [d.title for d in src.poll()] == ["headline 0", "headline 1", "headline 2"]
    assert src.poll() == []
    clock.jump(T0 + timedelta(days=1))
    assert len(src.poll()) == 2 and src.finished


def test_replay_clock_pause_and_speed():
    clock = ReplayClock(start=T0, speed=0)
    clock.pause()
    frozen = clock.now()
    clock.set_speed(1e6)
    assert clock.now() == frozen and clock.paused
    clock.resume()
    assert clock.now() >= frozen
