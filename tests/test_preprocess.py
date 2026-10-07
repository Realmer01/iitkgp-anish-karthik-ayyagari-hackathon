from datetime import UTC, datetime, timedelta

import numpy as np

from riskpulse.nlp.preprocess import StoryClusterer, clean, is_english, normalize_social, strip_publisher

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_clean_strips_html_urls_hashtag_tail_and_upper_cases_cashtags():
    raw = "<p>$tsla &amp; $aapl rally https://t.co/x</p> # Stocks #Investing"
    assert clean(raw) == "$TSLA & $AAPL rally"


def test_social_slang_and_emoji_are_rewritten():
    out = normalize_social("$TSLA to the moon 🚀 bagholders rekt")
    assert "rally" in out and "investors stuck with losses" in out and "heavy losses" in out


def test_strip_publisher_and_language_filter():
    assert strip_publisher("Fed holds rates - Reuters", "Reuters") == "Fed holds rates"
    assert is_english("The Fed held rates steady on Wednesday")
    assert not is_english("Банк России сохранил ключевую ставку")


def test_story_clusterer_dedup_and_corroboration():
    sc = StoryClusterer(dedup_threshold=0.95, story_threshold=0.8)
    a = sc.assign(unit([1, 0, 0]), "Reuters", T0, "Boeing halts 737 production")
    assert a.novel and a.mention_count == 1
    assert sc.assign(unit([1, 0, 0]), "Reuters", T0, "Boeing halts 737 production").duplicate  # exact repeat
    b = sc.assign(unit([0.97, 0.24, 0]), "AP", T0 + timedelta(hours=1), "Boeing stops 737 output")
    assert not b.duplicate and b.story_id == a.story_id and b.sources == ["Reuters", "AP"] and b.mention_count == 2
    c = sc.assign(unit([0, 1, 0]), "AP", T0, "Fed raises rates")
    assert c.novel and c.story_id != a.story_id


def test_syndicated_copy_adds_reach_not_confirmation():
    sc = StoryClusterer(dedup_threshold=0.95, story_threshold=0.85)
    sc.assign(unit([1, 0, 0]), "AP", T0, "Fed holds rates steady")
    dup = sc.assign(unit([1, 0, 0]), "Yahoo", T0 + timedelta(minutes=5), "Fed holds rates steady")
    assert dup.duplicate and dup.mention_count == 2 and dup.sources == ["AP"]


def test_story_expires_from_first_report_despite_continuous_coverage():
    """Regression: one company's stream of news must not grow a months-long 'story'."""
    sc = StoryClusterer(story_threshold=0.85, window_hours=24)
    first = sc.assign(unit([1, 0]), "a", T0, "Nvidia story number 0")
    for h in range(1, 30):
        last = sc.assign(unit([1, 0]), f"p{h}", T0 + timedelta(hours=h * 2), f"Nvidia story number {h}")
    assert last.story_id != first.story_id and last.mention_count < 13


def test_story_window_expires():
    sc = StoryClusterer(window_hours=1)
    a = sc.assign(unit([1, 0]), "x", T0, "first text here")
    b = sc.assign(unit([1, 0]), "y", T0 + timedelta(hours=3), "other text here")
    assert a.story_id != b.story_id
