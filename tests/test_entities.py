import pytest

from riskpulse.nlp.entities import link, mention_spans, regions


def ids(text):
    return [e.id for e in link(text)]


@pytest.mark.parametrize("text,expected", [
    ("$AAPL and $TSLA rip higher at the open", ["AAPL", "TSLA"]),
    ("Shares of NYSE:BA slid after the FAA statement", ["BA"]),
    ("NVDA beats on data-centre revenue", ["NVDA"]),
    ("Jamie Dimon warns of a stormy economy", ["JPM"]),
    ("Apple unveils a new iPhone", ["AAPL"]),
    ("$GOOG hits a record", ["GOOGL"]),
])
def test_company_links(text, expected):
    assert ids(text) == expected


@pytest.mark.parametrize("text", [
    "Grandma's apple pie recipe wins the county fair",
    "A meta-analysis of vaccine trials was published today",
    "Deforestation in the Amazon rainforest hit a record",
    "The BA degree programme opens applications in May",
    "My cat knocked over the coffee",
])
def test_false_positive_guard(text):
    assert not any(e.type == "company" for e in link(text))


def test_sector_and_market_fallbacks():
    assert link("Chipmakers slump as export curbs widen")[0].id == "Information Technology"
    assert link("Fed raises interest rates by 50 basis points")[0].id == "MARKET"
    assert link("Invasion fears grip Europe") and link("Invasion fears grip Europe")[0].type == "market"
    assert link("I had a lovely walk in the park today") == []


def test_entity_fields_and_spans():
    e = link("Boeing shares slump")[0]
    assert (e.ticker, e.sector, e.country) == ("BA", "Industrials", "US")
    assert mention_spans("Boeing and Airbus", e) == [(0, 6)]


def test_regions():
    assert set(regions("Russia hit with EU sanctions as oil from the Middle East surges")) >= {
        "Russia/CIS", "Europe", "Middle East"}
