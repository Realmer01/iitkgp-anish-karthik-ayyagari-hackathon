from riskpulse.universe import by_ticker, companies, sector_members, tickers


def test_universe_size_and_sectors():
    assert 10 <= len(companies()) <= 20
    assert len(set(tickers())) == len(tickers())
    assert len(sector_members()) >= 6


def test_ambiguous_tickers_need_cashtag():
    for t in ("BA", "GS", "KO", "CAT", "META", "WMT"):
        assert by_ticker()[t].bare_ticker is False
