import pytest

from riskpulse.nlp.framing import factuality


@pytest.mark.parametrize("text, score, label", [
    ("Global markets in turmoil as Trump tariffs wipe $2.5tn off Wall Street", 1.0, ""),
    ("Russia launches missile strikes on Ukraine; US and EU impose sweeping new sanctions", 1.0, ""),
    ("Oil output in May hit a record as OPEC+ added barrels", 1.0, ""),              # the month, not a hedge
    ("Boeing could not deliver any jets in March", 1.0, ""),                         # past inability, not speculation
    ("Can US agriculture withstand a trade war with China?", 0.0, "question"),
    ("Is the US economy likely to fall into recession - 7 Apr 2025", 0.0, "question"),
    ("What Were the Economic Effects of the Iran-Iraq War?", 0.0, "question"),
    ("The Fed issues its rate decision today. Here's what to expect", 0.0, "explainer"),
    ("3 Stocks That Could Benefit From New Trump Tariffs", 0.0, "listicle"),
    ("America may be headed for this rare type of economic crisis", 0.5, "hedged"),
    ("Why Microsoft Stock Is Sinking Today", 0.5, "explanatory"),
    ("Why a stock market selloff may trigger a US recession", 0.25, "explanatory+hedged"),
])
def test_factuality(text, score, label):
    assert factuality(text) == (score, label)
