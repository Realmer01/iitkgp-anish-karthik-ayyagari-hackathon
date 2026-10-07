"""Entity linking: which companies, sector or market does a text talk about?

For each company in the universe, in order:
  1. cashtags              $AAPL                      (social convention; "$GOOG" -> GOOGL)
  2. exchange prefixes     NYSE:BA, (NASDAQ: NVDA)
  3. bare tickers          NVDA                       only where `bare_ticker` is true - "BA", "GS",
                                                      "KO", "CAT" are ordinary words or abbreviations
  4. aliases               Microsoft, JPMorgan, iPhone, Jamie Dimon       (case-insensitive)
  5. case-sensitive        "Apple", "Meta"            so "apple pie" / "meta-analysis" do not match
A false-positive guard then removes alias hits inside known non-company phrases.
One text can yield several companies. With no company, the text links to a sector ("chipmakers")
or - for macro / market-wide / geopolitical language - to the MARKET entity. Texts with no link
are dropped as noise by the engine.
"""
from __future__ import annotations

import re
from functools import lru_cache

from riskpulse.nlp import lexicons as lx
from riskpulse.schema import Entity
from riskpulse.universe import Company, by_ticker, companies

CASHTAG_RX = re.compile(r"\$([A-Z]{1,5})\b")
EXCHANGE_RX = re.compile(r"\b(?:NYSE|NASDAQ|Nasdaq|NYSEARCA)\s*:\s*([A-Z]{1,5})\b")
BARE_RX = re.compile(r"(?<![\w$])([A-Z]{2,5})(?![\w])")
FALSE_POSITIVES = re.compile(
    r"\b(apple (pie|juice|cider|orchard|tree)s?|big apple|amazon (rainforest|river|basin)|"
    r"meta[- ](analysis|data)|goldman (prize|environmental)|coca[- ]leaf)\b", re.I)

# Banks in the universe often appear as the ANALYST, not the subject: "Hilton downgraded by Goldman Sachs",
# "JPMorgan sees recession odds at 60%". Those mentions are not news about GS / JPM.
_BROKER = r"(?:Goldman Sachs|Goldman|JPMorgan|JP Morgan|J\.P\. Morgan)"
BROKER_CONTEXT = re.compile(
    rf"\b(?:by|at|from|per|according to)\s+(?:analysts?\s+(?:at|from)\s+)?(?P<a>{_BROKER})\b"
    rf"|\b(?P<b>{_BROKER})(?:'s|’s)?\s+(?:analysts?|strategists?|economists?|research|upgrades?|downgrades?|initiates|"
    rf"reiterates)\b"
    rf"|\b(?P<c>{_BROKER})(?:'s|’s)?\s+(?:raises|cuts|lifts|lowers|trims|slashes|hikes|boosts)\b(?=.{{0,40}}"
    rf"\b(?:target|forecast|outlook|estimates?|rating|odds|view|call)s?\b)"
    rf"|\b(?P<d>{_BROKER})\s+(?:says|sees|expects|predicts|forecasts|warns)\b(?=.{{0,60}}\b(?:S&P|stocks|equities|oil|"
    rf"Fed|recession|inflation|rates|yields|economy|markets?|dollar|gold)\b)", re.I)


def broker_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for m in BROKER_CONTEXT.finditer(text):
        g = next(k for k in "abcd" if m.group(k))
        spans.append(m.span(g))
    return spans


def _alias_rx(aliases: tuple[str, ...], flags: int) -> re.Pattern | None:
    if not aliases:
        return None
    body = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
    return re.compile(rf"(?<![\w$])(?:{body})(?!\w)", flags)


@lru_cache(maxsize=1)
def _patterns() -> dict[str, tuple[re.Pattern | None, re.Pattern | None]]:
    return {c.ticker: (_alias_rx(c.aliases, re.IGNORECASE), _alias_rx(c.case_sensitive_aliases, 0))
            for c in companies()}


def company_entity(c: Company, matched: str) -> Entity:
    return Entity(type="company", id=c.ticker, ticker=c.ticker, name=c.name, sector=c.sector, country=c.country,
                  matched=matched)


def link(text: str) -> list[Entity]:
    idx = by_ticker()
    found: dict[str, Entity] = {}
    guarded = FALSE_POSITIVES.sub(" ", text)

    for rx in (CASHTAG_RX, EXCHANGE_RX):
        for m in rx.finditer(text):
            sym = "GOOGL" if m.group(1) == "GOOG" else m.group(1)
            if sym in idx and sym not in found:
                found[sym] = company_entity(idx[sym], m.group(0))
    for m in BARE_RX.finditer(text):
        c = idx.get(m.group(1))
        if c and c.bare_ticker and c.ticker not in found:
            found[c.ticker] = company_entity(c, m.group(1))
    brokers = broker_spans(guarded)
    for c in companies():
        if c.ticker in found:
            continue
        ci, cs = _patterns()[c.ticker]
        hits = [m for rx in (ci, cs) if rx for m in rx.finditer(guarded)]
        subject = [m for m in hits if not any(a <= m.start() < b for a, b in brokers)]
        if subject:
            found[c.ticker] = company_entity(c, subject[0].group(0))
    if found:
        return list(found.values())

    for sector, rx in lx.SECTOR_RX.items():
        m = rx.search(text)
        if m:
            return [Entity(type="sector", id=sector, name=sector, sector=sector, matched=m.group(0))]
    m = lx.MARKET_RX.search(text) or lx.GEOPOLITICAL_RX.search(text)
    if m:
        return [Entity(type="market", id="MARKET", name="Broad market", matched=m.group(0))]
    return []


def regions(text: str) -> list[str]:
    return [region for region, rx in lx.REGION_RX.items() if rx.search(text)]


def mention_spans(text: str, entity: Entity) -> list[tuple[int, int]]:
    """Character spans where a company is mentioned (used for entity-level sentiment)."""
    if entity.type != "company":
        return []
    ci, cs = _patterns()[entity.id]
    rxs = [r for r in (ci, cs) if r] + [re.compile(rf"\$?\b{re.escape(entity.id)}\b")]
    return [m.span() for rx in rxs for m in rx.finditer(text)]
