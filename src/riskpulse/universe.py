"""The mock index universe (config/universe.yaml), shared by the engine and both modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from riskpulse.config import load_yaml


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    sector: str
    country: str
    market_cap_bn: float
    bare_ticker: bool = True
    aliases: tuple[str, ...] = field(default_factory=tuple)
    case_sensitive_aliases: tuple[str, ...] = field(default_factory=tuple)


@lru_cache(maxsize=1)
def companies() -> tuple[Company, ...]:
    out = []
    for c in load_yaml("universe.yaml")["companies"]:
        out.append(Company(ticker=c["ticker"], name=c["name"], sector=c["sector"], country=c["country"],
                           market_cap_bn=float(c["market_cap_bn"]), bare_ticker=bool(c.get("bare_ticker", True)),
                           aliases=tuple(c.get("aliases", [])),
                           case_sensitive_aliases=tuple(c.get("case_sensitive_aliases", []))))
    return tuple(out)


def tickers() -> list[str]:
    return [c.ticker for c in companies()]


@lru_cache(maxsize=1)
def by_ticker() -> dict[str, Company]:
    return {c.ticker: c for c in companies()}


def sector_members() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in companies():
        out.setdefault(c.sector, []).append(c.ticker)
    return out
