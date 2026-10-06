"""Turn (event type, impact, headline, linked entities) into a concrete set of risk-factor shocks.

  1. route the event to a scenario in config/shocks.yaml ("tariffs" -> trade_war, "invasion" -> conflict)
  2. scale it by the Impact Score: scale = 0.7 at impact 7, 1.3 at impact 10 (linear, floored at 0.1)
  3. target it with the signal's entities: regions named in the headline get larger spread / PD shocks,
     the signal's sector gets larger shocks, and a named company gets an idiosyncratic overlay on every
     position that references it (its bonds, CDS, equity) - on top of a smaller systemic component.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from riskpulse.config import load_yaml


@lru_cache(maxsize=1)
def library() -> dict:
    return load_yaml("shocks.yaml")


def impact_scale(impact: float) -> float:
    return float(min(1.3, max(0.1, 0.7 + 0.2 * (impact - 7.0))))


@dataclass
class ShockSet:
    key: str
    name: str
    analogue: str
    scale: float
    spec: dict
    systemic: float = 1.0                       # share of the systemic scenario applied
    overlay: dict = field(default_factory=dict)  # name-specific overlay (already includes its own scale)
    target_ticker: str = ""
    target_sector: str = ""
    target_regions: tuple[str, ...] = ()

    # ---- factor lookups (all already scaled) -------------------------------------------------
    @property
    def k(self) -> float:
        return self.scale * self.systemic

    def rate_bp(self, ccy: str) -> float:
        r = self.spec.get("rates_bp", {})
        return self.k * float(r.get(ccy, r.get("default", 0.0)))

    def fx_pct(self, ccy: str) -> float:
        return 0.0 if ccy == "USD" else self.k * float(self.spec.get("fx_pct", {}).get(ccy, 0.0))

    @property
    def oil_pct(self) -> float:
        return self.k * float(self.spec.get("oil_pct", 0.0))

    def _sector_boost(self, sector: str) -> float:
        return float(library()["sector_targeting_multiplier"]) if sector and sector == self.target_sector else 1.0

    def _region_boost(self, region: str) -> float:
        return float(library()["region_targeting_multiplier"]) if region in self.target_regions else 1.0

    def _named(self, ticker: str, sector: str) -> float:
        """Overlay weight: 1 for the named company, `sector_share` for its sector peers."""
        if not self.overlay:
            return 0.0
        if ticker and ticker == self.target_ticker:
            return 1.0
        return float(self.overlay.get("sector_share", 0.0)) if sector and sector == self.target_sector else 0.0

    def spread_bp(self, rating: str, sector: str, region: str, ticker: str = "", country: str = "",
                  sovereign: bool = False) -> float:
        if sovereign:
            sov = self.spec.get("sovereign_spreads_bp", {})
            base = float(sov.get("EM" if country in library()["em_countries"] else "DM", 0.0))
        else:
            base = float(self.spec.get("spreads_bp", {}).get(rating, 0.0))
            base *= float(self.spec.get("sector_spread_multiplier", {}).get(sector, 1.0))
        systemic = self.k * base * self._sector_boost(sector) * self._region_boost(region)
        return systemic + self._named(ticker, sector) * float(self.overlay.get("spread_bp", 0.0))

    def equity_pct(self, sector: str, ticker: str = "") -> float:
        eq = self.spec.get("equity_pct", {})
        systemic = self.k * float(eq.get(sector, eq.get("default", 0.0))) * self._sector_boost(sector)
        return systemic + self._named(ticker, sector) * float(self.overlay.get("equity_pct", 0.0))

    @property
    def index_equity_pct(self) -> float:
        return self.k * float(self.spec.get("equity_pct", {}).get("default", 0.0))

    def pd_multiplier(self, sector: str, region: str, ticker: str = "") -> float:
        pdm = self.spec.get("pd_multiplier", {})
        m = float(pdm.get(sector, pdm.get("default", 1.0)))
        m *= float(self.spec.get("region_pd_multiplier", {}).get(region, 1.0))
        m *= self._sector_boost(sector) * self._region_boost(region) if m > 1 else 1.0
        m = m ** self.k                                            # geometric scaling of a multiplier
        named = self._named(ticker, sector)
        if named:
            m *= float(self.overlay.get("pd_multiplier", 1.0)) ** named
        return m

    def describe(self) -> dict:
        """Headline factor moves for display."""
        return {"scenario": self.name, "analogue": self.analogue, "scale": round(self.scale, 2),
                "systemic_share": self.systemic, "usd_rates_bp": round(self.rate_bp("USD"), 1),
                "bbb_spread_bp": round(self.spread_bp("BBB", "", ""), 1),
                "equity_index_pct": round(self.index_equity_pct * 100, 2),
                "eur_fx_pct": round(self.fx_pct("EUR") * 100, 2), "oil_pct": round(self.oil_pct * 100, 2),
                "target_ticker": self.target_ticker, "target_sector": self.target_sector,
                "target_regions": list(self.target_regions), "overlay": self.overlay}


def route(event_type: str, headline: str) -> dict:
    rules = library()["routing"].get(event_type)
    if not rules:
        raise ValueError(f"no stress scenario configured for event type {event_type!r}")
    text = headline.lower()
    for rule in rules:
        cues = rule.get("when")
        if not cues or any(re.search(rf"\b{re.escape(c)}\b", text) for c in cues):
            return rule
    return rules[-1]


def build(event_type: str, impact: float, headline: str = "", entity_type: str = "market", ticker: str = "",
          sector: str = "", regions: list[str] | tuple[str, ...] = ()) -> ShockSet:
    rule = route(event_type, headline)
    spec = library()["scenarios"][rule["scenario"]]
    scale = impact_scale(impact)
    overlay, systemic = {}, 1.0
    if rule.get("overlay") and entity_type == "company":
        overlay = dict(library()["overlays"][rule["overlay"]])
        systemic = float(rule.get("systemic_share_if_company", 1.0))
        for k in ("spread_bp", "equity_pct"):
            overlay[k] = overlay[k] * scale
    return ShockSet(key=rule["scenario"], name=spec["name"], analogue=spec.get("analogue", ""), scale=scale,
                    spec=spec, systemic=systemic, overlay=overlay, target_ticker=ticker or "",
                    target_sector=sector or "", target_regions=tuple(regions))


def naive() -> ShockSet:
    spec = library()["naive"]
    return ShockSet(key="naive", name=spec["name"], analogue="event-agnostic", scale=1.0, spec=spec)
