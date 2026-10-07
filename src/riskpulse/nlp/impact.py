"""Impact Score (1-10): predicted severity of a signal's potential market impact.

A transparent logistic factor model (config/impact.yaml) instead of a black box, so every score
can be explained to a risk manager:  impact = 1 + 9 * sigmoid(b + sum_i w_i * f_i).
Each signal carries its features and per-feature contributions (w_i * f_i) as `impact_drivers`.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

from riskpulse.config import MODELS, load_yaml
from riskpulse.nlp import lexicons as lx
from riskpulse.schema import Entity, ImpactDrivers
from riskpulse.universe import companies

FEATURES = ("severity", "sentiment", "intensity", "systemic", "confirmation", "velocity", "attention",
            "credibility", "novelty")


@lru_cache(maxsize=1)
def severity_priors() -> dict[str, float]:
    return {k: float(v["severity"]) for k, v in load_yaml("taxonomy.yaml")["events"].items()}


@lru_cache(maxsize=1)
def model_params() -> tuple[float, dict[str, float], str]:
    cfg = load_yaml("impact.yaml")
    calibrated = MODELS / "impact_calibrated.json"
    if cfg.get("use_calibrated") and calibrated.exists():
        fit = json.loads(calibrated.read_text(encoding="utf-8"))
        return float(fit["intercept"]), {k: float(v) for k, v in fit["weights"].items()}, "calibrated"
    return float(cfg["expert"]["intercept"]), {k: float(v) for k, v in cfg["expert"]["weights"].items()}, "expert"


@lru_cache(maxsize=1)
def _cap_share() -> dict[str, float]:
    total = sum(c.market_cap_bn for c in companies())
    return {c.ticker: c.market_cap_bn / total for c in companies()}


def intensity(text: str) -> tuple[float, list[str]]:
    cues = {m: lx.SEVERITY_CUES.get(m, 0.0) for m in lx.hits(lx.SEVERITY_RX, text)}
    for m in lx.PCT_RX.finditer(text):
        pct = float(m.group(1))
        if 3 <= pct <= 100:
            cues[m.group(0)] = round(min(1.0, 0.35 + pct / 25), 2)
    for m in lx.BP_RX.finditer(text):  # size of a rate / spread move
        cues[m.group(0)] = round(min(1.0, 0.35 + float(m.group(1)) / 200), 2)
    for m in lx.MONEY_RX.finditer(text):
        amount_bn = float(m.group(1)) * (1000 if m.group(2).lower().startswith("t") else 1)
        cues[m.group(0)] = round(min(1.0, 0.3 + math.log10(max(amount_bn, 1)) / 3.5), 2)
    top = sorted(cues, key=cues.get, reverse=True)[:4]
    return (max(cues.values()) if cues else 0.0), top


def systemic(entity: Entity) -> float:
    if entity.type == "market":
        return 1.0
    if entity.type == "sector":
        return 0.8
    return min(1.0, 0.45 + 0.55 * math.sqrt(_cap_share().get(entity.id, 0.01) / 0.15))


def features(*, event_type: str, event_conf: float, sentiment: float, sentiment_conf: float, text: str,
             entity: Entity, source_type: str, publisher: str, n_sources: int = 1, mentions: int = 1,
             novel: bool = True, entity_signals_24h: int = 1) -> tuple[dict[str, float], list[str]]:
    pos_factor = float(load_yaml("impact.yaml").get("positive_sentiment_factor", 0.75))
    inten, cues = intensity(text)
    f = {
        "severity": event_conf * severity_priors().get(event_type, 0.3) + (1 - event_conf) * 0.3,
        "sentiment": abs(sentiment) * sentiment_conf * (1.0 if sentiment < 0 else pos_factor),
        "intensity": inten,
        "systemic": systemic(entity),
        "confirmation": 1.0 - math.exp(-(max(n_sources, 1) - 1) / 3.0),
        "velocity": 1.0 - math.exp(-(max(mentions, 1) - 1) / 5.0),
        "attention": attention(entity_signals_24h),
        "credibility": lx.credibility(source_type, publisher),
        "novelty": 1.0 if novel else 0.0,
    }
    return {k: round(v, 4) for k, v in f.items()}, cues


def attention(entity_signals_24h: int) -> float:
    """Abnormal news volume on the entity: signals in the trailing 24h (this one included)."""
    return 1.0 - math.exp(-(max(entity_signals_24h, 1) - 1) / 4.0)


def score_features(f: dict[str, float]) -> tuple[float, dict[str, float], float]:
    b, w, _ = model_params()
    contrib = {k: round(w.get(k, 0.0) * f.get(k, 0.0), 4) for k in FEATURES}
    logit = b + sum(contrib.values())
    return round(1.0 + 9.0 / (1.0 + math.exp(-logit)), 1), contrib, round(logit, 4)


def score(**kwargs) -> tuple[float, ImpactDrivers]:
    f, cues = features(**kwargs)
    impact, contrib, logit = score_features(f)
    return impact, ImpactDrivers(features=f, contributions=contrib, logit=logit, cues=cues)


def weights_source() -> str:
    return model_params()[2]
