"""NLP tier resolution.

full : FinBERT sentiment + bge-small sentence embeddings (torch, transformers, sentence-transformers)
lite : VADER + finance lexicon sentiment, hashed TF-IDF vectors (scikit-learn only, no torch)

`auto` picks full when the deep-learning stack imports, otherwise lite. Model loading failures at
engine start-up (e.g. offline with an empty HuggingFace cache) also fall back to lite.
"""
from __future__ import annotations

import importlib.util
import logging

from riskpulse.config import requested_tier

log = logging.getLogger("riskpulse.tier")


def full_stack_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "sentence_transformers"))


def resolve_tier(requested: str | None = None) -> str:
    req = (requested or requested_tier()).lower()
    if req == "lite":
        return "lite"
    if full_stack_available():
        return "full"
    if req == "full":
        log.warning("full tier requested but torch/transformers are not installed - using lite")
    return "lite"
