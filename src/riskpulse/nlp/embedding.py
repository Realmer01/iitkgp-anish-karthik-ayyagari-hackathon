"""Text vectorisers shared by story clustering, event classification and the social sentiment head.

One encoding per document is reused by all three stages - the engine never embeds a text twice.
"""
from __future__ import annotations

import numpy as np

from riskpulse import config


class FullEmbedder:
    """bge-small-en-v1.5 sentence embeddings (384-d, L2-normalised)."""

    tier = "full"

    def __init__(self, model_id: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id or config.get("engine.embedding_model", "BAAI/bge-small-en-v1.5")
        self.model = SentenceTransformer(self.model_id, device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


class LiteEmbedder:
    """Hashed word uni/bi-gram TF vectors (2048-d, L2-normalised) - no fitting, no torch."""

    tier = "lite"

    def __init__(self, n_features: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = n_features
        self.vec = HashingVectorizer(n_features=n_features, ngram_range=(1, 2), alternate_sign=False,
                                     norm="l2", stop_words="english")

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.vec.transform(texts).toarray().astype(np.float32)


def make_embedder(tier: str):
    return FullEmbedder() if tier == "full" else LiteEmbedder()
