"""Sentence-transformer embedder.

Embeddings are L2-normalised before storage so cosine similarity equals
dot product — this lets pgvector's inner product index (<#>) serve as a
faster alternative to cosine (<=>).  We default to cosine for explainability.
"""

from __future__ import annotations

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from rag_agent.settings import Settings

log = structlog.get_logger(__name__)

# Module-level model cache: one model instance per (model_name, device) pair.
# Loading a SentenceTransformer downloads weights on first call (~90 MB for
# all-MiniLM-L6-v2); caching avoids re-loading within a process lifetime.
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        log.info("embedder.loading_model", model=model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class Embedder:
    """Wraps a SentenceTransformer to embed lists of strings."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.embedding_model
        self._expected_dim = settings.embedding_dim

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of strings and return an (N, dim) float32 array.

        Embeddings are L2-normalised so that cosine similarity == dot product.

        Args:
            texts: Non-empty list of strings to embed.
            batch_size: How many texts to process per forward pass.  Reduce
                        this if you hit GPU/CPU memory limits.

        Returns:
            Float32 array of shape (len(texts), embedding_dim).

        Raises:
            ValueError: If texts is empty or the returned dimension does not
                        match settings.embedding_dim.
        """
        if not texts:
            raise ValueError("texts must be non-empty")

        model = _get_model(self._model_name)
        vectors: np.ndarray = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = vectors.astype(np.float32)

        if vectors.shape[1] != self._expected_dim:
            raise ValueError(
                f"Model '{self._model_name}' returned dim={vectors.shape[1]}, "
                f"but settings.embedding_dim={self._expected_dim}. "
                "Update EMBEDDING_DIM in your .env to match."
            )

        log.info(
            "embedder.embedded",
            n=len(texts),
            model=self._model_name,
            dim=vectors.shape[1],
        )
        return vectors

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string and return a 1-D float32 array.

        Args:
            text: The string to embed.

        Returns:
            Float32 array of shape (embedding_dim,).
        """
        return self.embed([text])[0]
