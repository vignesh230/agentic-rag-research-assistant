"""Tests for the Embedder.

The actual model download (~90 MB) is gated behind the 'integration' mark
so the fast unit-test suite never hits the network.  Run integration tests
with: pytest -m integration
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_agent.ingestion.embedder import Embedder
from rag_agent.settings import Settings


@pytest.fixture()
def settings_384() -> Settings:
    return Settings(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
    )


# ── Unit tests (no network, model is mocked) ──────────────────────────────────


class TestEmbedderUnit:
    def _mock_embedder(self, settings: Settings, dim: int = 384) -> Embedder:
        """Return an Embedder whose underlying SentenceTransformer is mocked."""
        emb = Embedder(settings)
        fake_model = MagicMock()
        fake_model.encode.side_effect = lambda texts, **kwargs: np.random.rand(
            len(texts), dim
        ).astype(np.float32)
        with patch(
            "rag_agent.ingestion.embedder._MODEL_CACHE",
            {settings.embedding_model: fake_model},
        ):
            return emb

    def test_embed_returns_correct_shape(self, settings_384: Settings) -> None:
        emb = Embedder(settings_384)
        fake = MagicMock()
        fake.encode.return_value = np.zeros((3, 384), dtype=np.float32)
        with patch.dict(
            "rag_agent.ingestion.embedder._MODEL_CACHE",
            {settings_384.embedding_model: fake},
        ):
            result = emb.embed(["a", "b", "c"])
        assert result.shape == (3, 384)

    def test_embed_one_returns_1d(self, settings_384: Settings) -> None:
        emb = Embedder(settings_384)
        fake = MagicMock()
        fake.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        with patch.dict(
            "rag_agent.ingestion.embedder._MODEL_CACHE",
            {settings_384.embedding_model: fake},
        ):
            result = emb.embed_one("hello")
        assert result.ndim == 1
        assert result.shape == (384,)

    def test_embed_empty_raises(self, settings_384: Settings) -> None:
        emb = Embedder(settings_384)
        with pytest.raises(ValueError, match="non-empty"):
            emb.embed([])

    def test_dim_mismatch_raises(self, settings_384: Settings) -> None:
        """Model returns wrong dimension — should raise, not silently corrupt."""
        emb = Embedder(settings_384)
        fake = MagicMock()
        fake.encode.return_value = np.zeros((1, 768), dtype=np.float32)  # wrong dim
        with patch.dict(
            "rag_agent.ingestion.embedder._MODEL_CACHE",
            {settings_384.embedding_model: fake},
        ):
            with pytest.raises(ValueError, match="dim=768"):
                emb.embed(["hello"])

    def test_output_dtype_is_float32(self, settings_384: Settings) -> None:
        emb = Embedder(settings_384)
        fake = MagicMock()
        # Return float64 — embedder must cast to float32 for pgvector
        fake.encode.return_value = np.ones((2, 384), dtype=np.float64)
        with patch.dict(
            "rag_agent.ingestion.embedder._MODEL_CACHE",
            {settings_384.embedding_model: fake},
        ):
            result = emb.embed(["x", "y"])
        assert result.dtype == np.float32


# ── Integration tests (downloads real model, skipped in CI by default) ────────


@pytest.mark.integration
class TestEmbedderIntegration:
    def test_real_model_shape(self, settings_384: Settings) -> None:
        emb = Embedder(settings_384)
        result = emb.embed(["Graph neural networks learn on relational data."])
        assert result.shape == (1, 384)

    def test_real_model_normalised(self, settings_384: Settings) -> None:
        """Embeddings should have L2 norm ≈ 1 after normalisation."""
        emb = Embedder(settings_384)
        vecs = emb.embed(["hello world", "foo bar baz"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_determinism(self, settings_384: Settings) -> None:
        """Same input → same output (no dropout at inference)."""
        emb = Embedder(settings_384)
        v1 = emb.embed(["determinism check"])
        v2 = emb.embed(["determinism check"])
        np.testing.assert_array_equal(v1, v2)
