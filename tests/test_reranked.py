"""Unit tests for the reranked RAG mode."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_agent.api.schemas import AskResponse
from rag_agent.rag import reranked
from rag_agent.rag.reranker import rerank
from rag_agent.settings import Settings


@pytest.fixture()
def settings():
    return Settings(
        anthropic_api_key="test",
        top_k=2,
        retrieval_multiplier=3,
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )


def _chunks(n: int) -> list[dict]:
    return [
        {"content": f"Content {i}", "source": f"doc{i}.pdf",
         "title": None, "similarity": round(0.9 - i * 0.05, 2)}
        for i in range(n)
    ]


def _msg(text: str):
    m = MagicMock()
    m.content = text
    m.usage_metadata = {"total_tokens": 30}
    return m


# ── reranker unit tests ───────────────────────────────────────────────────────

def test_rerank_returns_top_k():
    chunks = _chunks(6)
    fake_scores = [0.1, 0.9, 0.3, 0.8, 0.2, 0.7]

    with patch("rag_agent.rag.reranker._get_model") as mock_get:
        mock_model = MagicMock()
        mock_model.predict.return_value = fake_scores
        mock_get.return_value = mock_model

        result = rerank("query", chunks, top_k=2, model_name="test-model")

    assert len(result) == 2
    # Best score (0.9) should be first.
    assert result[0]["content"] == "Content 1"
    assert result[0]["ce_score"] == pytest.approx(0.9)


def test_rerank_empty_input():
    result = rerank("query", [], top_k=3, model_name="test-model")
    assert result == []


def test_rerank_fewer_chunks_than_top_k():
    chunks = _chunks(2)
    with patch("rag_agent.rag.reranker._get_model") as mock_get:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.8]
        mock_get.return_value = mock_model

        result = rerank("query", chunks, top_k=5, model_name="test-model")

    assert len(result) == 2


# ── reranked.ask() tests ──────────────────────────────────────────────────────

def test_reranked_ask_retrieves_wider_then_narrows(settings):
    """Should call similarity_search with top_k * multiplier, return top_k."""
    db = MagicMock()
    db.similarity_search.return_value = _chunks(6)  # top_k*multiplier = 2*3 = 6
    embedder = MagicMock()
    embedder.embed_one.return_value = np.zeros(384, dtype=np.float32)

    fake_scores = [float(i) for i in range(6)]

    with patch("rag_agent.rag.reranker._get_model") as mock_ce, \
         patch("rag_agent.rag.reranked.get_llm") as mock_get_llm:
        mock_ce.return_value.predict.return_value = fake_scores
        mock_get_llm.return_value.invoke.return_value = _msg("Answer.")

        result = reranked.ask("Q?", settings, db, embedder)

    # Called with candidates = top_k * multiplier = 6.
    _, call_k = db.similarity_search.call_args[0]
    assert call_k == 6
    # Returned top_k = 2.
    assert len(result.sources) == 2
    assert result.mode == "reranked"


def test_reranked_ask_raises_on_empty_store(settings):
    db = MagicMock()
    db.similarity_search.return_value = []
    embedder = MagicMock()
    embedder.embed_one.return_value = np.zeros(384, dtype=np.float32)

    with pytest.raises(ValueError, match="empty"):
        reranked.ask("Q?", settings, db, embedder)


def test_reranked_ask_returns_response(settings):
    db = MagicMock()
    db.similarity_search.return_value = _chunks(6)
    embedder = MagicMock()
    embedder.embed_one.return_value = np.zeros(384, dtype=np.float32)

    with patch("rag_agent.rag.reranker._get_model") as mock_ce, \
         patch("rag_agent.rag.reranked.get_llm") as mock_get_llm:
        mock_ce.return_value.predict.return_value = list(range(6))
        mock_get_llm.return_value.invoke.return_value = _msg("Final answer.")

        result = reranked.ask("Q?", settings, db, embedder)

    assert isinstance(result, AskResponse)
    assert result.answer == "Final answer."
    assert result.tokens_used == 30
