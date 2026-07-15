"""Unit tests for the naive RAG pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_agent.api.schemas import AskResponse
from rag_agent.rag import naive
from rag_agent.settings import Settings


@pytest.fixture()
def settings():
    return Settings(
        anthropic_api_key="test-key",
        embedding_dim=384,
        top_k=3,
        rag_mode="naive",
    )


@pytest.fixture()
def mock_db():
    db = MagicMock()
    db.similarity_search.return_value = [
        {
            "content": "Transformers use self-attention mechanisms.",
            "source": "paper.pdf",
            "title": "Attention is All You Need",
            "similarity": 0.92,
        },
        {
            "content": "BERT is a pre-trained transformer model.",
            "source": "paper2.pdf",
            "title": "BERT Paper",
            "similarity": 0.85,
        },
    ]
    return db


@pytest.fixture()
def mock_embedder():
    emb = MagicMock()
    emb.embed_one.return_value = np.zeros(384, dtype=np.float32)
    return emb


def _make_anthropic_response(text: str, input_tokens: int = 10, output_tokens: int = 20):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


def test_ask_returns_response(settings, mock_db, mock_embedder):
    with patch("rag_agent.rag.naive.anthropic.Anthropic") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.messages.create.return_value = _make_anthropic_response("Transformers use attention [1].")

        result = naive.ask("What are transformers?", settings, mock_db, mock_embedder)

    assert isinstance(result, AskResponse)
    assert result.answer == "Transformers use attention [1]."
    assert result.mode == "naive"
    assert len(result.sources) == 2
    assert result.tokens_used == 30
    assert result.prompt_version is not None


def test_ask_uses_top_k_override(settings, mock_db, mock_embedder):
    with patch("rag_agent.rag.naive.anthropic.Anthropic") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.messages.create.return_value = _make_anthropic_response("Answer.")

        naive.ask("Q?", settings, mock_db, mock_embedder, top_k=7)

    mock_db.similarity_search.assert_called_once()
    _, call_k = mock_db.similarity_search.call_args[0]
    assert call_k == 7


def test_ask_raises_on_empty_store(settings, mock_embedder):
    db = MagicMock()
    db.similarity_search.return_value = []

    with pytest.raises(ValueError, match="empty"):
        naive.ask("Q?", settings, db, mock_embedder)


def test_sources_have_correct_refs(settings, mock_db, mock_embedder):
    with patch("rag_agent.rag.naive.anthropic.Anthropic") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.messages.create.return_value = _make_anthropic_response("A")

        result = naive.ask("Q?", settings, mock_db, mock_embedder)

    refs = [s.ref for s in result.sources]
    assert refs == [1, 2]
    assert result.sources[0].source == "paper.pdf"
