"""Unit tests for individual graph nodes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_agent.graph.nodes import make_critic, make_planner, make_retrieve, make_synthesizer
from rag_agent.graph.state import AgentState
from rag_agent.settings import Settings


@pytest.fixture()
def settings():
    return Settings(anthropic_api_key="test", max_critic_loops=3, top_k=3)


@pytest.fixture()
def mock_client():
    return MagicMock()


def _msg(text: str, in_tok: int = 10, out_tok: int = 20):
    m = MagicMock()
    m.content = text
    m.usage_metadata = {"total_tokens": in_tok + out_tok}
    return m


def _base_state(**kwargs) -> AgentState:
    defaults: AgentState = {
        "question": "What is attention?",
        "sub_questions": ["What is attention?"],
        "retrieved_chunks": [],
        "draft_answer": None,
        "critic_verdict": None,
        "critic_loops": 0,
        "sources": [],
        "final_answer": None,
    }
    defaults.update(kwargs)
    return defaults


# ── planner ─────────────────────────────────────────────────────────────────

def test_planner_parses_json(settings, mock_client):
    mock_client.invoke.return_value = _msg('["What is A?", "How does B work?"]')
    fn = make_planner(settings, mock_client)
    result = fn(_base_state())
    assert result["sub_questions"] == ["What is A?", "How does B work?"]


def test_planner_falls_back_on_bad_json(settings, mock_client):
    mock_client.invoke.return_value = _msg("Sorry, I can't decompose that.")
    fn = make_planner(settings, mock_client)
    result = fn(_base_state())
    assert result["sub_questions"] == ["What is attention?"]


# ── retrieve ─────────────────────────────────────────────────────────────────

def test_retrieve_appends_new_chunks(settings):
    db = MagicMock()
    db.similarity_search.return_value = [
        {"content": "Attention uses Q, K, V matrices.", "source": "paper.pdf",
         "title": None, "similarity": 0.9},
    ]
    embedder = MagicMock()
    embedder.embed_one.return_value = np.zeros(384, dtype=np.float32)

    fn = make_retrieve(settings, db, embedder, top_k=3)
    result = fn(_base_state())
    assert len(result["retrieved_chunks"]) == 1


def test_retrieve_deduplicates(settings):
    chunk = {"content": "Attention uses Q, K, V matrices.", "source": "paper.pdf",
             "title": None, "similarity": 0.9}
    db = MagicMock()
    db.similarity_search.return_value = [chunk]
    embedder = MagicMock()
    embedder.embed_one.return_value = np.zeros(384, dtype=np.float32)

    fn = make_retrieve(settings, db, embedder, top_k=3)
    # Pre-load the same chunk in existing state.
    state = _base_state(retrieved_chunks=[chunk])
    result = fn(state)
    # Dedup: still only 1 chunk, not 2.
    assert len(result["retrieved_chunks"]) == 1


# ── synthesizer ───────────────────────────────────────────────────────────────

def test_synthesizer_produces_draft(settings, mock_client):
    mock_client.invoke.return_value = _msg("Attention is [1] a mechanism.")
    fn = make_synthesizer(settings, mock_client)
    chunks = [{"content": "ctx", "source": "doc.pdf", "title": None, "similarity": 0.9}]
    result = fn(_base_state(retrieved_chunks=chunks))
    assert result["draft_answer"] == "Attention is [1] a mechanism."
    assert len(result["sources"]) == 1


def test_synthesizer_no_chunks(settings, mock_client):
    fn = make_synthesizer(settings, mock_client)
    result = fn(_base_state(retrieved_chunks=[]))
    assert "No relevant context" in result["draft_answer"]
    mock_client.invoke.assert_not_called()


# ── critic ────────────────────────────────────────────────────────────────────

def test_critic_supported(settings, mock_client):
    mock_client.invoke.return_value = _msg("supported")
    fn = make_critic(settings, mock_client)
    chunks = [{"content": "ctx", "source": "s", "title": None, "similarity": 0.9}]
    result = fn(_base_state(draft_answer="The answer.", retrieved_chunks=chunks))
    assert result["critic_verdict"] == "supported"
    assert result["final_answer"] == "The answer."


def test_critic_rewrite_sets_sub_questions(settings, mock_client):
    mock_client.invoke.return_value = _msg("rewrite: how does attention scale")
    fn = make_critic(settings, mock_client)
    chunks = [{"content": "ctx", "source": "s", "title": None, "similarity": 0.9}]
    result = fn(_base_state(draft_answer="Draft.", retrieved_chunks=chunks, critic_loops=0))
    assert result["sub_questions"] == ["how does attention scale"]
    assert "final_answer" not in result or result.get("final_answer") is None


def test_critic_caps_at_max_loops(settings, mock_client):
    mock_client.invoke.return_value = _msg("rewrite: something else")
    fn = make_critic(settings, mock_client)
    chunks = [{"content": "ctx", "source": "s", "title": None, "similarity": 0.9}]
    # critic_loops already at max - 1; after increment it hits max.
    result = fn(_base_state(
        draft_answer="Draft.", retrieved_chunks=chunks, critic_loops=2
    ))
    # Loop cap reached → finalise answer despite rewrite verdict.
    assert result["final_answer"] == "Draft."
