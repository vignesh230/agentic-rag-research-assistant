"""Integration-style tests for the full agentic graph and /ask?mode=agentic."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag_agent.api.schemas import AskResponse
from rag_agent.settings import Settings


def _mock_client(planner_text, synth_text, critic_text):
    """Return a fake Anthropic client that cycles through three canned responses."""
    client = MagicMock()
    responses = [planner_text, synth_text, critic_text]
    call_count = [0]

    def _create(**kwargs):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        m = MagicMock()
        m.content = [MagicMock(text=responses[idx])]
        m.usage = MagicMock(input_tokens=10, output_tokens=20)
        return m

    client.messages.create.side_effect = _create
    return client


# ── agentic.ask() unit test ───────────────────────────────────────────────────

def test_agentic_ask_happy_path():
    settings = Settings(anthropic_api_key="test", max_critic_loops=3, top_k=2)

    db = MagicMock()
    db.similarity_search.return_value = [
        {"content": "Transformers use self-attention.", "source": "paper.pdf",
         "title": None, "similarity": 0.9}
    ]
    embedder = MagicMock()
    embedder.embed_one.return_value = np.zeros(384, dtype=np.float32)

    fake_client = _mock_client(
        '["What is self-attention?"]',
        "Self-attention allows tokens to attend to each other [1].",
        "supported",
    )

    with patch("rag_agent.graph.nodes.anthropic.Anthropic", return_value=fake_client), \
         patch("rag_agent.graph.graph.anthropic.Anthropic", return_value=fake_client):
        from rag_agent.rag import agentic
        result = agentic.ask("What is attention?", settings, db, embedder)

    assert isinstance(result, AskResponse)
    assert result.mode == "agentic"
    assert len(result.sources) >= 1


def test_agentic_ask_raises_on_empty_store():
    settings = Settings(anthropic_api_key="test", max_critic_loops=1, top_k=2)
    db = MagicMock()
    db.similarity_search.return_value = []
    embedder = MagicMock()
    embedder.embed_one.return_value = np.zeros(384, dtype=np.float32)

    fake_client = _mock_client('["What is attention?"]', "No relevant context was retrieved.", "supported")

    with patch("rag_agent.graph.nodes.anthropic.Anthropic", return_value=fake_client), \
         patch("rag_agent.graph.graph.anthropic.Anthropic", return_value=fake_client):
        from rag_agent.rag import agentic
        # synthesizer returns "No relevant context" → graph still sets final_answer
        result = agentic.ask("Q?", settings, db, embedder)

    assert result.mode == "agentic"


# ── /ask endpoint with mode=agentic ──────────────────────────────────────────

@pytest.fixture()
def client():
    from rag_agent.api.main import create_app

    app = create_app()

    @asynccontextmanager
    async def _null_lifespan(a):
        settings = Settings(anthropic_api_key="test", embedding_dim=384)
        a.state.settings = settings
        a.state.db = MagicMock()
        a.state.embedder = MagicMock()
        a.state.embedder.embed_one.return_value = np.zeros(384, dtype=np.float32)
        yield

    app.router.lifespan_context = _null_lifespan

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_ask_agentic_endpoint(client):
    fake_resp = AskResponse(
        answer="Attention is a mechanism.",
        sources=[],
        mode="agentic",
        latency_ms=100.0,
        tokens_used=None,
        prompt_version=None,
    )
    with patch("rag_agent.api.routes.ask.agentic.ask", return_value=fake_resp):
        resp = client.post("/ask", json={"question": "What is attention?", "mode": "agentic"})

    assert resp.status_code == 200
    assert resp.json()["mode"] == "agentic"
