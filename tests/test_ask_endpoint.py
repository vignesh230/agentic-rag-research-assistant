"""Unit tests for the POST /ask endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag_agent.api.schemas import AskResponse, Source
from rag_agent.settings import Settings


def _make_fake_response() -> AskResponse:
    return AskResponse(
        answer="Attention is a mechanism.",
        sources=[
            Source(ref=1, content="ctx", source="doc.pdf", title=None, similarity=0.9)
        ],
        mode="naive",
        latency_ms=42.0,
        tokens_used=30,
        prompt_version="1.0",
    )


@pytest.fixture()
def client():
    """TestClient with a null lifespan that injects fake state (no Postgres needed)."""
    from rag_agent.api.main import create_app

    app = create_app()

    # Replace real lifespan so tests skip Postgres/model loading.
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


def test_ask_naive_ok(client):
    with patch("rag_agent.api.routes.ask.naive.ask", return_value=_make_fake_response()):
        resp = client.post("/ask", json={"question": "What is attention?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "naive"
    assert data["answer"] == "Attention is a mechanism."
    assert len(data["sources"]) == 1


def test_ask_question_too_short(client):
    resp = client.post("/ask", json={"question": "hi"})
    assert resp.status_code == 422




def test_ask_empty_store_returns_503(client):
    with patch("rag_agent.api.routes.ask.naive.ask", side_effect=ValueError("Vector store is empty")):
        resp = client.post("/ask", json={"question": "What is attention?"})

    assert resp.status_code == 503
    assert "empty" in resp.json()["detail"].lower()


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
