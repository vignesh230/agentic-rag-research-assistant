"""FastAPI dependency providers.

Resources (DBClient, Embedder) are initialised once at startup and stored on
app.state so they are reused across requests rather than reconstructed per-call.
"""

from __future__ import annotations

from fastapi import Request

from rag_agent.db.client import DBClient
from rag_agent.ingestion.embedder import Embedder
from rag_agent.settings import Settings


def get_settings(request: Request) -> Settings:
    """Return the Settings instance attached at startup."""
    return request.app.state.settings


def get_db(request: Request) -> DBClient:
    """Return the DBClient attached at startup."""
    return request.app.state.db


def get_embedder(request: Request) -> Embedder:
    """Return the Embedder attached at startup."""
    return request.app.state.embedder
