"""FastAPI application factory.

Run with:
    uvicorn rag_agent.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI

from rag_agent.api.routes.ask import router as ask_router
from rag_agent.api.schemas import HealthResponse
from rag_agent.db.client import DBClient
from rag_agent.ingestion.embedder import Embedder
from rag_agent.logging_config import configure_logging
from rag_agent.settings import Settings

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise shared resources once at startup; release at shutdown."""
    settings = Settings()
    configure_logging(settings.log_level)

    settings.validate_api_keys()
    log.info("startup.begin", mode=settings.rag_mode, model=settings.claude_model)

    db = DBClient(settings)
    db.create_tables()

    # Embedder loads the model weights on first call; doing it here warms the
    # cache so the first /ask request doesn't pay the ~1-2s model-load penalty.
    embedder = Embedder(settings)

    app.state.settings = settings
    app.state.db = db
    app.state.embedder = embedder

    log.info("startup.complete", embedding_model=settings.embedding_model)
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Agentic RAG Research Assistant",
        description=(
            "Answer questions over a document corpus using configurable RAG modes: "
            "naive (baseline), reranked, agentic (LangGraph)."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(ask_router, tags=["RAG"])

    @app.get("/health", response_model=HealthResponse, tags=["Ops"])
    def health() -> HealthResponse:
        """Liveness check — returns current mode and embedding model."""
        settings: Settings = app.state.settings
        return HealthResponse(
            status="ok",
            mode=settings.rag_mode,
            embedding_model=settings.embedding_model,
        )

    return app


app = create_app()
