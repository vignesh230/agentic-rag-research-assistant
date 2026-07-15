"""POST /ask route — routes to the configured RAG mode."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException

from rag_agent.api.deps import get_db, get_embedder, get_settings
from rag_agent.api.schemas import AskRequest, AskResponse
from rag_agent.db.client import DBClient
from rag_agent.ingestion.embedder import Embedder
from rag_agent.rag import agentic, naive
from rag_agent.settings import Settings

log = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    settings: Settings = Depends(get_settings),
    db: DBClient = Depends(get_db),
    embedder: Embedder = Depends(get_embedder),
) -> AskResponse:
    """Answer a question over the ingested document corpus.

    The RAG mode (naive | reranked | agentic) is taken from the request body
    if provided, otherwise falls back to settings.rag_mode.  This makes it
    trivial to compare modes by changing one field in the request.

    Args:
        body: AskRequest with question and optional mode/top_k overrides.
        settings: App-level settings (injected).
        db: Vector store client (injected).
        embedder: Query embedder (injected).

    Returns:
        AskResponse with answer, citations, latency, and token usage.
    """
    mode = body.mode or settings.rag_mode
    log.info("ask.received", question=body.question[:80], mode=mode)

    try:
        if mode == "naive":
            return naive.ask(body.question, settings, db, embedder, body.top_k)
        elif mode == "reranked":
            # Implemented in Phase 4.
            raise HTTPException(
                status_code=501, detail="reranked mode not yet implemented"
            )
        elif mode == "agentic":
            return agentic.ask(body.question, settings, db, embedder, body.top_k)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {mode!r}")

    except ValueError as exc:
        # Empty vector store or other domain errors.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("ask.error", error=str(exc))
        raise HTTPException(status_code=500, detail="Internal server error") from exc
