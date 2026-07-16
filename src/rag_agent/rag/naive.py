"""Naive RAG: embed → retrieve top-k → stuff context → generate.

This is the benchmark baseline.  It makes exactly one retrieval call and one
generation call with no re-ranking, planning, or self-critique.  Every other
mode (reranked, agentic) must beat this on RAGAS metrics to justify its cost.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from rag_agent.api.schemas import AskResponse, Source
from rag_agent.db.client import DBClient
from rag_agent.ingestion.embedder import Embedder
from rag_agent.llm import get_llm
from rag_agent.rag import prompt_loader
from rag_agent.settings import Settings

log = structlog.get_logger(__name__)


def ask(
    question: str,
    settings: Settings,
    db: DBClient,
    embedder: Embedder,
    top_k: int | None = None,
) -> AskResponse:
    """Answer a question using naive RAG.

    Args:
        question: The user's question.
        settings: Application settings.
        db: Initialised DBClient.
        embedder: Initialised Embedder.
        top_k: Override settings.top_k for this request.

    Returns:
        AskResponse with answer, cited sources, latency, and token counts.

    Raises:
        ValueError: If the vector store is empty (nothing ingested yet).
    """
    k = top_k or settings.top_k
    t0 = time.perf_counter()

    # ── 1. Embed query ────────────────────────────────────────────────────────
    query_vec = embedder.embed_one(question)

    # ── 2. Retrieve top-k chunks ──────────────────────────────────────────────
    hits = db.similarity_search(query_vec, k)
    if not hits:
        raise ValueError(
            "Vector store is empty. Run the ingestion pipeline first: "
            "python -m rag_agent.ingestion.pipeline --source <docs_dir>"
        )

    log.info("naive_rag.retrieved", n=len(hits), top_sim=hits[0]["similarity"])

    # ── 3. Format numbered context ────────────────────────────────────────────
    context_blocks = [
        f"[{i}] Source: {h['source']}\n{h['content']}"
        for i, h in enumerate(hits, 1)
    ]
    context = "\n\n".join(context_blocks)

    # ── 4. Load prompt + generate ─────────────────────────────────────────────
    system, user, prompt_version = prompt_loader.format_user(
        "naive_rag", context=context, question=question
    )
    log.info("naive_rag.generating", prompt_version=prompt_version, provider=settings.llm_provider)

    llm = get_llm(settings, max_tokens=1024)
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])

    answer = response.content
    tokens_used = (response.usage_metadata or {}).get("total_tokens")
    latency_ms = (time.perf_counter() - t0) * 1000

    log.info(
        "naive_rag.complete",
        latency_ms=round(latency_ms, 1),
        tokens=tokens_used,
        prompt_version=prompt_version,
    )

    sources = [
        Source(
            ref=i,
            content=h["content"],
            source=h["source"],
            title=h.get("title"),
            similarity=round(h["similarity"], 4),
        )
        for i, h in enumerate(hits, 1)
    ]

    return AskResponse(
        answer=answer,
        sources=sources,
        mode="naive",
        latency_ms=round(latency_ms, 1),
        tokens_used=tokens_used,
        prompt_version=prompt_version,
    )
