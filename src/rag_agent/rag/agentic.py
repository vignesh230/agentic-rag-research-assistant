"""Agentic RAG mode: LangGraph planner → retrieve → synthesizer → critic loop.

This is the "agentic" config exposed via /ask?mode=agentic.
It must beat naive RAG on RAGAS metrics to justify its extra LLM calls.
"""

from __future__ import annotations

import time

import structlog

from rag_agent.api.schemas import AskResponse, Source
from rag_agent.db.client import DBClient
from rag_agent.graph.graph import build_graph
from rag_agent.graph.state import AgentState
from rag_agent.ingestion.embedder import Embedder
from rag_agent.settings import Settings

log = structlog.get_logger(__name__)


def ask(
    question: str,
    settings: Settings,
    db: DBClient,
    embedder: Embedder,
    top_k: int | None = None,
) -> AskResponse:
    """Answer a question using the agentic LangGraph pipeline.

    Args:
        question: The user's question.
        settings: Application settings.
        db: Initialised DBClient.
        embedder: Initialised Embedder.
        top_k: Override settings.top_k for this request.

    Returns:
        AskResponse with answer, cited sources, latency, and token counts.

    Raises:
        ValueError: If no answer was produced (empty vector store or graph error).
    """
    k = top_k or settings.top_k
    t0 = time.perf_counter()

    graph = build_graph(settings, db, embedder, k)

    initial: AgentState = {
        "question": question,
        "sub_questions": [],
        "retrieved_chunks": [],
        "draft_answer": None,
        "critic_verdict": None,
        "critic_loops": 0,
        "sources": [],
        "final_answer": None,
    }

    log.info("agentic.ask.start", question=question[:80])
    result: AgentState = graph.invoke(initial)

    final_answer = result.get("final_answer")
    if not final_answer:
        raise ValueError(
            "Agentic graph produced no answer. "
            "Ensure the vector store has been populated via the ingestion pipeline."
        )

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    critic_loops = result.get("critic_loops", 0)
    log.info("agentic.ask.done", latency_ms=latency_ms, critic_loops=critic_loops)

    sources = [
        Source(
            ref=s["ref"],
            content=s["content"],
            source=s["source"],
            title=s.get("title"),
            similarity=s["similarity"],
        )
        for s in result.get("sources", [])
    ]

    return AskResponse(
        answer=final_answer,
        sources=sources,
        mode="agentic",
        latency_ms=latency_ms,
        tokens_used=None,  # aggregated across multiple calls; Phase 6 adds per-node tracing
        prompt_version=None,
    )
