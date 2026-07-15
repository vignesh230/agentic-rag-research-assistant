"""Reranked RAG: retrieve wide → cross-encoder rerank → generate.

Sits between naive (top-k by cosine) and agentic (multi-step graph) in the
quality/cost ladder.  Retrieves top_k * retrieval_multiplier candidates from
the vector store, reranks with a cross-encoder, then generates from the top-k.
Generation prompt is identical to naive so the only variable is chunk quality.
"""

from __future__ import annotations

import time

import anthropic
import structlog

from rag_agent.api.schemas import AskResponse, Source
from rag_agent.db.client import DBClient
from rag_agent.ingestion.embedder import Embedder
from rag_agent.rag import prompt_loader
from rag_agent.rag.reranker import rerank
from rag_agent.settings import Settings

log = structlog.get_logger(__name__)


def ask(
    question: str,
    settings: Settings,
    db: DBClient,
    embedder: Embedder,
    top_k: int | None = None,
) -> AskResponse:
    """Answer a question using reranked RAG.

    Retrieves top_k * retrieval_multiplier candidates by cosine similarity,
    reranks with a cross-encoder, then generates from the top-k survivors.

    Args:
        question: The user's question.
        settings: Application settings.
        db: Initialised DBClient.
        embedder: Initialised Embedder.
        top_k: Override settings.top_k for this request.

    Returns:
        AskResponse with answer, cited sources, latency, and token counts.

    Raises:
        ValueError: If the vector store is empty.
    """
    k = top_k or settings.top_k
    candidates = k * settings.retrieval_multiplier
    t0 = time.perf_counter()

    # ── 1. Embed + wide retrieval ─────────────────────────────────────────────
    query_vec = embedder.embed_one(question)
    hits = db.similarity_search(query_vec, candidates)
    if not hits:
        raise ValueError(
            "Vector store is empty. Run the ingestion pipeline first: "
            "python -m rag_agent.ingestion.pipeline --source <docs_dir>"
        )
    log.info("reranked.retrieved", candidates=len(hits))

    # ── 2. Cross-encoder rerank ───────────────────────────────────────────────
    top_chunks = rerank(question, hits, k, settings.cross_encoder_model)
    log.info("reranked.reranked", top_k=len(top_chunks))

    # ── 3. Format context + generate (same prompt as naive) ───────────────────
    context = "\n\n".join(
        f"[{i}] Source: {c['source']}\n{c['content']}"
        for i, c in enumerate(top_chunks, 1)
    )
    system, user, prompt_version = prompt_loader.format_user(
        "naive_rag", context=context, question=question
    )
    log.info("reranked.generating", prompt_version=prompt_version, model=settings.claude_model)

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    answer = message.content[0].text
    tokens_used = message.usage.input_tokens + message.usage.output_tokens
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    log.info("reranked.complete", latency_ms=latency_ms, tokens=tokens_used)

    sources = [
        Source(
            ref=i,
            content=c["content"],
            source=c["source"],
            title=c.get("title"),
            similarity=round(c["similarity"], 4),
        )
        for i, c in enumerate(top_chunks, 1)
    ]

    return AskResponse(
        answer=answer,
        sources=sources,
        mode="reranked",
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        prompt_version=prompt_version,
    )
