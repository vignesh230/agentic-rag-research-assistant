"""LangGraph node factories for the agentic RAG pipeline.

Each public function is a factory that captures dependencies (settings, db,
embedder, anthropic client) and returns a node callable with the signature
expected by LangGraph: (state: AgentState) -> dict.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import anthropic
import structlog

from rag_agent.db.client import DBClient
from rag_agent.graph.state import AgentState
from rag_agent.ingestion.embedder import Embedder
from rag_agent.rag import prompt_loader
from rag_agent.settings import Settings

log = structlog.get_logger(__name__)


def make_planner(settings: Settings, client: anthropic.Anthropic) -> Callable:
    """Decompose the user question into 1-3 focused sub-questions."""

    def planner(state: AgentState) -> dict:
        question = state["question"]
        log.info("agentic.planner.start", question=question[:80])

        system, user, version = prompt_loader.format_user("planner", question=question)
        msg = client.messages.create(
            model=settings.claude_model,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = msg.content[0].text.strip()
        log.debug("agentic.planner.raw", raw=raw, prompt_version=version)

        try:
            sub_questions = json.loads(raw)
            if not isinstance(sub_questions, list) or not sub_questions:
                raise ValueError("not a non-empty list")
        except (json.JSONDecodeError, ValueError):
            # Fallback: treat the original question as the only sub-question.
            log.warning("agentic.planner.parse_failed", raw=raw)
            sub_questions = [question]

        log.info("agentic.planner.done", n=len(sub_questions))
        return {"sub_questions": sub_questions}

    return planner


def make_retrieve(
    settings: Settings, db: DBClient, embedder: Embedder, top_k: int
) -> Callable:
    """Embed each sub-question and retrieve top-k chunks, deduped by content."""

    def retrieve(state: AgentState) -> dict:
        sub_questions = state["sub_questions"]
        existing = state.get("retrieved_chunks", [])
        seen: set[str] = {_chunk_key(c) for c in existing}

        log.info("agentic.retrieve.start", queries=sub_questions, existing=len(existing))
        new_chunks: list[dict] = []

        for q in sub_questions:
            vec = embedder.embed_one(q)
            hits = db.similarity_search(vec, top_k)
            for h in hits:
                key = _chunk_key(h)
                if key not in seen:
                    seen.add(key)
                    new_chunks.append(h)

        log.info("agentic.retrieve.done", new=len(new_chunks), total=len(existing) + len(new_chunks))
        return {"retrieved_chunks": existing + new_chunks}

    return retrieve


def make_synthesizer(settings: Settings, client: anthropic.Anthropic) -> Callable:
    """Draft an answer from all accumulated chunks."""

    def synthesizer(state: AgentState) -> dict:
        chunks = state["retrieved_chunks"]
        question = state["question"]

        if not chunks:
            log.warning("agentic.synthesizer.no_chunks")
            return {
                "draft_answer": "No relevant context was retrieved.",
                "sources": [],
            }

        context = "\n\n".join(
            f"[{i}] Source: {c['source']}\n{c['content']}"
            for i, c in enumerate(chunks, 1)
        )
        system, user, version = prompt_loader.format_user(
            "synthesizer", context=context, question=question
        )
        log.info("agentic.synthesizer.generating", chunks=len(chunks), prompt_version=version)

        msg = client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        draft = msg.content[0].text.strip()
        sources = [
            {"ref": i, "content": c["content"], "source": c["source"],
             "title": c.get("title"), "similarity": round(c["similarity"], 4)}
            for i, c in enumerate(chunks, 1)
        ]
        log.info("agentic.synthesizer.done", tokens=msg.usage.input_tokens + msg.usage.output_tokens)
        return {"draft_answer": draft, "sources": sources}

    return synthesizer


def make_critic(settings: Settings, client: anthropic.Anthropic) -> Callable:
    """Grade groundedness of the draft; emit rewrite query or mark supported."""

    def critic(state: AgentState) -> dict:
        draft = state.get("draft_answer", "") or ""
        chunks = state.get("retrieved_chunks", [])
        loops = state.get("critic_loops", 0)

        context = "\n\n".join(
            f"[{i}] {c['content']}" for i, c in enumerate(chunks, 1)
        )
        system, user, version = prompt_loader.format_user(
            "critic", context=context, draft=draft
        )
        log.info("agentic.critic.start", loop=loops, prompt_version=version)

        msg = client.messages.create(
            model=settings.claude_model,
            max_tokens=128,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        verdict = msg.content[0].text.strip().lower()
        log.info("agentic.critic.verdict", verdict=verdict[:80], loop=loops)

        update: dict[str, Any] = {"critic_verdict": verdict, "critic_loops": loops + 1}

        if verdict == "supported" or loops + 1 >= settings.max_critic_loops:
            update["final_answer"] = draft
        elif verdict.startswith("rewrite:"):
            rewrite_query = verdict[len("rewrite:"):].strip()
            update["sub_questions"] = [rewrite_query]

        return update

    return critic


def _chunk_key(chunk: dict) -> str:
    """Stable dedup key derived from source + first 120 chars of content."""
    raw = f"{chunk.get('source', '')}::{chunk.get('content', '')[:120]}"
    return hashlib.md5(raw.encode()).hexdigest()
