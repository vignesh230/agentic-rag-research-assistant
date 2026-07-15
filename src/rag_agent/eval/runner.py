"""Run one RAG mode over the full golden set and collect raw results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import structlog

from rag_agent.db.client import DBClient
from rag_agent.eval.schema import GoldenItem
from rag_agent.ingestion.embedder import Embedder
from rag_agent.rag import agentic, naive, reranked
from rag_agent.settings import Settings

log = structlog.get_logger(__name__)


@dataclass
class QueryResult:
    question: str
    ground_truth: str
    reference_contexts: Optional[List[str]]
    answer: str
    retrieved_contexts: List[str]
    latency_ms: float
    tokens_used: Optional[int]


@dataclass
class ModeRunResult:
    mode: str
    results: List[QueryResult] = field(default_factory=list)
    errors: int = 0


def run_mode(
    mode: str,
    golden: list[GoldenItem],
    settings: Settings,
    db: DBClient,
    embedder: Embedder,
) -> ModeRunResult:
    """Run every golden item through one RAG mode.

    Errors on individual questions are logged and counted but do not abort
    the run — partial results are still useful for comparison.

    Args:
        mode: One of "naive", "reranked", "agentic".
        golden: Validated golden set items.
        settings: Application settings (model, top_k, etc.)
        db: Initialised DBClient.
        embedder: Initialised Embedder.

    Returns:
        ModeRunResult with per-question outputs and an error count.
    """
    _ask = {"naive": naive.ask, "reranked": reranked.ask, "agentic": agentic.ask}[mode]
    run = ModeRunResult(mode=mode)

    for item in golden:
        try:
            resp = _ask(item.question, settings, db, embedder)
            run.results.append(QueryResult(
                question=item.question,
                ground_truth=item.ground_truth,
                reference_contexts=item.reference_contexts,
                answer=resp.answer,
                retrieved_contexts=[s.content for s in resp.sources],
                latency_ms=resp.latency_ms,
                tokens_used=resp.tokens_used,
            ))
            log.info("eval.runner.ok", mode=mode, q=item.question[:60])
        except Exception as exc:
            log.warning("eval.runner.error", mode=mode, q=item.question[:60], error=str(exc))
            run.errors += 1

    return run
