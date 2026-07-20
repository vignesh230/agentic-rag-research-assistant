"""Compute RAGAS metrics, latency stats, and cost estimates from run results."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, Optional

import structlog

from rag_agent.eval.runner import ModeRunResult, QueryResult

log = structlog.get_logger(__name__)

# Approximate claude-sonnet-4-6 pricing (USD per token).
# Update when Anthropic changes pricing.
_INPUT_COST_PER_TOKEN = 3.00 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000
# Assume ~80/20 input/output split for cost estimation when only total is known.
_INPUT_FRACTION = 0.8


@dataclass
class ModeMetrics:
    mode: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    cost_per_query_usd: Optional[float] = None
    n_questions: int = 0
    n_errors: int = 0
    ragas_error: Optional[str] = None


def compute_latency(results: list[QueryResult]) -> tuple[float, float]:
    """Return (p50, p95) latency in ms."""
    if not results:
        return 0.0, 0.0
    lats = sorted(r.latency_ms for r in results)
    p50 = statistics.median(lats)
    idx_95 = min(len(lats) - 1, int(len(lats) * 0.95))
    p95 = lats[idx_95]
    return round(p50, 1), round(p95, 1)


def compute_cost(results: list[QueryResult]) -> Optional[float]:
    """Estimate average cost per query in USD from token counts."""
    totals = [r.tokens_used for r in results if r.tokens_used is not None]
    if not totals:
        return None
    avg_tokens = statistics.mean(totals)
    # Assume 80/20 input/output split when only the total is available.
    input_t = avg_tokens * _INPUT_FRACTION
    output_t = avg_tokens * (1 - _INPUT_FRACTION)
    cost = input_t * _INPUT_COST_PER_TOKEN + output_t * _OUTPUT_COST_PER_TOKEN
    return round(cost, 6)


def _cosine(a, b) -> float:
    """Cosine similarity between two 1-D arrays."""
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def compute_ragas(run: ModeRunResult, settings=None, embedder=None) -> Dict[str, Optional[float]]:
    """Compute retrieval metrics using embedding cosine similarity.

    ragas 0.4.x LLM-graded metrics require the instructor library for
    structured output, which NIM's Llama endpoint does not support (hangs).
    We use embedding-based proxies instead:

    - context_recall:   for each reference context, max cosine similarity
                        across retrieved chunks; averaged over all questions.
                        Measures whether the retriever surfaces passages that
                        cover the reference material.
    - answer_relevancy: cosine similarity between the answer embedding and the
                        question embedding. Measures whether the answer
                        addresses the question (not whether it matches the
                        ground truth — that would be faithfulness).
    - faithfulness / context_precision: None (require LLM judge).

    Args:
        run: Completed ModeRunResult with answers and retrieved contexts.
        settings: Unused; kept for signature compatibility.
        embedder: Embedder instance. If None, falls back to a fresh
                  SentenceTransformer load (slow; pass embedder from harness).

    Returns:
        Dict with keys: faithfulness, answer_relevancy, context_precision,
        context_recall.
    """
    empty: Dict[str, Optional[float]] = {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }
    if not run.results:
        return empty

    try:
        # Resolve embedder: use passed-in instance or load model directly.
        if embedder is not None:
            _embed = embedder.embed
        else:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            _embed = lambda texts: _model.encode(texts, normalize_embeddings=True)

        recall_scores: list[float] = []
        relevancy_scores: list[float] = []

        for r in run.results:
            ref_ctxs = r.reference_contexts or []
            ret_ctxs = r.retrieved_contexts or []

            # context_recall: for each reference context, best cosine match
            # among retrieved chunks. Rewards the retriever for surfacing any
            # chunk that covers a reference passage.
            if ref_ctxs and ret_ctxs:
                all_texts = ref_ctxs + ret_ctxs
                embs = _embed(all_texts)
                ref_embs = embs[:len(ref_ctxs)]
                ret_embs = embs[len(ref_ctxs):]
                per_ref = [
                    max(_cosine(ref_e, ret_e) for ret_e in ret_embs)
                    for ref_e in ref_embs
                ]
                recall_scores.append(sum(per_ref) / len(per_ref))

            # answer_relevancy: does the answer address the question?
            # Cosine sim between answer and question embeddings.
            if r.answer and r.question:
                a_emb, q_emb = _embed([r.answer, r.question])
                relevancy_scores.append(_cosine(a_emb, q_emb))

        def _mean(lst: list) -> Optional[float]:
            return round(sum(lst) / len(lst), 4) if lst else None

        result = {
            "faithfulness": None,
            "answer_relevancy": _mean(relevancy_scores),
            "context_precision": None,
            "context_recall": _mean(recall_scores),
        }
        log.info("eval.metrics.done", mode=run.mode, **{k: v for k, v in result.items() if v is not None})
        return result

    except Exception as exc:
        log.warning("eval.metrics.failed", mode=run.mode, error=str(exc))
        return empty


def build_metrics(run: ModeRunResult, settings=None, embedder=None) -> ModeMetrics:
    """Assemble all metrics for one mode into a ModeMetrics object."""
    p50, p95 = compute_latency(run.results)
    cost = compute_cost(run.results)
    ragas_scores = compute_ragas(run, settings, embedder=embedder)

    return ModeMetrics(
        mode=run.mode,
        faithfulness=ragas_scores["faithfulness"],
        answer_relevancy=ragas_scores["answer_relevancy"],
        context_precision=ragas_scores["context_precision"],
        context_recall=ragas_scores["context_recall"],
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        cost_per_query_usd=cost,
        n_questions=len(run.results),
        n_errors=run.errors,
    )


def render_table(metrics: list[ModeMetrics]) -> str:
    """Render a markdown comparison table from a list of ModeMetrics."""

    def _fmt(v: Optional[float], decimals: int = 3) -> str:
        return f"{v:.{decimals}f}" if v is not None else "—"

    header = (
        "| Mode | Faithfulness | Answer Relevancy | Context Precision | "
        "Context Recall | Latency p50 (ms) | Latency p95 (ms) | Cost/query ($) | Questions | Errors |\n"
        "|------|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for m in metrics:
        rows.append(
            f"| {m.mode} "
            f"| {_fmt(m.faithfulness)} "
            f"| {_fmt(m.answer_relevancy)} "
            f"| {_fmt(m.context_precision)} "
            f"| {_fmt(m.context_recall)} "
            f"| {m.latency_p50_ms} "
            f"| {m.latency_p95_ms} "
            f"| {_fmt(m.cost_per_query_usd, 5)} "
            f"| {m.n_questions} "
            f"| {m.n_errors} |"
        )
    return "\n".join([header] + rows)


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return round(f, 4) if not (f != f) else None  # NaN check
    except (TypeError, ValueError):
        return None
