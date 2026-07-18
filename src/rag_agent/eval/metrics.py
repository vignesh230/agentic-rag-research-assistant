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


def compute_ragas(run: ModeRunResult, settings=None) -> Dict[str, Optional[float]]:
    """Compute reference-based retrieval metrics without an LLM judge.

    ragas 0.4.x LLM-graded metrics require OpenAI structured-output support
    (instructor / json_schema), which NIM's Llama endpoint does not provide.
    We fall back to string-overlap proxies that are deterministic and fast:

    - context_recall:    fraction of reference-context tokens found in
                         retrieved contexts (token-level F1 via difflib)
    - answer_relevancy:  BLEU-2 score between answer and ground truth
                         (NLTK bigram smoother — not RougeL despite the field name)
    - faithfulness / context_precision: None (require LLM judge)

    Args:
        run: Completed ModeRunResult with answers and retrieved contexts.
        settings: Unused; kept for signature compatibility.

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
        import difflib
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        recall_scores: list[float] = []
        relevancy_scores: list[float] = []

        for r in run.results:
            # context_recall: how much of the reference context is covered
            ref_tokens = " ".join(r.reference_contexts or []).lower().split()
            ret_tokens = " ".join(r.retrieved_contexts or []).lower().split()
            if ref_tokens:
                m = difflib.SequenceMatcher(None, ref_tokens, ret_tokens)
                recall_scores.append(m.ratio())

            # answer_relevancy: BLEU-2 between answer and ground truth
            hyp = r.answer.lower().split()
            ref = [r.ground_truth.lower().split()]
            if hyp and ref[0]:
                score = sentence_bleu(
                    ref, hyp,
                    weights=(0.5, 0.5),
                    smoothing_function=SmoothingFunction().method1,
                )
                relevancy_scores.append(score)

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

    except (ImportError, LookupError, ValueError) as exc:
        log.warning("eval.metrics.failed", mode=run.mode, error=str(exc))
        return empty


def build_metrics(run: ModeRunResult, settings=None) -> ModeMetrics:
    """Assemble all metrics for one mode into a ModeMetrics object."""
    p50, p95 = compute_latency(run.results)
    cost = compute_cost(run.results)
    ragas_scores = compute_ragas(run, settings)

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
