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


def compute_ragas(run: ModeRunResult, anthropic_api_key: str) -> Dict[str, Optional[float]]:
    """Run RAGAS evaluation on the collected results.

    Uses Claude (via LangChain wrapper) as the judge LLM so no OpenAI key
    is required.  Returns a dict of metric_name → score (0-1) or None on error.

    Args:
        run: Completed ModeRunResult with answers and retrieved contexts.
        anthropic_api_key: Key for the Claude judge LLM.

    Returns:
        Dict with keys: faithfulness, answer_relevancy, context_precision,
        context_recall.  Any metric that errors returns None.
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
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from langchain_anthropic import ChatAnthropic
            from ragas import EvaluationDataset, SingleTurnSample, evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics.collections import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

        llm = LangchainLLMWrapper(
            ChatAnthropic(
                model="claude-haiku-4-5-20251001",  # cheaper judge; swap to sonnet for higher accuracy
                api_key=anthropic_api_key,
            )
        )

        samples = [
            SingleTurnSample(
                user_input=r.question,
                response=r.answer,
                retrieved_contexts=r.retrieved_contexts,
                reference=r.ground_truth,
                reference_contexts=r.reference_contexts,
            )
            for r in run.results
        ]
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            show_progress=False,
            raise_exceptions=False,
        )
        scores = result.to_pandas().mean(numeric_only=True)

        log.info("eval.ragas.done", mode=run.mode, scores=scores.to_dict())
        return {
            "faithfulness": _safe_float(scores.get("faithfulness")),
            "answer_relevancy": _safe_float(scores.get("answer_relevancy")),
            "context_precision": _safe_float(scores.get("context_precision")),
            "context_recall": _safe_float(scores.get("context_recall")),
        }

    except Exception as exc:
        log.warning("eval.ragas.failed", mode=run.mode, error=str(exc))
        return empty


def build_metrics(run: ModeRunResult, anthropic_api_key: str) -> ModeMetrics:
    """Assemble all metrics for one mode into a ModeMetrics object."""
    p50, p95 = compute_latency(run.results)
    cost = compute_cost(run.results)
    ragas_scores = compute_ragas(run, anthropic_api_key)

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
