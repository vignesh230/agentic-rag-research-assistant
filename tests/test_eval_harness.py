"""Unit tests for the evaluation harness (loader, metrics, table rendering)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_agent.eval.loader import load_golden_set
from rag_agent.eval.metrics import (
    ModeMetrics,
    compute_cost,
    compute_latency,
    render_table,
)
from rag_agent.eval.runner import ModeRunResult, QueryResult
from rag_agent.eval.schema import GoldenItem


# ── loader ────────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(l) for l in lines))


def test_loader_reads_valid_file():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        path = Path(f.name)

    _write_jsonl(path, [
        {"question": "What is BERT?", "ground_truth": "A transformer model."},
        {"question": "What is GPT?", "ground_truth": "A generative model.",
         "reference_contexts": ["GPT is a language model."]},
    ])
    items = load_golden_set(path)
    assert len(items) == 2
    assert items[0].reference_contexts is None
    assert items[1].reference_contexts == ["GPT is a language model."]


def test_loader_skips_comment_lines():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        path = Path(f.name)
    path.write_text(
        "# this is a comment\n"
        '{"question": "What is X?", "ground_truth": "X is Y."}\n'
        "\n"  # blank line
    )
    items = load_golden_set(path)
    assert len(items) == 1


def test_loader_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_golden_set(Path("/nonexistent/golden_set.jsonl"))


def test_loader_raises_on_invalid_json():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        path = Path(f.name)
    path.write_text('{"question": "Q?"}')  # missing ground_truth
    with pytest.raises(ValueError, match="line 1"):
        load_golden_set(path)


# ── metrics ───────────────────────────────────────────────────────────────────

def _qr(latency_ms: float, tokens: int | None = 30) -> QueryResult:
    return QueryResult(
        question="Q?",
        ground_truth="A.",
        reference_contexts=None,
        answer="Answer.",
        retrieved_contexts=["ctx"],
        latency_ms=latency_ms,
        tokens_used=tokens,
    )


def test_compute_latency_p50_p95():
    results = [_qr(lat) for lat in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    p50, p95 = compute_latency(results)
    assert p50 == 55.0  # median of 10 values
    assert p95 == 100.0


def test_compute_latency_single():
    p50, p95 = compute_latency([_qr(42.0)])
    assert p50 == 42.0
    assert p95 == 42.0


def test_compute_latency_empty():
    assert compute_latency([]) == (0.0, 0.0)


def test_compute_cost_estimate():
    results = [_qr(10, tokens=1000)]
    cost = compute_cost(results)
    assert cost is not None
    assert cost > 0


def test_compute_cost_no_tokens():
    results = [_qr(10, tokens=None)]
    assert compute_cost(results) is None


# ── render_table ──────────────────────────────────────────────────────────────

def test_render_table_structure():
    metrics = [
        ModeMetrics(mode="naive", faithfulness=0.85, answer_relevancy=0.90,
                    context_precision=0.75, context_recall=0.80,
                    latency_p50_ms=120.0, latency_p95_ms=300.0,
                    cost_per_query_usd=0.00045, n_questions=10, n_errors=0),
        ModeMetrics(mode="reranked", faithfulness=0.88, answer_relevancy=0.91,
                    context_precision=0.78, context_recall=0.83,
                    latency_p50_ms=180.0, latency_p95_ms=420.0,
                    cost_per_query_usd=0.00046, n_questions=10, n_errors=0),
    ]
    table = render_table(metrics)
    assert "| Mode |" in table
    assert "naive" in table
    assert "reranked" in table
    assert "0.850" in table  # faithfulness formatted to 3dp


def test_render_table_none_values():
    metrics = [
        ModeMetrics(mode="naive", n_questions=5, n_errors=1)
    ]
    table = render_table(metrics)
    assert "—" in table  # None fields shown as dash
