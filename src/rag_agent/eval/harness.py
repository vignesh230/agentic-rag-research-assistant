"""Evaluation harness CLI.

Usage:
    python -m rag_agent.eval.harness
    python -m rag_agent.eval.harness --modes naive,reranked
    python -m rag_agent.eval.harness --golden data/golden_set.jsonl --out results.md

Requires:
    - data/golden_set.jsonl to exist and contain at least one entry.
    - ANTHROPIC_API_KEY set (or in .env).
    - Postgres running with documents already ingested.

Output:
    Prints a markdown comparison table to stdout (and optionally writes to --out).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from rag_agent.db.client import DBClient
from rag_agent.eval.loader import load_golden_set
from rag_agent.eval.metrics import ModeMetrics, build_metrics, render_table
from rag_agent.eval.runner import run_mode
from rag_agent.ingestion.embedder import Embedder
from rag_agent.logging_config import configure_logging
from rag_agent.settings import Settings

log = structlog.get_logger(__name__)

_ALL_MODES = ["naive", "reranked", "agentic"]


def run_eval(
    modes: list[str],
    golden_path: Path | None,
    settings: Settings,
) -> list[ModeMetrics]:
    """Run the full evaluation and return metrics for each mode.

    Args:
        modes: Which RAG modes to evaluate.
        golden_path: Path to golden_set.jsonl (None = default location).
        settings: Application settings.

    Returns:
        List of ModeMetrics, one per mode.
    """
    golden = load_golden_set(golden_path)
    log.info("eval.harness.start", n_questions=len(golden), modes=modes)

    db = DBClient(settings)
    embedder = Embedder(settings)

    all_metrics: list[ModeMetrics] = []
    for mode in modes:
        log.info("eval.harness.running_mode", mode=mode)
        run = run_mode(mode, golden, settings, db, embedder)
        m = build_metrics(run, settings.anthropic_api_key)
        all_metrics.append(m)
        log.info("eval.harness.mode_done", mode=mode, errors=m.n_errors)

    return all_metrics


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation harness")
    parser.add_argument(
        "--modes",
        default=",".join(_ALL_MODES),
        help="Comma-separated list of modes to evaluate (default: all three)",
    )
    parser.add_argument(
        "--golden",
        default=None,
        help="Path to golden_set.jsonl (default: data/golden_set.jsonl)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write markdown table to this file in addition to stdout",
    )
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level)

    modes = [m.strip() for m in args.modes.split(",") if m.strip() in _ALL_MODES]
    if not modes:
        print(f"No valid modes specified. Choose from: {_ALL_MODES}", file=sys.stderr)
        sys.exit(1)

    golden_path = Path(args.golden) if args.golden else None

    try:
        all_metrics = run_eval(modes, golden_path, settings)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    table = render_table(all_metrics)
    print(table)

    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"\nResults written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    _cli()
