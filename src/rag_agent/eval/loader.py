"""Load and validate the hand-written golden set from data/golden_set.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from rag_agent.eval.schema import GoldenItem

log = structlog.get_logger(__name__)

_DEFAULT_PATH = Path(__file__).parents[3] / "data" / "golden_set.jsonl"


def load_golden_set(path: Path | None = None) -> list[GoldenItem]:
    """Load and validate every line of the golden set JSONL file.

    Args:
        path: Path to the JSONL file.  Defaults to data/golden_set.jsonl
              relative to the project root.

    Returns:
        List of validated GoldenItem instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any line fails schema validation (line number included).
    """
    target = Path(path) if path else _DEFAULT_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"Golden set not found at {target}. "
            "Write your questions and ground-truth answers there first. "
            "See src/rag_agent/eval/schema.py for the expected format."
        )

    items: list[GoldenItem] = []
    for i, raw in enumerate(target.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            items.append(GoldenItem.model_validate(json.loads(raw)))
        except Exception as exc:
            raise ValueError(f"golden_set.jsonl line {i}: {exc}") from exc

    log.info("eval.loader.loaded", n=len(items), path=str(target))
    return items
