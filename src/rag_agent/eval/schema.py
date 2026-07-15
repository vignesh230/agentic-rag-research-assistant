"""Golden set schema for the evaluation harness.

Each line of data/golden_set.jsonl must be a JSON object matching GoldenItem.
You write this file by hand — the harness will validate it on load.

Schema example (do NOT copy placeholders, write real entries):
{
  "question": "What mechanism does BERT use for pre-training?",
  "ground_truth": "BERT uses masked language modelling (MLM) and next sentence prediction (NSP).",
  "reference_contexts": [
    "BERT is pre-trained using two tasks: masked language modeling and next sentence prediction."
  ]
}

Fields:
  question          (required) The query to pose to each RAG mode.
  ground_truth      (required) The reference answer; used for faithfulness and
                    answer_relevancy scoring.
  reference_contexts (optional) Ground-truth passages; enables context_precision
                    and context_recall.  Omit if you only have answer-level ground truth.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    question: str = Field(..., min_length=5)
    ground_truth: str = Field(..., min_length=5)
    reference_contexts: Optional[List[str]] = None
