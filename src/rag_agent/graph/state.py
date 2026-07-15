"""LangGraph state definition for the agentic RAG pipeline."""

from __future__ import annotations

from typing import List, Optional, TypedDict


class NodeTrace(TypedDict):
    """Per-node observability record appended by each graph node."""

    node: str
    latency_ms: float
    tokens: Optional[int]
    prompt_version: Optional[str]


class AgentState(TypedDict):
    """Shared mutable state threaded through every graph node.

    Design notes:
    - Flat structure: easier to log, inspect, and serialize than nested dicts.
    - retrieved_chunks accumulates across critic loops — each loop appends new
      evidence rather than replacing existing chunks, so the synthesizer always
      has the full context seen so far.
    - sub_questions is overwritten by the critic with a single rewrite query
      when it requests more retrieval, keeping retrieve() focused on only
      the new query rather than re-embedding already-processed questions.
    - node_traces accumulates one NodeTrace per graph node execution, enabling
      per-node latency and token-cost breakdowns in evaluation and logs.
    """

    question: str
    sub_questions: List[str]
    retrieved_chunks: List[dict]
    draft_answer: Optional[str]
    critic_verdict: Optional[str]
    critic_loops: int
    sources: List[dict]
    final_answer: Optional[str]
    node_traces: List[NodeTrace]
