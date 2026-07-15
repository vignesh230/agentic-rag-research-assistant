"""Cross-encoder reranker for the reranked RAG mode.

Uses sentence-transformers CrossEncoder to score (query, chunk) pairs and
return the top-k chunks by score.  The model is cached at module level so
it loads once per process, not once per request.
"""

from __future__ import annotations

import structlog
from sentence_transformers import CrossEncoder

log = structlog.get_logger(__name__)

# ponytail: module-level cache mirrors embedder.py pattern — avoids reloading
# weights on every request without needing a singleton class.
_CE_CACHE: dict[str, CrossEncoder] = {}


def _get_model(name: str) -> CrossEncoder:
    if name not in _CE_CACHE:
        log.info("reranker.loading", model=name)
        _CE_CACHE[name] = CrossEncoder(name)
    return _CE_CACHE[name]


def rerank(query: str, chunks: list[dict], top_k: int, model_name: str) -> list[dict]:
    """Score (query, chunk) pairs with a cross-encoder and return the top-k.

    Args:
        query: The user query.
        chunks: Candidate chunks from vector search (any length).
        top_k: How many to return after reranking.
        model_name: HuggingFace cross-encoder model id.

    Returns:
        Up to top_k chunks sorted by descending cross-encoder score.
        Each chunk dict gains a "ce_score" key with the raw logit.
    """
    if not chunks:
        return []

    model = _get_model(model_name)
    pairs = [(query, c["content"]) for c in chunks]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(scores, chunks), key=lambda x: float(x[0]), reverse=True
    )
    result = []
    for score, chunk in ranked[:top_k]:
        result.append({**chunk, "ce_score": float(score)})

    log.info("reranker.done", candidates=len(chunks), returned=len(result), top_score=result[0]["ce_score"])
    return result
