"""Request and response schemas for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Payload for POST /ask."""

    question: str = Field(..., min_length=3, max_length=2000)
    # Per-request overrides — if None, falls back to settings defaults.
    mode: Literal["naive", "reranked", "agentic"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class Source(BaseModel):
    """A single retrieved chunk cited in the answer."""

    ref: int
    content: str
    source: str
    title: str | None = None
    similarity: float


class AskResponse(BaseModel):
    """Response from POST /ask."""

    answer: str
    sources: list[Source]
    mode: str
    latency_ms: float
    tokens_used: int | None = None
    prompt_version: str | None = None


class HealthResponse(BaseModel):
    status: str
    mode: str
    embedding_model: str
