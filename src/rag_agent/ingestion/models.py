"""Shared data models for the ingestion pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A raw document loaded from disk or a URL."""

    source: str
    content: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A text segment derived from a Document."""

    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = Field(default_factory=dict)
