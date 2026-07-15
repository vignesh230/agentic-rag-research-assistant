"""Tests for the settings module."""

import pytest
from pydantic import ValidationError

from rag_agent.settings import Settings


def test_defaults_are_valid() -> None:
    s = Settings()
    assert s.chunk_size > 0
    assert s.chunk_overlap < s.chunk_size
    assert s.top_k >= 1
    assert s.max_critic_loops >= 1
    assert s.rag_mode in {"naive", "reranked", "agentic"}


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("TOP_K", "10")
    s = Settings()
    assert s.chunk_size == 256
    assert s.top_k == 10


def test_top_k_bounds() -> None:
    with pytest.raises(ValidationError):
        Settings(top_k=0)
    with pytest.raises(ValidationError):
        Settings(top_k=51)


def test_max_critic_loops_bounds() -> None:
    with pytest.raises(ValidationError):
        Settings(max_critic_loops=0)
    with pytest.raises(ValidationError):
        Settings(max_critic_loops=11)


def test_invalid_chunk_strategy() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_strategy="word")  # type: ignore[arg-type]


def test_invalid_rag_mode() -> None:
    with pytest.raises(ValidationError):
        Settings(rag_mode="fancy")  # type: ignore[arg-type]
