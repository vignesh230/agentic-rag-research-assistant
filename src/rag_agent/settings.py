"""Central settings module — every tunable knob lives here.

All ablation dimensions (chunk_strategy, top_k, rag_mode, max_critic_loops)
are first-class settings so experiments can be run by env-var override alone,
no code changes required.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Database ──────────────────────────────────────────────────────────────
    postgres_dsn: str = "postgresql://rag:rag@localhost:5432/ragdb"

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Dimension must match the model; schema is created with this value.
    # Changing the model without updating this (and recreating the table) will
    # cause a silent dimension mismatch at insert time.
    embedding_dim: int = 384

    # ── Chunking ──────────────────────────────────────────────────────────────
    # "recursive" is the default: it respects paragraph/sentence boundaries
    # before falling back to character splits, which tends to preserve semantic
    # coherence better than "fixed" at the same chunk_size.
    chunk_strategy: Literal["fixed", "sentence", "recursive"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(default=5, ge=1, le=50)
    # reranked mode retrieves top_k * retrieval_multiplier candidates before
    # cross-encoder reranking narrows them back to top_k.
    retrieval_multiplier: int = 4
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Generation ────────────────────────────────────────────────────────────
    llm_provider: Literal["anthropic", "nvidia"] = "anthropic"
    # Anthropic
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    # NVIDIA NIM (OpenAI-compatible endpoint)
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.1-8b-instruct"
    # Cap critic loops to prevent runaway retries; 3 is enough for convergence
    # in practice while keeping cost bounded.
    max_critic_loops: int = Field(default=3, ge=1, le=10)

    # ── Service ───────────────────────────────────────────────────────────────
    rag_mode: Literal["naive", "reranked", "agentic"] = "naive"
    log_level: str = "INFO"


# Module-level singleton — import this instead of constructing in every module.
settings = Settings()
