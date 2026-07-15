"""Shared pytest fixtures."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rag_agent.ingestion.models import Document
from rag_agent.settings import Settings


@pytest.fixture()
def settings() -> Settings:
    """Settings with fast/small values for unit tests."""
    return Settings(
        postgres_dsn="postgresql://rag:rag@localhost:5432/ragdb",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        chunk_strategy="recursive",
        chunk_size=200,
        chunk_overlap=20,
        top_k=3,
        anthropic_api_key="test-key",
        claude_model="claude-sonnet-4-6",
    )


@pytest.fixture()
def sample_document() -> Document:
    """A small document with predictable structure for chunking tests."""
    content = textwrap.dedent(
        """\
        Graph Neural Networks (GNNs) are deep learning models for graph data.

        Unlike CNNs that work on grids, GNNs operate on arbitrary graphs.
        They aggregate features from neighbouring nodes to learn representations.

        Common architectures include GCN, GAT, and GraphSAGE.
        Each differs in how neighbourhood aggregation is weighted.

        FAISS is a library for efficient similarity search over dense vectors.
        It supports flat, IVFFlat, and HNSW index types.
        """
    )
    return Document(
        source="/tmp/test_doc.txt",
        content=content,
        title="Test Document",
        metadata={"format": ".txt"},
    )


@pytest.fixture()
def sample_txt_file(tmp_path: Path, sample_document: Document) -> Path:
    """Write the sample document to a temporary .txt file."""
    p = tmp_path / "test_doc.txt"
    p.write_text(sample_document.content, encoding="utf-8")
    return p


@pytest.fixture()
def mock_db() -> MagicMock:
    """DBClient mock with sensible return values."""
    db = MagicMock()
    db.document_exists.return_value = False
    db.insert_document.return_value = "00000000-0000-0000-0000-000000000001"
    return db


@pytest.fixture()
def mock_embedder(settings: Settings) -> MagicMock:
    """Embedder mock that returns deterministic zero vectors."""
    emb = MagicMock()
    emb.embed.side_effect = lambda texts, **_: np.zeros(
        (len(texts), settings.embedding_dim), dtype=np.float32
    )
    emb.embed_one.side_effect = lambda text: np.zeros(
        settings.embedding_dim, dtype=np.float32
    )
    return emb
