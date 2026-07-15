"""Tests for the ingestion pipeline (DB and Embedder mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_agent.ingestion import pipeline
from rag_agent.ingestion.models import Document
from rag_agent.settings import Settings


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        chunk_strategy="recursive",
        chunk_size=100,
        chunk_overlap=10,
        embedding_dim=384,
    )


@pytest.fixture()
def txt_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.txt"
    f.write_text(
        "Graph neural networks are models for graph-structured data. "
        "They aggregate features from neighbouring nodes iteratively. "
        "Applications include drug discovery, social network analysis, "
        "and recommendation systems. FAISS provides fast similarity search.",
        encoding="utf-8",
    )
    return f


def _fake_embed(texts: list[str], **_) -> np.ndarray:
    return np.zeros((len(texts), 384), dtype=np.float32)


class TestRunPipeline:
    def test_ingest_single_file(
        self, txt_file: Path, settings: Settings, mock_db: MagicMock
    ) -> None:
        with patch(
            "rag_agent.ingestion.pipeline.Embedder"
        ) as MockEmb:
            MockEmb.return_value.embed.side_effect = _fake_embed
            stats = pipeline.run(txt_file, settings, mock_db)

        assert stats["ingested"] == 1
        assert stats["skipped"] == 0
        assert stats["total_chunks"] >= 1
        mock_db.insert_document.assert_called_once()
        mock_db.insert_chunks.assert_called_once()

    def test_skips_existing_document(
        self, txt_file: Path, settings: Settings, mock_db: MagicMock
    ) -> None:
        mock_db.document_exists.return_value = True  # already in DB

        with patch("rag_agent.ingestion.pipeline.Embedder") as MockEmb:
            MockEmb.return_value.embed.side_effect = _fake_embed
            stats = pipeline.run(txt_file, settings, mock_db)

        assert stats["ingested"] == 0
        assert stats["skipped"] == 1
        mock_db.insert_document.assert_not_called()

    def test_ingest_directory(
        self, tmp_path: Path, settings: Settings, mock_db: MagicMock
    ) -> None:
        for i in range(3):
            (tmp_path / f"doc_{i}.txt").write_text(
                f"Document {i} " * 50, encoding="utf-8"
            )

        with patch("rag_agent.ingestion.pipeline.Embedder") as MockEmb:
            MockEmb.return_value.embed.side_effect = _fake_embed
            stats = pipeline.run(tmp_path, settings, mock_db)

        assert stats["ingested"] == 3
        assert mock_db.insert_document.call_count == 3

    def test_embedder_called_with_chunk_texts(
        self, txt_file: Path, settings: Settings, mock_db: MagicMock
    ) -> None:
        with patch("rag_agent.ingestion.pipeline.Embedder") as MockEmb:
            mock_emb_instance = MagicMock()
            mock_emb_instance.embed.side_effect = _fake_embed
            MockEmb.return_value = mock_emb_instance

            pipeline.run(txt_file, settings, mock_db)

        call_args = mock_emb_instance.embed.call_args
        texts_passed = call_args[0][0]
        assert isinstance(texts_passed, list)
        assert all(isinstance(t, str) for t in texts_passed)

    def test_returns_correct_chunk_count(
        self, tmp_path: Path, settings: Settings, mock_db: MagicMock
    ) -> None:
        f = tmp_path / "big.txt"
        # ~2000 chars — should produce multiple chunks at size=100
        f.write_text("word " * 400, encoding="utf-8")

        with patch("rag_agent.ingestion.pipeline.Embedder") as MockEmb:
            MockEmb.return_value.embed.side_effect = _fake_embed
            stats = pipeline.run(f, settings, mock_db)

        assert stats["total_chunks"] > 1
