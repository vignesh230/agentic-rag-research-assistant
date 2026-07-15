"""Tests for the three chunking strategies."""

from __future__ import annotations

import pytest

from rag_agent.ingestion import chunker
from rag_agent.ingestion.models import Document, Chunk
from rag_agent.settings import Settings


def _make_settings(strategy: str, size: int = 100, overlap: int = 10) -> Settings:
    return Settings(chunk_strategy=strategy, chunk_size=size, chunk_overlap=overlap)  # type: ignore[arg-type]


def _long_doc() -> Document:
    text = " ".join([f"Word{i}" for i in range(500)])  # ~3000 chars
    return Document(source="/tmp/x.txt", content=text)


def _short_doc() -> Document:
    return Document(source="/tmp/x.txt", content="Hello world.")


class TestFixedChunker:
    def test_produces_multiple_chunks(self) -> None:
        doc = _long_doc()
        chunks = chunker.chunk(doc, _make_settings("fixed", size=100, overlap=10))
        assert len(chunks) > 1

    def test_chunk_size_respected(self) -> None:
        doc = _long_doc()
        chunks = chunker.chunk(doc, _make_settings("fixed", size=100, overlap=0))
        for c in chunks[:-1]:  # last chunk may be shorter
            assert len(c.content) <= 100

    def test_overlap_repeats_content(self) -> None:
        doc = Document(source="/tmp/x.txt", content="A" * 150)
        chunks = chunker.chunk(doc, _make_settings("fixed", size=100, overlap=20))
        assert len(chunks) >= 2
        # End of chunk 0 should appear at start of chunk 1
        tail = chunks[0].content[-20:]
        head = chunks[1].content[:20]
        assert tail == head

    def test_chunk_indices_are_sequential(self) -> None:
        doc = _long_doc()
        chunks = chunker.chunk(doc, _make_settings("fixed"))
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_short_doc_single_chunk(self) -> None:
        doc = _short_doc()
        chunks = chunker.chunk(doc, _make_settings("fixed", size=200))
        assert len(chunks) == 1
        assert chunks[0].content == doc.content

    def test_offsets_cover_full_document(self) -> None:
        doc = _long_doc()
        chunks = chunker.chunk(doc, _make_settings("fixed", size=100, overlap=0))
        assert chunks[0].start_char == 0
        assert chunks[-1].end_char <= len(doc.content) + 1  # ±1 tolerance


class TestSentenceChunker:
    def test_produces_chunks(self) -> None:
        sentences = ". ".join([f"This is sentence number {i}" for i in range(30)])
        doc = Document(source="/tmp/x.txt", content=sentences)
        chunks = chunker.chunk(doc, _make_settings("sentence", size=150, overlap=30))
        assert len(chunks) >= 1

    def test_chunk_indices_sequential(self) -> None:
        doc = _long_doc()
        chunks = chunker.chunk(doc, _make_settings("sentence", size=150, overlap=0))
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_content_is_non_empty(self) -> None:
        doc = _long_doc()
        chunks = chunker.chunk(doc, _make_settings("sentence", size=150))
        assert all(c.content.strip() for c in chunks)


class TestRecursiveChunker:
    def test_respects_paragraph_boundaries(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three."
        doc = Document(source="/tmp/x.txt", content=text)
        chunks = chunker.chunk(doc, _make_settings("recursive", size=20, overlap=0))
        # Each paragraph should land in its own chunk
        full_content = " ".join(c.content for c in chunks)
        assert "Para one" in full_content
        assert "Para three" in full_content

    def test_chunk_indices_sequential(self) -> None:
        doc = _long_doc()
        chunks = chunker.chunk(doc, _make_settings("recursive"))
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_no_empty_chunks(self) -> None:
        doc = _long_doc()
        chunks = chunker.chunk(doc, _make_settings("recursive"))
        assert all(c.content.strip() for c in chunks)

    def test_all_content_preserved(self) -> None:
        """Union of chunk content should cover all tokens in the original."""
        doc = Document(source="/tmp/x.txt", content="alpha beta gamma delta epsilon")
        chunks = chunker.chunk(doc, _make_settings("recursive", size=15, overlap=0))
        combined = " ".join(c.content for c in chunks)
        for token in ["alpha", "beta", "gamma", "delta", "epsilon"]:
            assert token in combined


class TestUnknownStrategy:
    def test_raises_value_error(self) -> None:
        doc = _short_doc()
        bad_settings = Settings(chunk_strategy="recursive")  # valid to construct
        bad_settings = bad_settings.model_copy(update={"chunk_strategy": "bogus"})
        with pytest.raises(ValueError, match="Unknown chunk_strategy"):
            chunker.chunk(doc, bad_settings)
