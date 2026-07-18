"""Text chunking strategies.

Three strategies are provided, selectable via settings.chunk_strategy:

  fixed      — character-window sliding split.  Fast, ignores sentence
               boundaries; good for ablation baselines.

  sentence   — NLTK sentence tokenizer groups sentences until chunk_size
               is reached, then starts a new chunk.  Respects sentence
               boundaries but can produce very uneven chunk sizes.

  recursive  — LangChain RecursiveCharacterTextSplitter tries paragraph ->
               sentence -> word -> character separators in order.  Best
               semantic coherence at a given chunk_size; default choice.
"""

from __future__ import annotations

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent.ingestion.models import Chunk, Document
from rag_agent.settings import Settings

log = structlog.get_logger(__name__)


def chunk(doc: Document, settings: Settings) -> list[Chunk]:
    """Split a Document into Chunks using the configured strategy.

    Args:
        doc: The source document.
        settings: Application settings (reads chunk_strategy, chunk_size,
                  chunk_overlap).

    Returns:
        Ordered list of Chunk objects with accurate char offsets.
    """
    strategy = settings.chunk_strategy
    log.info(
        "chunker.splitting",
        source=doc.source,
        strategy=strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if strategy == "fixed":
        return _fixed(doc, settings.chunk_size, settings.chunk_overlap)
    elif strategy == "sentence":
        return _sentence(doc, settings.chunk_size, settings.chunk_overlap)
    elif strategy == "recursive":
        return _recursive(doc, settings.chunk_size, settings.chunk_overlap)
    else:
        raise ValueError(f"Unknown chunk_strategy: {strategy!r}")


# ── Strategy implementations ──────────────────────────────────────────────────


def _fixed(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Sliding window over raw characters.

    Args:
        doc: Source document.
        size: Window size in characters.
        overlap: Number of characters to repeat between consecutive windows.

    Returns:
        List of Chunk objects.
    """
    text = doc.content
    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    idx = 0
    i = 0
    while idx < len(text):
        end = min(idx + size, len(text))
        chunks.append(
            Chunk(
                content=text[idx:end],
                chunk_index=i,
                start_char=idx,
                end_char=end,
            )
        )
        if end == len(text):
            break
        idx += step
        i += 1
    return chunks


def _sentence(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Group NLTK sentences into chunks up to `size` characters.

    Overlap is approximated by including the last N characters worth of
    sentences from the previous chunk at the start of the next one.

    Args:
        doc: Source document.
        size: Target maximum chunk size in characters.
        overlap: Approximate character overlap between chunks.

    Returns:
        List of Chunk objects.
    """
    try:
        import nltk

        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
    except ImportError as exc:
        raise ImportError(
            "nltk is required for sentence chunking: pip install nltk"
        ) from exc

    text = doc.content
    sentences = sent_tokenize(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    char_offset = 0
    chunk_index = 0

    for sent in sentences:
        if current and current_len + len(sent) > size:
            content = " ".join(current)
            start = char_offset
            end = start + len(content)
            chunks.append(
                Chunk(
                    content=content,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                )
            )
            char_offset = end

            # Carry forward sentences until we have `overlap` chars.
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len >= overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s)
            current = overlap_sents
            current_len = overlap_len
            chunk_index += 1

        current.append(sent)
        current_len += len(sent)

    if current:
        content = " ".join(current)
        chunks.append(
            Chunk(
                content=content,
                chunk_index=chunk_index,
                start_char=char_offset,
                end_char=char_offset + len(content),
            )
        )

    return chunks


def _recursive(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """LangChain RecursiveCharacterTextSplitter.

    Tries \\n\\n, \\n, ' ', '' separators in order — paragraph splits are
    preferred, falling back to word then character splits only when necessary.
    This preserves semantic units (paragraphs, sentences) better than fixed
    splitting at equivalent chunk sizes.

    Args:
        doc: Source document.
        size: Target chunk size in characters.
        overlap: Character overlap between chunks.

    Returns:
        List of Chunk objects with accurate char offsets.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True,  # includes start_index in metadata
    )
    lc_docs = splitter.create_documents([doc.content])
    return [
        Chunk(
            content=d.page_content,
            chunk_index=i,
            start_char=d.metadata.get("start_index", 0),
            end_char=d.metadata.get("start_index", 0) + len(d.page_content),
        )
        for i, d in enumerate(lc_docs)
    ]
