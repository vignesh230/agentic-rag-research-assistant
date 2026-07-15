"""Document loader — reads .txt, .md, and .pdf files from disk.

Returns a list of Document objects so the rest of the pipeline is
format-agnostic.  Add new formats here without touching chunker/embedder.
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from rag_agent.ingestion.models import Document

log = structlog.get_logger(__name__)

_SUPPORTED = {".txt", ".md", ".pdf"}


def load(source: str | Path) -> list[Document]:
    """Load one file or all supported files in a directory.

    Args:
        source: Path to a file or directory.

    Returns:
        List of Document objects, one per file.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If a file format is not supported.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")

    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.suffix.lower() in _SUPPORTED
        )
        log.info("loader.found_files", directory=str(path), count=len(files))
    else:
        files = [path]

    return [_load_file(f) for f in files]


def _load_file(path: Path) -> Document:
    """Load a single file and return a Document.

    Args:
        path: Path to an individual file.

    Returns:
        Document with content and inferred title.

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED:
        raise ValueError(f"Unsupported file type '{suffix}': {path}")

    log.info("loader.loading", file=str(path), format=suffix)

    if suffix == ".pdf":
        content = _load_pdf(path)
    else:
        content = path.read_text(encoding="utf-8", errors="replace")

    # Normalise whitespace: collapse runs of blank lines to a single blank line.
    content = re.sub(r"\n{3,}", "\n\n", content).strip()

    return Document(
        source=str(path.resolve()),
        content=content,
        title=path.stem.replace("_", " ").replace("-", " ").title(),
        metadata={"file_name": path.name, "format": suffix},
    )


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf.

    Args:
        path: Path to the PDF.

    Returns:
        Concatenated page text, pages separated by a form-feed character.
    """
    try:
        from pypdf import PdfReader  # lazy import — not required for txt/md
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF loading. Run: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\f".join(pages)
