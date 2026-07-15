"""Ingestion pipeline — orchestrates load -> chunk -> embed -> store.

Run as a CLI::

    python -m rag_agent.ingestion.pipeline --source docs/

Or import and call ``run()`` programmatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import structlog

from rag_agent.db.client import DBClient
from rag_agent.ingestion import chunker, loader
from rag_agent.ingestion.embedder import Embedder
from rag_agent.logging_config import configure_logging
from rag_agent.settings import Settings

log = structlog.get_logger(__name__)


def run(source: str | Path, settings: Settings, db: DBClient) -> dict[str, int]:
    """Ingest all documents from source into the vector store.

    Documents already present (by source path) are skipped so the pipeline
    is safe to re-run incrementally.

    Args:
        source: Path to a file or directory.
        settings: Application settings controlling chunking/embedding config.
        db: Initialised DBClient (tables must already exist).

    Returns:
        Dict with counts: {"ingested": N, "skipped": N, "total_chunks": N}.
    """
    embedder = Embedder(settings)
    docs = loader.load(source)
    stats: dict[str, int] = {"ingested": 0, "skipped": 0, "total_chunks": 0}

    for doc in docs:
        if db.document_exists(doc.source):
            log.info("pipeline.skipping", source=doc.source)
            stats["skipped"] += 1
            continue

        log.info("pipeline.ingesting", source=doc.source)

        chunks = chunker.chunk(doc, settings)
        if not chunks:
            log.warning("pipeline.no_chunks", source=doc.source)
            continue

        texts = [c.content for c in chunks]
        embeddings = embedder.embed(texts)

        doc_id = db.insert_document(
            source=doc.source,
            title=doc.title,
            metadata=doc.metadata,
        )
        db.insert_chunks(
            doc_id=doc_id,
            chunks=[c.model_dump() for c in chunks],
            embeddings=embeddings,
        )

        stats["ingested"] += 1
        stats["total_chunks"] += len(chunks)
        log.info(
            "pipeline.done",
            source=doc.source,
            n_chunks=len(chunks),
        )

    log.info("pipeline.summary", **stats)
    return stats


def _cli() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG store")
    parser.add_argument("--source", required=True, help="File or directory to ingest")
    args = parser.parse_args()

    settings = Settings()
    db = DBClient(settings)
    db.create_tables()
    run(args.source, settings, db)


if __name__ == "__main__":
    _cli()
