"""Postgres + pgvector client (synchronous, psycopg2).

Phase 1 uses sync psycopg2 for simplicity; Phase 2 (FastAPI) will layer
an async connection pool on top via asyncpg.  The interface is kept narrow
so swapping the transport is a local change.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Generator

import numpy as np
import psycopg2
import psycopg2.extras
import structlog
from pgvector.psycopg2 import register_vector

from rag_agent.settings import Settings

log = structlog.get_logger(__name__)


class DBClient:
    """Thin wrapper around psycopg2 + pgvector for document/chunk CRUD."""

    def __init__(self, settings: Settings) -> None:
        self._dsn = settings.postgres_dsn
        self._dim = settings.embedding_dim

    # ── Connection ────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Yield a connection with pgvector registered; always commit/rollback."""
        conn = psycopg2.connect(self._dsn)
        register_vector(conn)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Schema ────────────────────────────────────────────────────────────────

    def create_tables(self) -> None:
        """Create the pgvector extension and application tables if absent.

        The vector dimension is taken from settings so the schema always
        matches the configured embedding model.  If you change the model,
        drop and recreate the chunks table (or use Alembic migrations).
        """
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS documents (
            id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source   TEXT NOT NULL UNIQUE,
            title    TEXT,
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id      UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            content     TEXT NOT NULL,
            embedding   vector({self._dim}),
            chunk_index INTEGER NOT NULL,
            start_char  INTEGER,
            end_char    INTEGER,
            metadata    JSONB DEFAULT '{{}}',
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

        -- HNSW chosen over IVFFlat: works on empty tables (no training step),
        -- faster queries at the cost of slightly higher insert latency.
        -- m=16, ef_construction=64 are pgvector defaults; tune for recall/speed.
        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
        log.info("db.tables_ready", dim=self._dim)

    # ── Documents ─────────────────────────────────────────────────────────────

    def document_exists(self, source: str) -> bool:
        """Return True if a document with this source path is already stored."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM documents WHERE source = %s LIMIT 1", (source,)
                )
                return cur.fetchone() is not None

    def insert_document(
        self,
        source: str,
        title: str | None,
        metadata: dict[str, Any],
    ) -> uuid.UUID:
        """Insert a document record and return its UUID.

        Args:
            source: Canonical file path or URL — used as dedup key.
            title: Human-readable title (may be None).
            metadata: Arbitrary key-value pairs stored as JSONB.

        Returns:
            The newly created document UUID.
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (source, title, metadata)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (source, title, psycopg2.extras.Json(metadata)),
                )
                row = cur.fetchone()
                assert row is not None
                doc_id: uuid.UUID = row[0]
        log.info("db.document_inserted", source=source, doc_id=str(doc_id))
        return doc_id

    # ── Chunks ────────────────────────────────────────────────────────────────

    def insert_chunks(
        self,
        doc_id: uuid.UUID,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """Batch-insert chunks with their embeddings.

        Args:
            doc_id: Parent document UUID.
            chunks: List of dicts with keys: content, chunk_index,
                    start_char, end_char, metadata.
            embeddings: Float32 array of shape (len(chunks), embedding_dim).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )

        rows = [
            (
                doc_id,
                c["content"].replace("\x00", ""),
                embeddings[i].tolist(),
                c["chunk_index"],
                c.get("start_char"),
                c.get("end_char"),
                psycopg2.extras.Json(c.get("metadata", {})),
            )
            for i, c in enumerate(chunks)
        ]

        with self._conn() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO chunks
                        (doc_id, content, embedding, chunk_index,
                         start_char, end_char, metadata)
                    VALUES %s
                    """,
                    rows,
                    template="(%s, %s, %s, %s, %s, %s, %s)",
                )
        log.info("db.chunks_inserted", doc_id=str(doc_id), n=len(chunks))

    # ── Retrieval (used from Phase 2 onward) ─────────────────────────────────

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return the top-k most similar chunks by cosine similarity.

        pgvector's <=> operator returns cosine *distance* (lower = more similar).
        We return similarity = 1 - distance so callers get an intuitive score.

        Args:
            query_embedding: Normalized 1-D float array.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: chunk_id, doc_id, content, source,
            similarity, metadata.
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        c.id          AS chunk_id,
                        c.doc_id,
                        c.content,
                        c.metadata    AS chunk_metadata,
                        d.source,
                        d.title,
                        1 - (c.embedding <=> %s::vector) AS similarity
                    FROM chunks c
                    JOIN documents d ON d.id = c.doc_id
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding.tolist(), query_embedding.tolist(), top_k),
                )
                return [dict(row) for row in cur.fetchall()]
