"""SQLite persistence for RAG documents, chunks, and embedding vectors."""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SQLiteVectorRepository:
    """Store RAG vectors and their source text in a local SQLite database."""

    def __init__(self, database_path: str):
        self.database_path = database_path
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        if database_path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    document_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    PRIMARY KEY (document_id, ordinal),
                    FOREIGN KEY (document_id) REFERENCES documents(document_id)
                        ON DELETE CASCADE
                );
                """
            )

    def replace_document(
        self,
        document_id: str,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Atomically replace a document and all of its indexed chunks."""
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts) or len(texts) != len(metadata):
            raise ValueError("Embeddings, texts, and metadata must have matching rows.")

        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO documents (document_id) VALUES (?)
                ON CONFLICT(document_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                """,
                (document_id,),
            )
            self._connection.execute(
                "DELETE FROM chunks WHERE document_id = ?", (document_id,)
            )
            self._connection.executemany(
                """
                INSERT INTO chunks (
                    document_id, ordinal, text, metadata_json, embedding, dimension
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        ordinal,
                        text,
                        json.dumps(meta),
                        vector.tobytes(),
                        int(vector.size),
                    )
                    for ordinal, (text, meta, vector) in enumerate(
                        zip(texts, metadata, vectors)
                    )
                ],
            )

    def load_all_documents(
        self,
    ) -> List[Tuple[str, List[str], List[Dict[str, Any]], np.ndarray]]:
        """Load each document's texts, metadata, and embedding matrix."""
        with self._lock:
            document_ids = [
                row["document_id"]
                for row in self._connection.execute(
                    "SELECT document_id FROM documents ORDER BY document_id"
                )
            ]
            loaded = []
            for document_id in document_ids:
                rows = self._connection.execute(
                    """
                    SELECT text, metadata_json, embedding, dimension
                    FROM chunks WHERE document_id = ? ORDER BY ordinal
                    """,
                    (document_id,),
                ).fetchall()
                if not rows:
                    continue
                dimensions = {row["dimension"] for row in rows}
                if len(dimensions) != 1:
                    raise ValueError(
                        f"Document '{document_id}' has inconsistent embedding dimensions."
                    )
                dimension = dimensions.pop()
                texts = [row["text"] for row in rows]
                metadata = [json.loads(row["metadata_json"]) for row in rows]
                vectors = np.vstack(
                    [
                        np.frombuffer(row["embedding"], dtype=np.float32)
                        for row in rows
                    ]
                ).reshape(len(rows), dimension)
                loaded.append((document_id, texts, metadata, vectors))
            return loaded

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return persisted document IDs and chunk counts."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT documents.document_id, COUNT(chunks.ordinal) AS chunks_indexed
                FROM documents
                LEFT JOIN chunks ON chunks.document_id = documents.document_id
                GROUP BY documents.document_id
                ORDER BY documents.document_id
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks, returning whether it existed."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Delete all persisted documents and chunks."""
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM documents")

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._connection.close()
