"""Embedded DuckDB storage for document chunks, vectors, and memory records."""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import psutil


@dataclass(frozen=True)
class DocSearchResult:
    id: str
    source: str
    chunk: int
    content: str
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class MemorySearchResult:
    id: str
    text: str
    memory_type: str
    topic: list[str]
    importance: float
    intensity: float
    confidence: float
    behavior_hint: str
    metadata: dict[str, Any]
    similarity: float
    score: float


class DuckDBStore:
    """Small embedded vector store adapted for this MCP server's document and memory flows."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._initialize_db()

    def _initialize_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id VARCHAR PRIMARY KEY,
                collection VARCHAR NOT NULL,
                vector BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                partition_id INTEGER NOT NULL,
                kind VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_collection_kind ON vectors(collection, kind)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_partition ON vectors(partition_id)")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS doc_chunks (
                id VARCHAR PRIMARY KEY,
                collection VARCHAR NOT NULL,
                source VARCHAR NOT NULL,
                chunk INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata JSON,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_collection_source ON doc_chunks(collection, source)")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_items (
                id VARCHAR PRIMARY KEY,
                collection VARCHAR NOT NULL,
                text TEXT NOT NULL,
                memory_type VARCHAR NOT NULL,
                topic JSON,
                importance DOUBLE NOT NULL,
                intensity DOUBLE NOT NULL,
                confidence DOUBLE NOT NULL,
                behavior_hint TEXT NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'active',
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP,
                use_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_collection_type ON memory_items(collection, memory_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_status ON memory_items(status)")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_profiles (
                id VARCHAR PRIMARY KEY,
                collection VARCHAR NOT NULL,
                profile_key VARCHAR NOT NULL,
                compiled_summary TEXT NOT NULL,
                behavior_profile TEXT NOT NULL,
                metadata JSON,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(collection, profile_key)
            )
            """
        )

    @staticmethod
    def _serialize_vector(vector: list[float]) -> bytes:
        return np.asarray(vector, dtype=np.float64).tobytes()

    @staticmethod
    def _deserialize_vector(data: bytes, dimensions: int) -> np.ndarray:
        vector = np.frombuffer(data, dtype=np.float64)
        if len(vector) != dimensions:
            return vector[:dimensions]
        return vector

    @staticmethod
    def _partition_id(key: str, partition_count: int = 1000) -> int:
        return abs(hash(key)) % partition_count

    @staticmethod
    def _json_loads(value: str | None, fallback: Any) -> Any:
        if not value:
            return fallback
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback

    @staticmethod
    def _bounded_score(value: float) -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return max(0.0, min(1.0, float(value)))

    def close(self) -> None:
        self.conn.close()

    def reset_docs_collection(self, collection: str) -> None:
        self.conn.execute(
            "DELETE FROM vectors WHERE collection = ? AND kind = 'doc'",
            [collection],
        )
        self.conn.execute("DELETE FROM doc_chunks WHERE collection = ?", [collection])

    def delete_doc_source(self, collection: str, source: str) -> int:
        rows = self.conn.execute(
            "SELECT id FROM doc_chunks WHERE collection = ? AND source = ?",
            [collection, source],
        ).fetchall()
        ids = [row[0] for row in rows]
        if not ids:
            return 0
        placeholders = ",".join(["?"] * len(ids))
        self.conn.execute(f"DELETE FROM vectors WHERE id IN ({placeholders})", ids)
        self.conn.execute(
            "DELETE FROM doc_chunks WHERE collection = ? AND source = ?",
            [collection, source],
        )
        return len(ids)

    def upsert_doc_chunks(
        self,
        collection: str,
        source: str,
        chunks: list[tuple[int, str, dict[str, Any]]],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.delete_doc_source(collection, source)
        if not chunks:
            return 0

        vector_rows = []
        chunk_rows = []
        for (chunk_index, content, metadata), vector in zip(chunks, embeddings):
            item_id = f"doc:{collection}:{source}:{chunk_index}"
            vector_rows.append(
                (
                    item_id,
                    collection,
                    self._serialize_vector(vector),
                    len(vector),
                    self._partition_id(item_id),
                    "doc",
                )
            )
            chunk_rows.append(
                (
                    item_id,
                    collection,
                    source,
                    chunk_index,
                    content,
                    json.dumps(metadata),
                )
            )

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO vectors (id, collection, vector, dimensions, partition_id, kind)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            vector_rows,
        )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO doc_chunks (id, collection, source, chunk, content, metadata, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            chunk_rows,
        )
        return len(chunks)

    def _search_vectors(self, collection: str, kind: str, query_vector: list[float], limit: int) -> list[tuple[str, float]]:
        if limit <= 0:
            return []
        rows = self.conn.execute(
            """
            SELECT id, vector, dimensions
            FROM vectors
            WHERE collection = ? AND kind = ? AND dimensions = ?
            """,
            [collection, kind, len(query_vector)],
        ).fetchall()
        if not rows:
            return []

        ids = [row[0] for row in rows]
        matrix = np.vstack([self._deserialize_vector(row[1], row[2]) for row in rows])
        query = np.asarray(query_vector, dtype=np.float64)
        query_norm = np.linalg.norm(query)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        denom = matrix_norms * query_norm
        dots = matrix @ query
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.divide(dots, denom, out=np.zeros_like(dots), where=denom != 0)

        take = min(limit, len(ids))
        if take == len(ids):
            order = np.argsort(scores)[::-1]
        else:
            candidates = np.argpartition(scores, -take)[-take:]
            order = candidates[np.argsort(scores[candidates])[::-1]]
        return [(ids[int(i)], float(scores[int(i)])) for i in order]

    def search_docs(self, collection: str, query_vector: list[float], limit: int) -> list[DocSearchResult]:
        matches = self._search_vectors(collection, "doc", query_vector, limit)
        results: list[DocSearchResult] = []
        for item_id, score in matches:
            row = self.conn.execute(
                """
                SELECT id, source, chunk, content, metadata
                FROM doc_chunks
                WHERE id = ? AND collection = ?
                """,
                [item_id, collection],
            ).fetchone()
            if row is None:
                continue
            results.append(
                DocSearchResult(
                    id=row[0],
                    source=row[1],
                    chunk=int(row[2]),
                    content=row[3],
                    metadata=self._json_loads(row[4], {}),
                    score=score,
                )
            )
        return results

    def doc_chunk_count(self, collection: str) -> int:
        return int(
            self.conn.execute(
                "SELECT COUNT(*) FROM doc_chunks WHERE collection = ?",
                [collection],
            ).fetchone()[0]
        )

    def doc_sources(self, collection: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT source, COUNT(*) AS chunks, MAX(updated_at) AS updated_at
            FROM doc_chunks
            WHERE collection = ?
            GROUP BY source
            ORDER BY source
            """,
            [collection],
        ).fetchall()
        return [
            {
                "source": row[0],
                "chunks": int(row[1]),
                "updated_at": str(row[2]),
            }
            for row in rows
        ]

    def vector_dimensions(self, collection: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT DISTINCT dimensions FROM vectors WHERE collection = ? ORDER BY dimensions",
            [collection],
        ).fetchall()
        return [int(row[0]) for row in rows]

    def memory_upsert(
        self,
        collection: str,
        text: str,
        vector: list[float],
        memory_type: str,
        topic: list[str],
        importance: float,
        intensity: float,
        confidence: float,
        behavior_hint: str,
        metadata: dict[str, Any],
        memory_id: str | None = None,
    ) -> str:
        item_id = memory_id or f"mem:{uuid.uuid4().hex}"
        self.conn.execute(
            """
            INSERT OR REPLACE INTO vectors (id, collection, vector, dimensions, partition_id, kind)
            VALUES (?, ?, ?, ?, ?, 'memory')
            """,
            [
                item_id,
                collection,
                self._serialize_vector(vector),
                len(vector),
                self._partition_id(item_id),
            ],
        )
        self.conn.execute(
            """
            INSERT OR REPLACE INTO memory_items (
                id, collection, text, memory_type, topic, importance, intensity, confidence,
                behavior_hint, status, metadata, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, CURRENT_TIMESTAMP)
            """,
            [
                item_id,
                collection,
                text,
                memory_type,
                json.dumps(topic),
                self._bounded_score(importance),
                self._bounded_score(intensity),
                self._bounded_score(confidence),
                behavior_hint,
                json.dumps(metadata),
            ],
        )
        return item_id

    def memory_search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int,
        memory_type: str | None = None,
        min_importance: float | None = None,
    ) -> list[MemorySearchResult]:
        raw = self._search_vectors(collection, "memory", query_vector, max(limit * 8, limit))
        if not raw:
            return []
        similarity_by_id = dict(raw)
        ids = list(similarity_by_id)
        placeholders = ",".join(["?"] * len(ids))
        params: list[Any] = [collection, *ids]
        filters = ["collection = ?", f"id IN ({placeholders})", "status = 'active'"]
        if memory_type:
            filters.append("memory_type = ?")
            params.append(memory_type)
        if min_importance is not None:
            filters.append("importance >= ?")
            params.append(self._bounded_score(min_importance))

        rows = self.conn.execute(
            f"""
            SELECT id, text, memory_type, topic, importance, intensity, confidence, behavior_hint, metadata
            FROM memory_items
            WHERE {" AND ".join(filters)}
            """,
            params,
        ).fetchall()

        results: list[MemorySearchResult] = []
        for row in rows:
            similarity = similarity_by_id.get(row[0], 0.0)
            importance = self._bounded_score(row[4])
            intensity = self._bounded_score(row[5])
            confidence = self._bounded_score(row[6])
            score = (similarity * 0.65) + (importance * 0.20) + (intensity * 0.10) + (confidence * 0.05)
            results.append(
                MemorySearchResult(
                    id=row[0],
                    text=row[1],
                    memory_type=row[2],
                    topic=self._json_loads(row[3], []),
                    importance=importance,
                    intensity=intensity,
                    confidence=confidence,
                    behavior_hint=row[7],
                    metadata=self._json_loads(row[8], {}),
                    similarity=similarity,
                    score=score,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        selected = results[:limit]
        for item in selected:
            self.conn.execute(
                """
                UPDATE memory_items
                SET last_used_at = CURRENT_TIMESTAMP, use_count = use_count + 1
                WHERE id = ?
                """,
                [item.id],
            )
        return selected

    def profile_get(self, collection: str, profile_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT profile_key, compiled_summary, behavior_profile, metadata, updated_at
            FROM memory_profiles
            WHERE collection = ? AND profile_key = ?
            """,
            [collection, profile_key],
        ).fetchone()
        if row is None:
            return None
        return {
            "profile_key": row[0],
            "compiled_summary": row[1],
            "behavior_profile": row[2],
            "metadata": self._json_loads(row[3], {}),
            "updated_at": str(row[4]),
        }

    def memory_items(self, collection: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT id, text, memory_type, topic, importance, intensity, confidence,
                   behavior_hint, status, metadata, created_at, updated_at, last_used_at, use_count
            FROM memory_items
            WHERE collection = ?
            ORDER BY importance DESC, intensity DESC, updated_at DESC
            LIMIT ?
            """,
            [collection, limit],
        ).fetchall()
        return [
            {
                "id": row[0],
                "text": row[1],
                "memory_type": row[2],
                "topic": self._json_loads(row[3], []),
                "importance": self._bounded_score(row[4]),
                "intensity": self._bounded_score(row[5]),
                "confidence": self._bounded_score(row[6]),
                "behavior_hint": row[7],
                "status": row[8],
                "metadata": self._json_loads(row[9], {}),
                "created_at": str(row[10]),
                "updated_at": str(row[11]),
                "last_used_at": str(row[12]) if row[12] is not None else None,
                "use_count": int(row[13]),
            }
            for row in rows
        ]

    def profiles(self, collection: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT profile_key, compiled_summary, behavior_profile, metadata, updated_at
            FROM memory_profiles
            WHERE collection = ?
            ORDER BY profile_key
            """,
            [collection],
        ).fetchall()
        return [
            {
                "profile_key": row[0],
                "compiled_summary": row[1],
                "behavior_profile": row[2],
                "metadata": self._json_loads(row[3], {}),
                "updated_at": str(row[4]),
            }
            for row in rows
        ]

    def profile_upsert(
        self,
        collection: str,
        profile_key: str,
        compiled_summary: str,
        behavior_profile: str,
        metadata: dict[str, Any],
    ) -> None:
        item_id = f"profile:{collection}:{profile_key}"
        self.conn.execute(
            "DELETE FROM memory_profiles WHERE collection = ? AND profile_key = ?",
            [collection, profile_key],
        )
        self.conn.execute(
            """
            INSERT INTO memory_profiles (
                id, collection, profile_key, compiled_summary, behavior_profile, metadata, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [item_id, collection, profile_key, compiled_summary, behavior_profile, json.dumps(metadata)],
        )

    def health(self, collection: str) -> dict[str, Any]:
        process = psutil.Process()
        vectors = self.conn.execute(
            "SELECT COUNT(*) FROM vectors WHERE collection = ?",
            [collection],
        ).fetchone()[0]
        memories = self.conn.execute(
            "SELECT COUNT(*) FROM memory_items WHERE collection = ? AND status = 'active'",
            [collection],
        ).fetchone()[0]
        return {
            "storage": "duckdb",
            "duckdb_path": str(self.db_path),
            "collection": collection,
            "chunks_in_index": self.doc_chunk_count(collection),
            "active_memories": int(memories),
            "vectors": int(vectors),
            "vector_dimensions": self.vector_dimensions(collection),
            "memory_usage_mb": round(process.memory_info().rss / 1024 / 1024, 3),
        }


def normalize_topic(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = raw.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = re.split(r"[,;\n]", text)
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


def parse_metadata_json(raw: str | None) -> dict[str, Any]:
    if raw is None or not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("metadata_json must decode to an object")
    return parsed
