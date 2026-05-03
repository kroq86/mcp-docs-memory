from __future__ import annotations

from pathlib import Path

from docs_memory_mcp.duckdb_store import DuckDBStore


def test_doc_chunk_insert_search_and_delete(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "test.duckdb")
    try:
        added = store.upsert_doc_chunks(
            "test_collection",
            "docs/example.md",
            [
                (0, "alpha memory architecture", {"source": "docs/example.md", "chunk": 0}),
                (1, "beta runtime search", {"source": "docs/example.md", "chunk": 1}),
            ],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        assert added == 2
        assert store.doc_chunk_count("test_collection") == 2

        results = store.search_docs("test_collection", [1.0, 0.0], 1)
        assert len(results) == 1
        assert results[0].source == "docs/example.md"
        assert results[0].chunk == 0

        deleted = store.delete_doc_source("test_collection", "docs/example.md")
        assert deleted == 2
        assert store.doc_chunk_count("test_collection") == 0
    finally:
        store.close()


def test_reset_docs_collection_preserves_memories(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "test.duckdb")
    try:
        store.upsert_doc_chunks(
            "test_collection",
            "docs/example.md",
            [(0, "doc text", {})],
            [[1.0, 0.0]],
        )
        store.memory_upsert(
            collection="test_collection",
            text="important memory",
            vector=[1.0, 0.0],
            memory_type="decision",
            topic=["memory"],
            importance=0.9,
            intensity=0.8,
            confidence=1.0,
            behavior_hint="be concrete",
            metadata={},
        )

        store.reset_docs_collection("test_collection")
        assert store.doc_chunk_count("test_collection") == 0
        assert len(store.memory_search("test_collection", [1.0, 0.0], 5)) == 1
    finally:
        store.close()


def test_memory_search_uses_metadata_score(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "test.duckdb")
    try:
        low_id = store.memory_upsert(
            collection="test_collection",
            text="similar but low value",
            vector=[1.0, 0.0],
            memory_type="note",
            topic=["search"],
            importance=0.0,
            intensity=0.0,
            confidence=0.0,
            behavior_hint="",
            metadata={},
        )
        high_id = store.memory_upsert(
            collection="test_collection",
            text="slightly less similar but important",
            vector=[0.9, 0.1],
            memory_type="note",
            topic=["search"],
            importance=1.0,
            intensity=1.0,
            confidence=1.0,
            behavior_hint="use this first",
            metadata={},
        )

        results = store.memory_search("test_collection", [1.0, 0.0], 2)
        assert [item.id for item in results] == [high_id, low_id]
    finally:
        store.close()


def test_profile_upsert_get(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "test.duckdb")
    try:
        store.profile_upsert(
            "test_collection",
            "default",
            "compiled summary",
            "behavior profile",
            {"version": 1},
        )
        profile = store.profile_get("test_collection", "default")
        assert profile is not None
        assert profile["profile_key"] == "default"
        assert profile["compiled_summary"] == "compiled summary"
        assert profile["behavior_profile"] == "behavior profile"
        assert profile["metadata"] == {"version": 1}
    finally:
        store.close()
