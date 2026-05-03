from __future__ import annotations

from pathlib import Path

from docs_memory_mcp.duckdb_store import DuckDBStore
from docs_memory_mcp import indexer


def test_incremental_index_replaces_stale_chunks(tmp_path: Path, monkeypatch) -> None:
    docs_root = tmp_path / "workspace"
    docs_dir = docs_root / "docs"
    docs_dir.mkdir(parents=True)
    target = docs_dir / "example.md"
    target.write_text("alpha first version", encoding="utf-8")

    def fake_embed(texts, host, model):
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]

    monkeypatch.setattr(indexer, "embed_documents", fake_embed)
    db_path = tmp_path / "index.duckdb"

    added, errors = indexer.index_relative_paths(
        docs_root=docs_root,
        duckdb_path=db_path,
        collection_name="test_collection",
        ollama_host="http://example.invalid",
        embed_model="test",
        chunk_size=100,
        chunk_overlap=0,
        batch_size=2,
        relative_paths=["docs/example.md"],
    )
    assert added == 1
    assert errors == []

    target.write_text("beta second version", encoding="utf-8")
    added, errors = indexer.index_relative_paths(
        docs_root=docs_root,
        duckdb_path=db_path,
        collection_name="test_collection",
        ollama_host="http://example.invalid",
        embed_model="test",
        chunk_size=100,
        chunk_overlap=0,
        batch_size=2,
        relative_paths=["docs/example.md"],
    )
    assert added == 1
    assert errors == []

    store = DuckDBStore(db_path)
    try:
        assert store.doc_chunk_count("test_collection") == 1
        results = store.search_docs("test_collection", [0.0, 1.0], 1)
        assert results[0].content == "beta second version"
    finally:
        store.close()


def test_full_reindex_removes_stale_sources(tmp_path: Path, monkeypatch) -> None:
    docs_root = tmp_path / "workspace"
    docs_dir = docs_root / "docs"
    docs_dir.mkdir(parents=True)
    first = docs_dir / "first.md"
    second = docs_dir / "second.md"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    def fake_embed(texts, host, model):
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]

    monkeypatch.setattr(indexer, "embed_documents", fake_embed)
    db_path = tmp_path / "index.duckdb"

    count = indexer.run_full_index(
        docs_root=docs_root,
        duckdb_path=db_path,
        collection_name="test_collection",
        ollama_host="http://example.invalid",
        embed_model="test",
        chunk_size=100,
        chunk_overlap=0,
        batch_size=2,
    )
    assert count == 2

    second.unlink()
    count = indexer.run_full_index(
        docs_root=docs_root,
        duckdb_path=db_path,
        collection_name="test_collection",
        ollama_host="http://example.invalid",
        embed_model="test",
        chunk_size=100,
        chunk_overlap=0,
        batch_size=2,
    )
    assert count == 1

    store = DuckDBStore(db_path)
    try:
        assert store.doc_chunk_count("test_collection") == 1
        results = store.search_docs("test_collection", [1.0, 0.0], 1)
        assert results[0].source == "docs/first.md"
    finally:
        store.close()
