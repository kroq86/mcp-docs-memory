from __future__ import annotations

from pathlib import Path

from docs_memory_mcp.duckdb_store import DuckDBStore
from docs_memory_mcp.wiki_export import export_wiki


def test_export_wiki_writes_readable_pages(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "test.duckdb")
    try:
        store.upsert_doc_chunks(
            "test_collection",
            "docs/example.md",
            [(0, "example docs", {"source": "docs/example.md", "chunk": 0})],
            [[1.0, 0.0]],
        )
        store.memory_upsert(
            collection="test_collection",
            text="Important repo decision",
            vector=[1.0, 0.0],
            memory_type="decision",
            topic=["repo"],
            importance=0.9,
            intensity=0.2,
            confidence=1.0,
            behavior_hint="Use this decision in future answers.",
            metadata={},
        )
        store.profile_upsert(
            "test_collection",
            "default",
            "Compiled summary",
            "Behavior profile",
            {},
        )

        result = export_wiki(store, "test_collection", tmp_path / "wiki")
        assert result["sources"] == 1
        assert result["memories"] == 1
        assert result["profiles"] == 1

        index = (tmp_path / "wiki" / "index.md").read_text(encoding="utf-8")
        memories = (tmp_path / "wiki" / "memories.md").read_text(encoding="utf-8")
        sources = (tmp_path / "wiki" / "sources.md").read_text(encoding="utf-8")
        profiles = (tmp_path / "wiki" / "profiles.md").read_text(encoding="utf-8")

        assert "Docs Memory Index" in index
        assert "Important repo decision" in memories
        assert "docs/example.md" in sources
        assert "Compiled summary" in profiles
    finally:
        store.close()
