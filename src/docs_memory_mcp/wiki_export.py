"""Generate a readable markdown wiki view from the DuckDB store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docs_memory_mcp.duckdb_store import DuckDBStore


def _frontmatter(title: str, page_type: str) -> str:
    return f"---\ntitle: {title}\ntype: {page_type}\ngenerated: true\n---\n\n"


def _safe_line(text: Any) -> str:
    return str(text).replace("\n", " ").strip()


def _format_topics(topics: list[str]) -> str:
    return ", ".join(topics) if topics else "none"


def export_wiki(
    store: DuckDBStore,
    collection: str,
    output_dir: Path,
    memory_limit: int = 100,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    health = store.health(collection)
    sources = store.doc_sources(collection)
    memories = store.memory_items(collection, limit=memory_limit)
    profiles = store.profiles(collection)

    files: list[Path] = []
    files.append(_write_index(output_dir, health, sources, memories, profiles))
    files.append(_write_sources(output_dir, sources))
    files.append(_write_memories(output_dir, memories))
    files.append(_write_profiles(output_dir, profiles))

    return {
        "status": "success",
        "output_dir": str(output_dir),
        "files": [str(path) for path in files],
        "sources": len(sources),
        "memories": len(memories),
        "profiles": len(profiles),
    }


def _write_index(
    output_dir: Path,
    health: dict[str, Any],
    sources: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
) -> Path:
    path = output_dir / "index.md"
    lines = [
        _frontmatter("Docs Memory Index", "index"),
        "# Docs Memory Index",
        "",
        "## Status",
        "",
        f"- Storage: `{health.get('storage')}`",
        f"- Collection: `{health.get('collection')}`",
        f"- Document chunks: `{health.get('chunks_in_index')}`",
        f"- Active memories: `{health.get('active_memories')}`",
        f"- Vectors: `{health.get('vectors')}`",
        "",
        "## Pages",
        "",
        "- [[sources]] - indexed document sources",
        "- [[memories]] - ranked durable memories",
        "- [[profiles]] - compiled behavior/context profiles",
        "",
        "## Top Memories",
        "",
    ]
    if memories:
        for item in memories[:10]:
            lines.append(
                f"- `{item['memory_type']}` importance={item['importance']:.2f}: {_safe_line(item['text'])}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Source Count", "", f"- Indexed sources: `{len(sources)}`", f"- Profiles: `{len(profiles)}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_sources(output_dir: Path, sources: list[dict[str, Any]]) -> Path:
    path = output_dir / "sources.md"
    lines = [_frontmatter("Sources", "sources"), "# Sources", ""]
    if not sources:
        lines.append("No indexed sources.")
    for source in sources:
        lines.append(f"- `{source['source']}` - chunks={source['chunks']}, updated={source['updated_at']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_memories(output_dir: Path, memories: list[dict[str, Any]]) -> Path:
    path = output_dir / "memories.md"
    lines = [_frontmatter("Memories", "memories"), "# Memories", ""]
    if not memories:
        lines.append("No stored memories.")
    for item in memories:
        lines.extend(
            [
                f"## {item['id']}",
                "",
                f"- Type: `{item['memory_type']}`",
                f"- Status: `{item['status']}`",
                f"- Topic: {_format_topics(item['topic'])}",
                f"- Importance: `{item['importance']:.2f}`",
                f"- Intensity: `{item['intensity']:.2f}`",
                f"- Confidence: `{item['confidence']:.2f}`",
                f"- Use count: `{item['use_count']}`",
                "",
                item["text"],
                "",
            ]
        )
        if item["behavior_hint"]:
            lines.extend(["Behavior hint:", "", f"> {item['behavior_hint']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_profiles(output_dir: Path, profiles: list[dict[str, Any]]) -> Path:
    path = output_dir / "profiles.md"
    lines = [_frontmatter("Profiles", "profiles"), "# Profiles", ""]
    if not profiles:
        lines.append("No compiled profiles.")
    for profile in profiles:
        lines.extend(
            [
                f"## {profile['profile_key']}",
                "",
                f"- Updated: `{profile['updated_at']}`",
                "",
                "### Compiled Summary",
                "",
                profile["compiled_summary"],
                "",
                "### Behavior Profile",
                "",
                profile["behavior_profile"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
