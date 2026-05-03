"""MCP tools: document search, indexing, health, read, diff, and memory."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from docs_memory_mcp.duckdb_store import DuckDBStore, normalize_topic, parse_metadata_json
from docs_memory_mcp.indexer import (
    collection_chunk_count,
    index_relative_paths,
    read_doc_file,
    run_full_index,
)
from docs_memory_mcp.ollama_embed import embed_documents
from docs_memory_mcp.settings_singleton import get_settings
from docs_memory_mcp.wiki_export import export_wiki

mcp = FastMCP(
    "docs-memory",
    instructions=(
        "Document and memory tools: docs_search, docs_read, docs_diff, docs_index_file, "
        "docs_reindex, docs_health, memory_upsert, memory_search, memory_profile_get, "
        "memory_profile_upsert, wiki_export, wiki_read. Paths are relative to DOCS_ROOT."
    ),
)

_READ_HARD_MAX = 500_000
_DIFF_MAX_LINES = 800


def _resolve_under_root(root: Path, relative_path: str) -> Path | None:
    rel = relative_path.strip()
    if not rel:
        return None
    root_r = root.resolve()
    try:
        fp = (root_r / rel).resolve()
    except OSError:
        return None
    try:
        fp.relative_to(root_r)
    except ValueError:
        return None
    if not fp.is_file():
        return None
    return fp


def _store() -> DuckDBStore:
    settings = get_settings()
    return DuckDBStore(settings.duckdb_path)


def _json_error(message: str) -> str:
    return json.dumps({"error": message})


def _resolve_output_dir(root: Path, relative_dir: str) -> Path | None:
    target = (root / relative_dir.strip()).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    return target


@mcp.tool()
def docs_search(query: str, top_k: int | None = None) -> str:
    """Search indexed documents by meaning. Returns snippets with source paths and chunk indices."""
    settings = get_settings()
    k = top_k if top_k is not None else settings.rag_top_k
    try:
        vecs = embed_documents([query], settings.ollama_host, settings.ollama_embed_model)
        if not vecs:
            return "Embedding failed (empty result)."
        store = _store()
        try:
            if store.doc_chunk_count(settings.collection_name) == 0:
                return "Index is empty. Run docs_reindex (Ollama must be running)."
            results = store.search_docs(settings.collection_name, vecs[0], k)
        finally:
            store.close()
    except Exception as exc:
        return f"Search error: {exc}"

    if not results:
        return "No results (empty collection or no matches)."

    lines: list[str] = []
    for i, result in enumerate(results):
        lines.append(
            f"--- rank {i + 1} | source={result.source} chunk={result.chunk} score={result.score:.4f} ---"
        )
        lines.append(result.content.strip())
        lines.append("")
    return "\n".join(lines).strip()


@mcp.tool()
def docs_read(relative_path: str, max_chars: int | None = None) -> str:
    """
    Return plain text extracted from a doc under the configured root.
    No index and no embeddings. Optional max_chars truncates output.
    """
    settings = get_settings()
    root = settings.docs_root.resolve()
    fp = _resolve_under_root(root, relative_path)
    if fp is None:
        return f"Not found or not a file under docs root: {relative_path!r}"
    text = read_doc_file(fp)
    if not text:
        return f"No extractable text: {relative_path!r}"
    cap = min(max_chars, _READ_HARD_MAX) if max_chars is not None else _READ_HARD_MAX
    if len(text) > cap:
        return text[:cap] + f"\n\n[truncated: {len(text)} chars total, showing first {cap}]"
    return text


@mcp.tool()
def docs_diff(relative_path_a: str, relative_path_b: str, max_diff_lines: int | None = None) -> str:
    """
    Line-oriented unified diff of extracted text from two files.
    Labels lines as A/B using the relative paths you pass.
    """
    settings = get_settings()
    root = settings.docs_root.resolve()
    fa = _resolve_under_root(root, relative_path_a)
    fb = _resolve_under_root(root, relative_path_b)
    if fa is None:
        return f"A: not found or not under docs root: {relative_path_a!r}"
    if fb is None:
        return f"B: not found or not under docs root: {relative_path_b!r}"
    ta = read_doc_file(fa)
    tb = read_doc_file(fb)
    if not ta:
        return f"A: no extractable text: {relative_path_a!r}"
    if not tb:
        return f"B: no extractable text: {relative_path_b!r}"
    lim = max_diff_lines if max_diff_lines is not None else _DIFF_MAX_LINES
    diff_iter = difflib.unified_diff(
        ta.splitlines(),
        tb.splitlines(),
        fromfile=relative_path_a.strip(),
        tofile=relative_path_b.strip(),
        lineterm="",
        n=3,
    )
    out_lines: list[str] = []
    for i, line in enumerate(diff_iter):
        if i >= lim:
            out_lines.append(f"\n... [diff truncated after {lim} lines; widen max_diff_lines if needed] ...")
            break
        out_lines.append(line)
    if not out_lines:
        return "No differences (extracted texts are identical line-by-line)."
    return "\n".join(out_lines)


@mcp.tool()
def docs_index_file(relative_path: str) -> str:
    """
    Incrementally index one or more files: removes old chunks for each path, then embeds only those files.
    Paths are relative to DOCS_ROOT and may be comma- or newline-separated.
    """
    settings = get_settings()
    root = settings.docs_root.resolve()
    if not root.is_dir():
        return f"DOCS_ROOT is not a directory: {root}"
    raw = relative_path.replace("\n", ",")
    paths = [p.strip() for p in raw.split(",") if p.strip()]

    def prog(msg: str, n: int) -> None:
        pass

    try:
        n_added, errs = index_relative_paths(
            docs_root=root,
            duckdb_path=settings.duckdb_path,
            collection_name=settings.collection_name,
            ollama_host=settings.ollama_host,
            embed_model=settings.ollama_embed_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            batch_size=settings.index_batch_size,
            relative_paths=paths,
            progress=prog,
        )
    except Exception as exc:
        return f"Incremental index failed: {exc}"
    err_s = ("; " + "; ".join(errs)) if errs else ""
    return f"OK: added {n_added} chunk(s) for {len(paths)} path(s).{err_s}"


@mcp.tool()
def docs_reindex() -> str:
    """Full rebuild of indexed documents for the active collection."""
    settings = get_settings()
    root = settings.docs_root.resolve()
    if not root.is_dir():
        return f"DOCS_ROOT is not a directory: {root}"

    def prog(msg: str, n: int) -> None:
        pass

    try:
        n = run_full_index(
            docs_root=root,
            duckdb_path=settings.duckdb_path,
            collection_name=settings.collection_name,
            ollama_host=settings.ollama_host,
            embed_model=settings.ollama_embed_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            batch_size=settings.index_batch_size,
            progress=prog,
        )
    except Exception as exc:
        return f"Reindex failed: {exc}"
    return f"OK: indexed {n} chunks into {settings.duckdb_path} (collection={settings.collection_name})."


@mcp.tool()
def docs_health() -> str:
    """Check Ollama reachability and DuckDB index size."""
    settings = get_settings()
    ollama_ok = False
    ollama_detail = ""
    try:
        host = settings.ollama_host.rstrip("/")
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{host}/api/tags")
            response.raise_for_status()
            ollama_ok = True
            tags = response.json().get("models") or []
            names = [model.get("name", "") for model in tags]
            embed_model = settings.ollama_embed_model
            present = any(name == embed_model or name.startswith(f"{embed_model}:") for name in names)
            ollama_detail = f"models={len(names)}; embed_model={embed_model!r} in_ollama={present}"
    except Exception as exc:
        ollama_detail = str(exc)

    try:
        store = _store()
        try:
            payload: dict[str, Any] = store.health(settings.collection_name)
        finally:
            store.close()
    except Exception as exc:
        payload = {
            "storage": "duckdb",
            "duckdb_path": str(settings.duckdb_path),
            "collection": settings.collection_name,
            "chunks_in_index": collection_chunk_count(settings.duckdb_path, settings.collection_name),
            "error": str(exc),
        }
    payload.update(
        {
            "ollama_ok": ollama_ok,
            "ollama": ollama_detail,
            "docs_root": str(settings.docs_root.resolve()),
        }
    )
    return json.dumps(payload, indent=2)


@mcp.tool()
def memory_upsert(
    text: str,
    memory_type: str,
    importance: float,
    intensity: float,
    confidence: float,
    topic: str = "",
    behavior_hint: str = "",
    metadata_json: str | None = None,
) -> str:
    """Store one active memory with an embedding and ranking metadata."""
    settings = get_settings()
    try:
        metadata = parse_metadata_json(metadata_json)
        topics = normalize_topic(topic)
        vecs = embed_documents([text], settings.ollama_host, settings.ollama_embed_model)
        if not vecs:
            return _json_error("embedding failed")
        store = _store()
        try:
            item_id = store.memory_upsert(
                collection=settings.collection_name,
                text=text,
                vector=vecs[0],
                memory_type=memory_type,
                topic=topics,
                importance=importance,
                intensity=intensity,
                confidence=confidence,
                behavior_hint=behavior_hint,
                metadata=metadata,
            )
        finally:
            store.close()
        return json.dumps({"status": "success", "id": item_id})
    except Exception as exc:
        return _json_error(str(exc))


@mcp.tool()
def memory_search(
    query: str,
    top_k: int | None = None,
    memory_type: str | None = None,
    min_importance: float | None = None,
) -> str:
    """Search active memories using semantic similarity plus importance, intensity, and confidence."""
    settings = get_settings()
    k = top_k if top_k is not None else settings.rag_top_k
    try:
        vecs = embed_documents([query], settings.ollama_host, settings.ollama_embed_model)
        if not vecs:
            return _json_error("embedding failed")
        store = _store()
        try:
            results = store.memory_search(
                collection=settings.collection_name,
                query_vector=vecs[0],
                limit=k,
                memory_type=memory_type,
                min_importance=min_importance,
            )
        finally:
            store.close()
        return json.dumps(
            {
                "results": [
                    {
                        "id": item.id,
                        "text": item.text,
                        "memory_type": item.memory_type,
                        "topic": item.topic,
                        "importance": item.importance,
                        "intensity": item.intensity,
                        "confidence": item.confidence,
                        "behavior_hint": item.behavior_hint,
                        "metadata": item.metadata,
                        "similarity": item.similarity,
                        "score": item.score,
                    }
                    for item in results
                ]
            },
            indent=2,
        )
    except Exception as exc:
        return _json_error(str(exc))


@mcp.tool()
def memory_profile_get(profile_key: str) -> str:
    """Return one compiled memory profile by key."""
    settings = get_settings()
    try:
        store = _store()
        try:
            profile = store.profile_get(settings.collection_name, profile_key)
        finally:
            store.close()
        if profile is None:
            return _json_error(f"profile not found: {profile_key}")
        return json.dumps(profile, indent=2)
    except Exception as exc:
        return _json_error(str(exc))


@mcp.tool()
def memory_profile_upsert(
    profile_key: str,
    compiled_summary: str,
    behavior_profile: str,
    metadata_json: str | None = None,
) -> str:
    """Create or replace one compiled profile in the same DuckDB file."""
    settings = get_settings()
    try:
        metadata = parse_metadata_json(metadata_json)
        store = _store()
        try:
            store.profile_upsert(
                settings.collection_name,
                profile_key,
                compiled_summary,
                behavior_profile,
                metadata,
            )
        finally:
            store.close()
        return json.dumps({"status": "success", "profile_key": profile_key})
    except Exception as exc:
        return _json_error(str(exc))


@mcp.tool()
def wiki_export(output_dir: str = "wiki", memory_limit: int = 100) -> str:
    """Generate a readable markdown wiki view from the current DuckDB collection."""
    settings = get_settings()
    root = settings.docs_root.resolve()
    target = _resolve_output_dir(root, output_dir)
    if target is None:
        return _json_error(f"output_dir must stay under DOCS_ROOT: {output_dir!r}")
    try:
        store = _store()
        try:
            result = export_wiki(store, settings.collection_name, target, memory_limit=memory_limit)
        finally:
            store.close()
        return json.dumps(result, indent=2)
    except Exception as exc:
        return _json_error(str(exc))


@mcp.tool()
def wiki_read(relative_path: str = "wiki/index.md", max_chars: int | None = None) -> str:
    """Read a generated wiki markdown file under DOCS_ROOT."""
    settings = get_settings()
    root = settings.docs_root.resolve()
    fp = _resolve_under_root(root, relative_path)
    if fp is None:
        return f"Not found or not a file under docs root: {relative_path!r}"
    text = fp.read_text(encoding="utf-8", errors="replace")
    cap = min(max_chars, _READ_HARD_MAX) if max_chars is not None else _READ_HARD_MAX
    if len(text) > cap:
        return text[:cap] + f"\n\n[truncated: {len(text)} chars total, showing first {cap}]"
    return text
