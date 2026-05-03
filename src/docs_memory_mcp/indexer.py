"""Discover document files, extract text, chunk, embed, and store in DuckDB."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from docs_memory_mcp.duckdb_store import DuckDBStore
from docs_memory_mcp.ollama_embed import embed_documents, reset_embed_endpoint_cache

_SKIP_DIR_PARTS = frozenset(
    {
        "node_modules",
        ".git",
        "__pycache__",
        "dist",
        "build",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "mcp-data",
    }
)


def _path_has_skip_part(path: Path) -> bool:
    return any(part in _SKIP_DIR_PARTS for part in path.parts)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    n = len(text)
    step = max(chunk_size - overlap, 1)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start += step
    return chunks


def html_to_plain(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(src=True):
        src = tag.get("src") or ""
        if isinstance(src, str) and src.strip().lower().startswith("data:"):
            tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _line_looks_like_base64(s: str) -> bool:
    if len(s) < 32:
        return False
    if s.count(" ") > max(2, len(s) // 40):
        return False
    alphabet = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    ok = sum(1 for c in s if c in alphabet)
    return ok >= len(s) * 0.92


def scrub_embedded_binary_noise(text: str) -> str:
    lines_out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            lines_out.append(line)
            continue
        if len(s) > 100 and s.startswith("%PDF"):
            continue
        if _line_looks_like_base64(s):
            continue
        if len(s) > 120:
            spaces = s.count(" ")
            if spaces < max(3, len(s) // 80):
                alnum_like = sum(1 for c in s if c.isalnum() or c in "+/=\n\r")
                if alnum_like > len(s) * 0.88:
                    continue
        if "endstream" in s and len(s) > 60 and s.count(" ") < 5:
            continue
        if re.match(r"^[0-9]+\s+\d+\s+obj", s):
            continue
        lines_out.append(line)
    return "\n".join(lines_out).strip()


def read_pdf_file(path: Path) -> str | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        reader = PdfReader(str(path), strict=False)
    except Exception:
        return None
    parts: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text()
        except Exception:
            text = ""
        if text and text.strip():
            parts.append(text.strip())
    text = "\n\n".join(parts).strip()
    if not text:
        return None
    out = scrub_embedded_binary_noise(text)
    return out or None


def read_doc_file(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_file(path)
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if suffix in {".md", ".py", ".toml", ".txt", ".json", ".yaml", ".yml"}:
        out = scrub_embedded_binary_noise(raw.strip())
        return out or None
    if suffix == ".doc":
        plain = scrub_embedded_binary_noise(html_to_plain(raw))
        return plain or None
    return None


def discover_doc_paths(docs_root: Path) -> list[Path]:
    root = docs_root.resolve()
    if not root.is_dir():
        return []

    candidates: list[Path] = []
    for root_file in ["README.md", "AGENTS.md", "pyproject.toml", "struct.md"]:
        candidate = root / root_file
        if candidate.is_file():
            candidates.append(candidate)

    globs = [
        "*.md",
        "docs/**/*.md",
        "docs/**/*.doc",
        "docs/**/*.pdf",
        "src/**/*.py",
        "tests/**/*.py",
        "wiki/**/*.md",
        "core/**/*.md",
        "core/docs/**/*.doc",
        "core/docs/**/*.pdf",
        "etl/**/*.md",
        "frontend/**/*.md",
        "frontend/docs/**/*.md",
        "mcp/**/*.md",
    ]
    for pattern in globs:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            if _path_has_skip_part(path.relative_to(root)):
                continue
            candidates.append(path)

    seen: set[Path] = set()
    out: list[Path] = []
    for path in sorted(candidates, key=lambda item: str(item).lower()):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _relative_to_root(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def index_relative_paths(
    docs_root: Path,
    duckdb_path: Path,
    collection_name: str,
    ollama_host: str,
    embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    relative_paths: list[str],
    progress: Callable[[str, int], None] | None = None,
) -> tuple[int, list[str]]:
    reset_embed_endpoint_cache()
    root = docs_root.resolve()
    store = DuckDBStore(duckdb_path)
    total_added = 0
    errors: list[str] = []

    try:
        for raw in relative_paths:
            rel_raw = raw.strip()
            if not rel_raw:
                continue
            path = (root / rel_raw).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                errors.append(f"not under docs root: {rel_raw!r}")
                continue
            if not path.is_file():
                errors.append(f"not a file: {rel_raw!r}")
                continue

            rel = _relative_to_root(path, root)
            text = read_doc_file(path)
            store.delete_doc_source(collection_name, rel)
            if not text:
                errors.append(f"no extractable text: {rel!r}")
                if progress:
                    progress(f"skipped empty: {rel}", total_added)
                continue

            pieces = chunk_text(text, chunk_size, chunk_overlap)
            if not pieces:
                continue
            if progress:
                progress(f"indexing: {rel}", total_added)
            chunks = [(i, piece, {"source": rel, "chunk": i}) for i, piece in enumerate(pieces)]
            added = _embed_and_store_chunks(
                store,
                collection_name,
                rel,
                chunks,
                ollama_host,
                embed_model,
                batch_size,
            )
            total_added += added
            if progress:
                progress(f"indexed {total_added} chunks", total_added)
    finally:
        store.close()

    return total_added, errors


def run_full_index(
    docs_root: Path,
    duckdb_path: Path,
    collection_name: str,
    ollama_host: str,
    embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    progress: Callable[[str, int], None] | None = None,
) -> int:
    reset_embed_endpoint_cache()
    root = docs_root.resolve()
    store = DuckDBStore(duckdb_path)
    total_chunks = 0
    try:
        store.reset_docs_collection(collection_name)
        for path in discover_doc_paths(root):
            rel = _relative_to_root(path, root)
            text = read_doc_file(path)
            if not text:
                continue
            pieces = chunk_text(text, chunk_size, chunk_overlap)
            chunks = [(i, piece, {"source": rel, "chunk": i}) for i, piece in enumerate(pieces)]
            total_chunks += _embed_and_store_chunks(
                store,
                collection_name,
                rel,
                chunks,
                ollama_host,
                embed_model,
                batch_size,
            )
            if progress:
                progress(f"indexed {total_chunks} chunks", total_chunks)
    finally:
        store.close()
    return total_chunks


def _embed_and_store_chunks(
    store: DuckDBStore,
    collection_name: str,
    source: str,
    chunks: list[tuple[int, str, dict]],
    ollama_host: str,
    embed_model: str,
    batch_size: int,
) -> int:
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), max(1, batch_size)):
        batch = chunks[start : start + max(1, batch_size)]
        texts = [content for _, content, _ in batch]
        embeddings.extend(embed_documents(texts, ollama_host, embed_model))
    return store.upsert_doc_chunks(collection_name, source, chunks, embeddings)


def collection_chunk_count(duckdb_path: Path, collection_name: str) -> int | None:
    try:
        store = DuckDBStore(duckdb_path)
        try:
            return store.doc_chunk_count(collection_name)
        finally:
            store.close()
    except Exception:
        return None
