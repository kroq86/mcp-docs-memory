# docs-memory-mcp

MCP server (stdio) for semantic document search and structured memory using **Ollama** embeddings and **DuckDB** persistent storage.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally (default `http://127.0.0.1:11434`)
- Embedding model pulled, e.g. `ollama pull nomic-embed-text` (or set `OLLAMA_EMBED_MODEL` to a tag you already have)

## Install

```bash
cd mcp/docs-memory-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Cursor / MCP Config

Use exactly one config file for `docs-memory`, otherwise Cursor may show duplicate servers.

After changing `mcp.json`: **Developer: Reload Window** (or restart Cursor). If the row still says Disabled, turn on the toggle for `docs-memory` in **Settings -> MCP**.

`docs_index_file` updates only the file(s) you name. `docs_reindex` clears and rebuilds the active document collection. Neither is required to start the MCP server.

## Why It May Feel No Different

This server does runtime retrieval, not automatic memory injection. The LLM only gets extra knowledge when the MCP client exposes the tools and the model calls `docs_search`, `docs_read`, `docs_diff`, or memory tools.

If it feels the same as a plain LLM session, check these first:

1. **Consent / trust button:** in Cursor, open **Settings -> MCP** and make sure `docs-memory` is enabled/trusted/approved.
2. **Tool visibility:** ask the agent directly: "Use `docs_search` before answering."
3. **Index health:** call `docs_health`. You want `ollama_ok=true` and a non-zero `chunks_in_index`.
4. **Retrieval quality:** try a query with a phrase you know appears in the docs.
5. **Task fit:** vector search helps with evidence lookup. Memory tools help with durable decisions, preferences, and behavior hints.

## Tools

| Tool | Purpose |
|------|---------|
| `docs_search` | Query indexed documents by meaning; returns snippets with `source` path and chunk index |
| `docs_read` | Direct read: extracted plain text from one path (`.md` / `.doc` / `.pdf`) under `DOCS_ROOT` |
| `docs_diff` | Unified diff between two extracted documents |
| `docs_index_file` | Incremental reindex for one or more relative paths |
| `docs_reindex` | Full rebuild of indexed documents for the active collection |
| `docs_health` | Ollama reachability, DuckDB path, chunk count, and vector dimensions |
| `memory_upsert` | Store one typed memory with importance, intensity, confidence, and behavior hint |
| `memory_search` | Search active memories using similarity plus ranking metadata |
| `memory_profile_get` | Read one compiled profile |
| `memory_profile_upsert` | Create or replace one compiled profile |
| `wiki_export` | Generate a readable markdown wiki view from DuckDB |
| `wiki_read` | Read one generated wiki markdown file |

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `DOCS_ROOT` | Auto: repository root for this package | Root directory for indexed documents |
| `DUCKDB_PATH` | `{DOCS_ROOT}/mcp-data/docs-memory.duckdb` | DuckDB persistence file |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama base URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model used for indexing and search |
| `COLLECTION_NAME` | `docs_memory` | Logical collection in DuckDB |
| `CHUNK_SIZE` | `1500` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_TOP_K` | `5` | Default top-k for search |
| `INDEX_BATCH_SIZE` | `32` | Embedding batch size during indexing |

## What Gets Indexed

Extensions: `.md` (UTF-8), `.doc` (HTML export -> plain text via BeautifulSoup), `.pdf` (text extraction via pypdf).

Paths under `DOCS_ROOT`:

- Root markdown files and `pyproject.toml`
- `struct.md` at repository root, if present
- `docs/**/*.md`, `docs/**/*.doc`, `docs/**/*.pdf`
- `src/**/*.py`
- `tests/**/*.py`
- `wiki/**/*.md`
- `core/**/*.md`, `core/docs/**/*.doc`, `core/docs/**/*.pdf`
- `etl/**/*.md`
- `frontend/**/*.md`, `frontend/docs/**/*.md`
- `mcp/**/*.md`

Skipped path segments include `node_modules`, `.git`, `__pycache__`, `dist`, `build`, `.venv`, and `mcp-data`.

## Memory Model

Documents and memory records live in the same DuckDB file.

`memory_upsert` stores a memory with:

- `text`
- `memory_type`
- `topic`
- `importance`
- `intensity`
- `confidence`
- `behavior_hint`
- optional JSON metadata

`memory_search` combines semantic similarity with importance, intensity, and confidence. This is useful for recalling not only what is similar, but what should affect the next answer.

## Wiki View

DuckDB is the source of truth. The markdown wiki is a generated view.

`wiki_export` writes:

- `wiki/index.md`
- `wiki/sources.md`
- `wiki/memories.md`
- `wiki/profiles.md`

Use this when you want a human-readable compiled layer for Obsidian, diffs, review, or handoff. Use `docs_search` and `memory_search` when you want runtime retrieval.

## Run Manually

```bash
cd mcp/docs-memory-mcp
.venv/bin/python -m docs_memory_mcp
```

The process waits on stdio for the MCP client.
