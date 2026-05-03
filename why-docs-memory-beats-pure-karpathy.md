# Why Docs-Memory Beats the Pure Karpathy Approach

The pure Karpathy-style approach is attractive because it is simple: keep the important knowledge in plain text, give the model enough context, and let a capable LLM reason over it directly. It is a very good baseline. It keeps the system understandable, avoids premature infrastructure, and reminds us that most agent systems fail because they hide simple thinking behind complicated machinery.

But once a project starts living longer than a single session, pure context is not enough. A working assistant needs to find evidence, remember decisions, preserve priorities, and distinguish durable project knowledge from whatever happened to be pasted into the current prompt. That is where `docs-memory-mcp` is better: it keeps the Karpathy-style plain-text spirit, but adds retrieval, persistence, ranking, and tool boundaries around it.

## The Karpathy Baseline Is Good

The best part of the pure approach is its taste for simplicity. Markdown files, source code, notes, and a strong model can go very far. The user can inspect everything. There is no hidden database ritual before the system becomes useful. The model reads the material and reasons.

That matters. A memory system that cannot be inspected becomes a second source of confusion. A retrieval system that replaces judgment with magic keywords is worse than no retrieval at all. So the right comparison is not "simple text versus overbuilt RAG." The right comparison is:

- plain files and prompt context only
- plain files plus a small MCP layer for search, memory, indexing, and health checks

`docs-memory-mcp` keeps the source of truth human-readable while making it operational.

## Where Pure Context Breaks Down

Pure context has a few failure modes that show up quickly in real work.

First, the context window is temporary. If a decision matters tomorrow, it has to be copied back into a file or rediscovered later. The model may remember the tone of the conversation, but the project does not have a durable record unless someone writes one.

Second, context is usually selected manually. The user or agent guesses which files matter. That works when the project is tiny, but it becomes brittle when the relevant clue is in a test, a generated handoff, an old design note, or an implementation detail that nobody thought to paste.

Third, pure context does not separate evidence from preference. A source file, a README instruction, a health baseline, and a durable architectural decision are different kinds of knowledge. If they all arrive as flat text in a prompt, the model has to infer their importance every time.

Fourth, there is no built-in health signal. If retrieval feels stale, if the wrong model is used for embeddings, or if the index is empty, pure context has no equivalent of `docs_health`. The system just feels worse, and the user has to debug by vibes.

## What Docs-Memory Adds

`docs-memory-mcp` adds a thin operational layer around the same materials:

- `docs_search` finds relevant indexed documentation, source, tests, and wiki pages by meaning.
- `docs_read` gives direct access to the exact extracted text of a known file.
- `docs_diff` compares extracted documents without asking the model to eyeball two pasted blobs.
- `docs_index_file` and `docs_reindex` make indexing explicit and repeatable.
- `docs_health` reports whether DuckDB, Ollama, vector dimensions, and chunk counts look sane.
- `memory_upsert` stores durable decisions with importance, confidence, topic, and behavior hints.
- `memory_search` recalls those decisions using both semantic similarity and ranking metadata.
- `wiki_export` creates a readable view for humans while DuckDB remains the source of truth.

This is not a replacement for plain text. It is a way to make plain text usable by an agent across time.

## The Technical Part: DuckDB Is Both a Normal Database and a Vector Store

The important architectural move is that `docs-memory-mcp` does not treat "memory" as a mystical agent feature. It stores memory as ordinary data.

DuckDB is used as an embedded local database. That means the project gets normal relational tables, indexes, JSON metadata, timestamps, counters, and file-backed persistence without running Postgres, Qdrant, Redis, or a separate vector service. The current store has tables for:

- `doc_chunks`: extracted file text split into numbered chunks, with source paths and metadata.
- `memory_items`: durable memories with `memory_type`, `topic`, `importance`, `intensity`, `confidence`, `behavior_hint`, status, timestamps, and usage counters.
- `memory_profiles`: compiled summaries or behavior profiles.
- `vectors`: serialized embedding vectors shared by both document chunks and memories.

So the design is not "a vector DB instead of a real DB." It is a normal embedded database plus vectors in the same file. The relational rows carry meaning, provenance, metadata, and lifecycle. The vector rows carry semantic searchability.

That combination is the key difference from pure prompt stuffing. A plain text approach can keep files readable, but it cannot query them like structured state. A pure vector store can retrieve similar chunks, but often loses the ordinary database affordances that make a system debuggable. DuckDB gives both: boring tables for facts and metadata, vector blobs for semantic lookup.

## How Ollama Fits In

Ollama provides local embeddings. When the server indexes a file, it:

1. Finds eligible files under `DOCS_ROOT`.
2. Extracts text from Markdown, Python, TOML, PDF, or `.doc` HTML exports.
3. Scrubs obvious binary noise.
4. Splits text into overlapping chunks.
5. Sends each chunk to Ollama, usually `nomic-embed-text`.
6. Stores the chunk in `doc_chunks`.
7. Stores the corresponding embedding in `vectors` with `kind = 'doc'`.

The same pattern is used for durable memories. When `memory_upsert` stores a decision, the text is embedded with Ollama and written into `vectors` with `kind = 'memory'`, while the structured memory fields are written into `memory_items`.

This matters because the LLM does not have to carry everything in its immediate context. The context can be rebuilt on demand from a local semantic index. Ollama turns text into vectors; DuckDB keeps those vectors next to the normal records; MCP exposes the operations as tools the agent can call.

## Retrieval Is Not Just "Nearest Text"

For document search, the query is embedded through Ollama, then compared against stored document vectors. The implementation loads vectors for the active collection and matching dimension, computes cosine similarity with NumPy, and returns the best matching chunks with their source path and chunk number.

For memory search, the system does something more interesting. It first finds semantically similar memory vectors, then re-ranks them with metadata:

```text
score = similarity * 0.65
      + importance * 0.20
      + intensity * 0.10
      + confidence * 0.05
```

That is a real improvement over pure semantic search. A memory can be slightly less textually similar but still more important, more confident, or more behaviorally relevant. This is exactly the kind of thing pure prompt context cannot represent cleanly. In a prompt, everything is just text. In this system, some text is evidence, some text is a durable decision, some text has higher importance, and some text is only useful if it matches the current task.

## Why the Database Layer Matters

The ordinary database part is not incidental. It gives the system practical control:

- Incremental reindexing can delete and replace chunks for one source file without rebuilding everything.
- Full reindexing can clear only document vectors while leaving durable memories intact.
- Health checks can count chunks, vectors, active memories, dimensions, and memory usage.
- Search can filter by collection, vector kind, memory type, status, and minimum importance.
- Wiki export can render a human-readable view from the database without making the wiki the source of truth.

This is why `docs-memory-mcp` is stronger than a loose pile of Markdown plus a long prompt. The Markdown remains inspectable, but the operational state is queryable. You can ask: how many chunks exist, which model produced the embeddings, whether memory records survived a document reindex, whether vectors have the expected dimensions, and whether the local embedding service is reachable.

## MCP Makes the Architecture Usable by Agents

MCP is the boundary that turns this from a local script into agent infrastructure. The agent does not need to know DuckDB SQL or Ollama HTTP details. It gets small, named tools:

- search docs
- read a file
- diff two documents
- index changed files
- rebuild the index
- check health
- write memory
- search memory
- export/read the wiki view

That tool boundary is boring in the best way. It makes retrieval explicit, repeatable, and reviewable. A pure Karpathy approach depends on what the human pasted or what the assistant guessed. The MCP approach gives the assistant a protocol for asking the project itself.

## Why It Is Better

The biggest improvement is continuity. A pure prompt can be brilliant in the moment, but `docs-memory-mcp` lets the project carry memory between moments. A health check, an architectural decision, or a known repo convention can be stored once and found later without re-explaining the whole story.

The second improvement is evidence. Instead of relying on whatever was pasted into the chat, the agent can ask the repository what it knows. Search results come with source paths and chunks. Direct reads can confirm exact text. This gives the conversation a stronger anchor than memory alone.

The third improvement is prioritization. Not every remembered thing should matter equally. The memory model stores `importance`, `intensity`, `confidence`, `topic`, and `behavior_hint`, so retrieval can reflect what should guide the next answer, not just what sounds semantically nearby.

The fourth improvement is debuggability. With `docs_health`, the system can say whether Ollama is reachable, which collection is active, how many chunks are indexed, how many vectors exist, and which embedding dimensions are present. That turns "the assistant feels blind today" into something inspectable.

The fifth improvement is tool discipline. MCP tools make retrieval an explicit act. The assistant can say, "I searched memory first," or "I refreshed the index for the changed file." That is much easier to review than a giant hidden prompt assembled somewhere off-screen.

## The Practical Difference

In a pure Karpathy-style workflow, the user says:

> Here are the files. Here is the context. Please remember this.

In a docs-memory workflow, the project can say:

> Here is the indexed corpus. Here are the durable decisions. Here is the health of retrieval. Here is the exact file text when needed.

That difference matters most when the work is iterative. The assistant can start a task by checking prior decisions, search the codebase semantically, make a change, index the changed file, and store the new conclusion. The next session begins with a better project, not just a longer chat history.

## The Tradeoff

The MCP approach does add moving parts. It needs DuckDB storage, Ollama embeddings, an index, and client tool access. If the project is tiny or disposable, pure context may be enough. If the user only needs one answer from one file, there is no need to pretend a memory system is profound.

But for a living repository, those moving parts pay rent. They are small, inspectable, and local. They do not erase the plain-text workflow; they make it searchable, durable, and measurable.

## Bottom Line

The pure Karpathy approach is the right philosophical baseline: keep the system simple, textual, and model-friendly. `docs-memory-mcp` is better because it preserves that baseline while adding the missing operational layer: semantic retrieval, durable memory, ranking metadata, indexing commands, and health checks.

It is not "RAG instead of reasoning." It is reasoning with a better notebook.
