"""Sync Ollama embedding calls for the indexer.

Supports both API styles:
- Modern: POST /api/embed with {"model", "input"} → {"embeddings": [[...]]}
- Legacy: POST /api/embeddings with {"model", "prompt"} → {"embedding": [...]}

Ollama often returns HTTP 404 with a JSON body when the *model* is missing — that is NOT
"wrong URL". We only fall back to /api/embeddings on a literal route 404 (plain body).
"""

from __future__ import annotations

import json
from typing import List

import httpx

# After first successful strategy, stick to it (avoids double round-trip per chunk).
_mode: str | None = None  # "embed" | "embeddings"


def _raise_if_model_missing(r: httpx.Response, model: str) -> None:
    """404 + JSON error about missing model → clear error (do not treat as wrong path)."""
    if r.status_code != 404:
        return
    try:
        data = r.json()
    except json.JSONDecodeError:
        return
    err = str(data.get("error", "")).lower()
    if "model" in err and ("not found" in err or "pull" in err):
        raise RuntimeError(
            f"Ollama has no model {model!r} for embeddings: {data.get('error')}. "
            f"Install one: ollama pull nomic-embed-text   OR set OLLAMA_EMBED_MODEL to a tag from `ollama list` "
            f"you already have (e.g. llama3.2:3b). Verify: curl -s http://127.0.0.1:11434/api/tags"
        ) from None


def _is_literal_route_404(r: httpx.Response) -> bool:
    if r.status_code != 404:
        return False
    try:
        r.json()
        return False
    except json.JSONDecodeError:
        return True


def _parse_modern(data: dict) -> List[float]:
    embs = data.get("embeddings")
    if embs is not None:
        if len(embs) == 1:
            return list(embs[0])
        if embs and isinstance(embs[0], (int, float)):
            return list(embs)
        raise ValueError(f"Unexpected embeddings shape: {type(embs)}")
    if "embedding" in data:
        return list(data["embedding"])
    raise ValueError(f"Unexpected Ollama /api/embed response keys: {list(data.keys())}")


def _parse_legacy(data: dict) -> List[float]:
    if "embedding" in data:
        return list(data["embedding"])
    raise ValueError(f"Unexpected Ollama /api/embeddings response keys: {list(data.keys())}")


def embed_documents(
    texts: List[str],
    ollama_host: str,
    model: str,
    timeout: float = 120.0,
) -> List[List[float]]:
    if not texts:
        return []
    global _mode
    host = ollama_host.rstrip("/")
    out: List[List[float]] = []
    with httpx.Client(timeout=timeout) as client:
        for t in texts:
            if _mode == "embeddings":
                r = client.post(
                    f"{host}/api/embeddings",
                    json={"model": model, "prompt": t},
                )
                if r.status_code == 404:
                    _raise_if_model_missing(r, model)
                r.raise_for_status()
                out.append(_parse_legacy(r.json()))
                continue

            if _mode == "embed":
                r = client.post(
                    f"{host}/api/embed",
                    json={"model": model, "input": t},
                )
                if r.status_code == 404:
                    _raise_if_model_missing(r, model)
                r.raise_for_status()
                out.append(_parse_modern(r.json()))
                continue

            # First text: try /api/embed; fall back only on literal route 404
            r = client.post(
                f"{host}/api/embed",
                json={"model": model, "input": t},
            )
            if r.status_code == 200:
                _mode = "embed"
                out.append(_parse_modern(r.json()))
                continue

            if r.status_code == 404:
                _raise_if_model_missing(r, model)
                if _is_literal_route_404(r):
                    r2 = client.post(
                        f"{host}/api/embeddings",
                        json={"model": model, "prompt": t},
                    )
                    if r2.status_code == 404:
                        _raise_if_model_missing(r2, model)
                    r2.raise_for_status()
                    _mode = "embeddings"
                    out.append(_parse_legacy(r2.json()))
                    continue

            r.raise_for_status()
            _mode = "embed"
            out.append(_parse_modern(r.json()))

    return out


def reset_embed_endpoint_cache() -> None:
    global _mode
    _mode = None
