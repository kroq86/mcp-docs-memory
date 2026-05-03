"""Environment configuration."""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_docs_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_host: str = Field(default="http://127.0.0.1:11434", validation_alias="OLLAMA_HOST")
    ollama_embed_model: str = Field(default="nomic-embed-text", validation_alias="OLLAMA_EMBED_MODEL")
    docs_root: Path = Field(default_factory=_default_docs_root, validation_alias="DOCS_ROOT")
    duckdb_path: Path | None = Field(default=None, validation_alias="DUCKDB_PATH")
    collection_name: str = Field(default="docs_memory", validation_alias="COLLECTION_NAME")
    rag_top_k: int = Field(default=5, validation_alias="RAG_TOP_K")
    chunk_size: int = Field(default=1500, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, validation_alias="CHUNK_OVERLAP")
    index_batch_size: int = Field(default=32, validation_alias="INDEX_BATCH_SIZE")

    @model_validator(mode="after")
    def default_duckdb_path(self) -> Self:
        if self.duckdb_path is None:
            object.__setattr__(self, "duckdb_path", self.docs_root / "mcp-data" / "docs-memory.duckdb")
        return self
