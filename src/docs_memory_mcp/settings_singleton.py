"""Process-wide settings (single load per MCP server process)."""

from functools import lru_cache

from docs_memory_mcp.settings import Settings


@lru_cache
def get_settings() -> Settings:
    return Settings()
