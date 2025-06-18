"""Configuration settings for the HTTP Translator agent."""

from pathlib import Path


class Config:
    """Central configuration for the HTTP Translator agent."""

    # Model settings
    MODEL_NAME = "claude-sonnet-4-20250514"
    FIND_ENDPOINTS_MAX_TOKENS = 1000
    CONSTRUCT_REQUEST_MAX_TOKENS = 2000

    # Embedding settings
    EMBEDDING_MODEL = "voyage-3.5"
    TOP_K_ENDPOINTS = 10

    # Cache settings
    CACHE_DIR = Path("cache")
    DEFAULT_CACHE_FILE = str(CACHE_DIR / "api_cache.json")
