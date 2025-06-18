#!/usr/bin/env python3
"""Script to add URLs to embedding cache."""

import asyncio
import logging
import sys

from src.agent.embedding_cache import add_url_to_cache
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


async def main() -> None:
    """Add URL to embedding cache."""
    if len(sys.argv) < 2:
        logging.error("Usage: python create_cache.py <api_spec_url> [cache_file]")
        logging.error(
            "Example: python create_cache.py https://api.example.com/openapi.json"
        )
        sys.exit(1)

    api_spec_url = sys.argv[1]
    cache_file = sys.argv[2] if len(sys.argv) > 2 else Config.DEFAULT_CACHE_FILE

    logging.info("🚀 Adding URL to embedding cache...")
    logging.info(f"   URL: {api_spec_url}")
    logging.info(f"   Cache file: {cache_file}")
    logging.info(f"   Model: {Config.EMBEDDING_MODEL}")
    logging.info("")

    try:
        await add_url_to_cache(
            api_spec_url=api_spec_url,
            cache_file=cache_file,
            model=Config.EMBEDDING_MODEL,
        )

    except Exception as e:
        logging.error(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
