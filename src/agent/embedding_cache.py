"""Utility for caching API spec embeddings."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import voyageai

logger = logging.getLogger(__name__)


async def extract_endpoint_documents(
    api_spec_url: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract endpoint documents from API spec URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(api_spec_url)
        response.raise_for_status()
        api_spec = response.json()

    # Extract endpoint documents from API spec
    paths = api_spec.get("paths", {})
    endpoint_documents = []

    for path, methods in paths.items():
        for method, details in methods.items():
            if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                summary = details.get("summary", "")
                description = details.get("description", "")

                # Create document text for embedding
                doc_text = f"{summary}\n{description}"

                endpoint_documents.append(
                    {
                        "path": path,
                        "method": method.upper(),
                        "summary": summary,
                        "description": description,
                        "text": doc_text,
                    }
                )

    return endpoint_documents, api_spec


async def load_cache_dict(cache_file: str) -> Dict[str, Any]:
    """Load or create cache dictionary."""

    def _load_cache() -> Dict[str, Any]:
        cache_path = Path(cache_file)
        if cache_path.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)  # type: ignore[no-any-return]
            except Exception as e:
                logger.warning(f"⚠️ Error loading cache, creating new one: {e}")
        return {}

    return await asyncio.to_thread(_load_cache)


async def save_cache_dict(cache_dict: Dict[str, Any], cache_file: str) -> None:
    """Save cache dictionary to file."""

    def _save_cache() -> None:
        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_dict, f, indent=2, ensure_ascii=False)

    await asyncio.to_thread(_save_cache)


async def add_url_to_cache(api_spec_url: str, cache_file: str, model: str) -> None:
    """Add or update URL embeddings in cache."""
    logger.info(f"📥 Processing {api_spec_url}...")

    # Load existing cache
    cache_dict = await load_cache_dict(cache_file)

    # Check if URL already exists
    if api_spec_url in cache_dict:
        logger.info("✅ URL already in cache, skipping...")
        return

    # Extract endpoint documents
    logger.info("📊 Extracting endpoints...")
    endpoint_documents, api_spec = await extract_endpoint_documents(api_spec_url)
    logger.info(f"   Found {len(endpoint_documents)} endpoints")

    # Embed documents
    logger.info(f"🔮 Creating embeddings with {model}...")
    vo = voyageai.AsyncClient()  # type: ignore[attr-defined]
    doc_texts = [doc["text"] for doc in endpoint_documents]
    doc_result = await vo.embed(doc_texts, model=model, input_type="document")

    # Store in cache
    cache_dict[api_spec_url] = {
        "api_spec": api_spec,
        "endpoint_documents": endpoint_documents,
        "embeddings": doc_result.embeddings,
        "model": model,
    }

    # Save cache
    logger.info(f"💾 Saving to {cache_file}...")
    await save_cache_dict(cache_dict, cache_file)

    cache_path = Path(cache_file)
    logger.info("✅ Cache updated!")
    logger.info(f"   - File: {cache_file}")
    logger.info(f"   - Size: {cache_path.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info(f"   - URLs cached: {len(cache_dict)}")


async def get_cached_embeddings(
    api_spec_url: str, cache_file: str
) -> Optional[Dict[str, Any]]:
    """Get cached embeddings and documents for a specific URL."""
    cache_dict = await load_cache_dict(cache_file)

    if api_spec_url not in cache_dict:
        return None

    cache_data = cache_dict[api_spec_url]
    return {
        "endpoint_documents": cache_data["endpoint_documents"],
        "embeddings": cache_data["embeddings"],
        "model": cache_data["model"],
    }


async def get_cached_api_spec(
    api_spec_url: str, cache_file: str
) -> Optional[Dict[str, Any]]:
    """Get cached API spec for a URL."""
    cache_dict = await load_cache_dict(cache_file)

    if api_spec_url in cache_dict:
        return cache_dict[api_spec_url]["api_spec"]  # type: ignore[no-any-return]

    return None
