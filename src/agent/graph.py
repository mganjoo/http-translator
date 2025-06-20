"""HTTP Translator Agent.

Translates natural language queries into HTTP requests using OpenAPI specs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

import aiofiles.os
import httpx
import numpy as np
import voyageai
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from src.agent.embedding_cache import (
    get_cached_api_spec,
    get_cached_embeddings,
)
from src.config import Config

logger = logging.getLogger(__name__)


class Configuration(TypedDict):
    """Configurable parameters for the agent."""

    cache_file: str | None


@dataclass
class State:
    """State for the HTTP translator agent."""

    user_query: str = ""
    api_spec_url: str = ""
    api_spec: Dict[str, Any] | None = None
    rag_results: List[Dict[str, Any]] | None = None
    all_rag_scores: List[Dict[str, Any]] | None = None
    relevant_endpoints: List[Dict[str, Any]] | None = None
    http_request: Dict[str, Any] | None = None


def extract_endpoint_documents(api_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract endpoint documents from API spec for embedding."""
    paths = api_spec.get("paths", {})
    endpoint_documents = []

    for path, methods in paths.items():
        for method, details in methods.items():
            if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                summary = details.get("summary", "")
                description = details.get("description", "")

                # Create document text for embedding
                doc_text = f"Path: {path}\nMethod: {method.upper()}\nSummary: {summary}\nDescription: {description}"

                endpoint_documents.append(
                    {
                        "path": path,
                        "method": method.upper(),
                        "summary": summary,
                        "description": description,
                        "text": doc_text,
                    }
                )

    return endpoint_documents


async def extract_api_spec(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Extract API specification from the provided URL or cache."""
    configuration = config.get("configurable", {})
    api_url = state.api_spec_url
    if not api_url:
        raise ValueError("API spec URL must be provided in state")

    cache_file = configuration.get("cache_file", Config.DEFAULT_CACHE_FILE)

    # Ensure cache directory exists (async)
    if cache_file:
        await aiofiles.os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Try to get from cache first
    if cache_file:
        logger.info(f"🔍 Checking for cached api spec at {cache_file}...")
        cached_spec = await get_cached_api_spec(api_url, cache_file)
        if cached_spec:
            logger.info(f"✅ Using cached API spec for {api_url}")
            return {"api_spec": cached_spec}

    # Fallback to downloading
    logger.info(f"📥 Downloading API spec from {api_url}")
    async with httpx.AsyncClient() as client:
        response = await client.get(api_url)
        response.raise_for_status()
        api_spec = response.json()

    return {"api_spec": api_spec}


async def rag_retrieve_endpoints(
    state: State, config: RunnableConfig
) -> Dict[str, Any]:
    """Use RAG with Voyage embeddings to retrieve relevant endpoints."""
    configuration = config.get("configurable", {})
    api_url = state.api_spec_url
    if not api_url:
        raise ValueError("API spec URL must be provided in state")
    cache_file = configuration.get("cache_file", Config.DEFAULT_CACHE_FILE)

    # Try to get cached embeddings first
    endpoint_documents = None
    doc_embds = None

    if cache_file:
        logger.info(f"🔍 Checking for cached embeddings at {cache_file}...")
        cached_data = await get_cached_embeddings(api_url, cache_file)

        if cached_data:
            logger.info(f"✅ Using cached embeddings for {api_url}")
            endpoint_documents = cached_data["endpoint_documents"]
            doc_embds = cached_data["embeddings"]
        else:
            logger.info("⚠️ No cached embeddings found, creating new ones...")

    # Create embeddings if not cached
    if endpoint_documents is None or doc_embds is None:
        if not state.api_spec:
            raise ValueError("API spec not available")

        logger.info("🔮 Creating new embeddings...")

        # Extract endpoint documents from API spec
        endpoint_documents = extract_endpoint_documents(state.api_spec)

        # Initialize Voyage async client
        vo = voyageai.AsyncClient()  # type: ignore[attr-defined]

        # Embed all endpoint documents
        doc_texts = [doc["text"] for doc in endpoint_documents]
        doc_result = await vo.embed(
            doc_texts, model=Config.EMBEDDING_MODEL, input_type="document"
        )
        doc_embds = doc_result.embeddings

    # Now embed user query (always need fresh query embedding)
    vo = voyageai.AsyncClient()  # type: ignore[attr-defined]
    query_result = await vo.embed(
        [state.user_query], model=Config.EMBEDDING_MODEL, input_type="query"
    )
    query_embd = query_result.embeddings[0]

    # Compute similarities using dot product (cosine similarity since embeddings are normalized)
    similarities = np.dot(doc_embds, query_embd)

    # Get all endpoints sorted by similarity score (descending)
    all_indices = np.argsort(similarities)[::-1]

    # Create all_rag_scores with just endpoint name and score
    all_rag_scores = []
    for idx in all_indices:
        endpoint = endpoint_documents[idx]
        endpoint_name = f"{endpoint['method']} {endpoint['path']}"
        all_rag_scores.append(
            {
                "endpoint": endpoint_name,
                "score": float(similarities[idx]),
            }
        )

    # Get top K most similar endpoints for rag_results
    top_indices = all_indices[: Config.TOP_K_ENDPOINTS]

    rag_results = []
    for idx in top_indices:
        endpoint = endpoint_documents[idx]
        rag_results.append(
            {
                "path": endpoint["path"],
                "method": endpoint["method"],
                "summary": endpoint["summary"],
                "description": endpoint["description"],
                "similarity": float(similarities[idx]),
            }
        )

    return {"rag_results": rag_results, "all_rag_scores": all_rag_scores}


async def find_relevant_endpoints(
    state: State, config: RunnableConfig
) -> Dict[str, Any]:
    """Find relevant API endpoints based on user query."""
    if not state.rag_results:
        raise ValueError("RAG results not available")

    # Use the RAG results directly instead of re-processing the full API spec
    rag_endpoints = state.rag_results

    # Use Claude to find relevant endpoints from RAG results
    llm = ChatAnthropic(
        model=Config.MODEL_NAME,  # type: ignore[call-arg]
        max_tokens=Config.FIND_ENDPOINTS_MAX_TOKENS,  # type: ignore[call-arg]
    )

    prompt = f"""Given this user query: "{state.user_query}"

And thist list of pre-filtered API endpoints from RAG (top most relevant):
{json.dumps(rag_endpoints, indent=2)}

Please identify the MINIMAL set of API endpoints needed to fulfill the user's request. Prioritize:
1. Single endpoints that can accomplish the entire task
2. Batch operations over multiple single-item calls
3. The most efficient and direct approach

Important:
- Consider the descriptions and summaries of the endpoints to determine relevance.
- ONLY consider the pre-filtered endpoints provided above.
- Do NOT reference your own knowledge of APIs.

Return as few endpoints as possible - ideally just one if it can handle the request completely.

Return your response as a JSON list of objects with keys: path, method, summary, description.
Do NOT add your own summary or description, simply copy the relevant fields from the provided endpoints.
Only return the JSON, no other text."""

    response = await llm.ainvoke([HumanMessage(content=prompt)])

    try:
        content_str = str(response.content)
        relevant_endpoints = json.loads(content_str)
    except json.JSONDecodeError:
        # Fallback: extract JSON from response
        content_str = str(response.content)
        start = content_str.find("[")
        end = content_str.rfind("]") + 1
        if start != -1 and end != 0:
            relevant_endpoints = json.loads(content_str[start:end])
        else:
            relevant_endpoints = []

    return {"relevant_endpoints": relevant_endpoints}


def _extract_schema_refs(obj: Any, refs: set[str]) -> None:
    """Recursively extract schema references from a JSON object."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "$ref" and isinstance(value, str):
                # Extract schema name from references like "#/components/schemas/SchemaName"
                if value.startswith("#/components/schemas/"):
                    schema_name = value.split("/")[-1]
                    refs.add(schema_name)
            else:
                _extract_schema_refs(value, refs)
    elif isinstance(obj, list):
        for item in obj:
            _extract_schema_refs(item, refs)


async def construct_http_request(
    state: State, config: RunnableConfig
) -> Dict[str, Any]:
    """Construct the final HTTP request."""
    if not state.relevant_endpoints or not state.api_spec:
        raise ValueError("Relevant endpoints or API spec not available")

    # Get full specs for relevant endpoints
    paths = state.api_spec.get("paths", {})
    components = state.api_spec.get("components", {})

    full_endpoint_specs = []
    for endpoint in state.relevant_endpoints:
        path = endpoint["path"]
        method = endpoint["method"].lower()

        if path in paths and method in paths[path]:
            full_spec = paths[path][method]
            full_endpoint_specs.append(
                {"path": path, "method": method.upper(), "spec": full_spec}
            )

    # Use Claude to construct the HTTP request
    llm = ChatAnthropic(
        model=Config.MODEL_NAME,  # type: ignore[call-arg]
        max_tokens=Config.CONSTRUCT_REQUEST_MAX_TOKENS,  # type: ignore[call-arg]
    )

    # Extract schemas referenced by the endpoints
    referenced_schemas = set()
    for endpoint_spec in full_endpoint_specs:
        _extract_schema_refs(endpoint_spec["spec"], referenced_schemas)

    # Recursively extract schemas referenced by other schemas (transitive dependencies)
    if "schemas" in components:
        to_check = list(referenced_schemas)
        while to_check:
            schema_name = to_check.pop()
            if schema_name in components["schemas"]:
                schema_obj = components["schemas"][schema_name]
                new_refs = set()
                _extract_schema_refs(schema_obj, new_refs)
                for ref in new_refs:
                    if ref not in referenced_schemas:
                        referenced_schemas.add(ref)
                        to_check.append(ref)

    limited_components = {}
    if "schemas" in components and referenced_schemas:
        limited_components["schemas"] = {
            k: components["schemas"][k]
            for k in referenced_schemas
            if k in components["schemas"]
        }

    prompt = f"""Given this user query: "{state.user_query}"

And these API endpoint specifications:
{json.dumps(full_endpoint_specs, indent=2)}

And these schema components for reference (limited):
{json.dumps(limited_components, indent=2)}

Please construct the most efficient HTTP request to fulfill the user's query. Choose batch endpoints over multiple single-item calls when possible.

Return your response as a JSON object with these keys:
- method: HTTP method (GET, POST, etc.)
- url: Full URL path
- headers: Required headers as object
- query_params: Query parameters as object (if any)
- body: Request body as object (if any)
- description: Brief explanation of what this request does

Only return the JSON, no other text."""

    response = await llm.ainvoke([HumanMessage(content=prompt)])

    try:
        content_str = str(response.content)
        http_request = json.loads(content_str)
    except json.JSONDecodeError:
        # Fallback: extract JSON from response
        content_str = str(response.content)
        start = content_str.find("{")
        end = content_str.rfind("}") + 1
        if start != -1 and end != 0:
            http_request = json.loads(content_str[start:end])
        else:
            http_request = {"error": "Failed to parse HTTP request"}

    return {"http_request": http_request}


# Define the graph
graph = (
    StateGraph(State, config_schema=Configuration)
    .add_node("extract_api_spec", extract_api_spec)
    .add_node("rag_retrieve_endpoints", rag_retrieve_endpoints)
    .add_node("find_relevant_endpoints", find_relevant_endpoints)
    .add_node("construct_http_request", construct_http_request)
    .add_edge("__start__", "extract_api_spec")
    .add_edge("extract_api_spec", "rag_retrieve_endpoints")
    .add_edge("rag_retrieve_endpoints", "find_relevant_endpoints")
    .add_edge("find_relevant_endpoints", "construct_http_request")
    .compile(name="HTTP Translator")
)
