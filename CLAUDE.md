# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Linting and Formatting
- `make lint` - run linters (ruff check, format diff, mypy strict)
- `make format` - run code formatters (ruff format, fix imports)
- `make lint_package` - lint only src/ directory

### Development Server
- `langgraph dev` - start LangGraph development server on localhost:2024
- `langgraph build` - build for production
- `langgraph deploy` - deploy to production

### Cache Management
- `python create_cache.py "<api_spec_url>"` - pre-cache API embeddings for faster performance

### Testing Agent
Example curl to test agent locally:
```bash
curl -X POST "http://localhost:2024/runs/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "agent",
    "input": {
      "user_query": "Create a new issue in project ABC",
      "api_spec_url": "https://dac-static.atlassian.com/cloud/jira/platform/swagger-v3.v3.json?_v=1.7940.0-0.1323.0"
    }
  }' | grep '^data: ' | tail -n 1 | sed -e 's/^data: //' | jq .http_request
```

## Architecture

This is a LangGraph-based HTTP translator agent that converts natural language queries into HTTP requests using OpenAPI specifications.

### Core Workflow (src/agent/graph.py)
The agent implements a 4-step pipeline:

1. **extract_api_spec** - Downloads and caches OpenAPI specifications
2. **rag_retrieve_endpoints** - Uses Voyage embeddings to find relevant API endpoints
3. **find_relevant_endpoints** - Uses Claude to select minimal set of endpoints
4. **construct_http_request** - Generates final HTTP request with parameters

### Key Components

- **State Management**: Uses dataclass State with fields for user_query, api_spec_url, api_spec, rag_results, relevant_endpoints, http_request
- **Configuration**: Centralized in src/config.py with model settings, embedding parameters, and cache configuration
- **Caching System**: src/agent/embedding_cache.py handles persistent caching of API specs and embeddings using pickle
- **RAG Pipeline**: Uses Voyage-3.5 embeddings with cosine similarity search, retrieves top 20 endpoints by default

### Models Used
- **Claude Sonnet 4** (claude-sonnet-4-20250514) for endpoint selection and HTTP request construction
- **Voyage-3.5** for document/query embeddings
- **Token limits**: 1000 for endpoint finding, 2000 for request construction

### Environment Requirements
Requires `.env` file with:
- `ANTHROPIC_API_KEY`
- `VOYAGE_API_KEY`

### Cache Strategy
- Default cache location: `cache/api_cache.pkl`
- Caches both API specifications and their embeddings
- Async cache operations to avoid blocking
- Cache is URL-keyed for multiple API spec support

