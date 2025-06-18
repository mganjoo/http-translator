# HTTP Translator Agent

A LangGraph agent that translates natural language queries into properly formed HTTP requests using OpenAPI specifications. Built with help from [Claude Code](https://www.anthropic.com/claude-code).

## Overview

This agent takes a natural language query and an OpenAPI spec URL, then:

1. **Extracts API spec** - Downloads and caches the OpenAPI specification
2. **RAG retrieval** - Uses Voyage embeddings to find the most relevant API endpoints
3. **Endpoint selection** - Uses Claude to identify the minimal set of endpoints needed
4. **HTTP request construction** - Generates the final HTTP request with proper parameters

## Quick Start

### Prerequisites

- Python 3.11+
- uv package manager
- API keys for Anthropic and Voyage AI (set in `.env`)

### Installation

```bash
# Clone the repository
git clone https://github.com/mganjoo/http-translator.git
cd http-translator

# Install dependencies
uv sync
```

### Environment Setup

Copy `.env.example` into an `.env` file and replace the two keys below:

```bash
ANTHROPIC_API_KEY=your_anthropic_key_here
VOYAGE_API_KEY=your_voyage_key_here
```

### Running the Agent

#### Development Mode

```bash
# Start the LangGraph development server
langgraph dev

# Pre-cache API embeddings for faster performance (recommended for large APIs)
python create_cache.py "https://dac-static.atlassian.com/cloud/jira/platform/swagger-v3.v3.json?_v=1.7940.0-0.1323.0"

# In another terminal, test with a query
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

#### Production Mode

```bash
# Build and deploy
langgraph build
langgraph deploy
```

## Configuration

All configuration is centralized in `src/config.py`:

- **Model settings**: Claude model and token limits
- **Embedding settings**: Voyage model and retrieval parameters
- **Cache settings**: Default cache file location
- **API settings**: Default OpenAPI spec URL

## Example Queries

- "Get all issues assigned to me"
- "Create a new issue in project ABC"
- "Search for issues with status 'In Progress'"
- "Get project details for project KEY-123"

## Project Structure

```
src/
├── agent/
│   ├── graph.py           # Main LangGraph workflow
│   └── embedding_cache.py # Caching utilities
├── config.py              # Centralized configuration
└── ...

cache/                     # API spec and embedding cache
langgraph.json            # LangGraph configuration
```

## Development

See [CLAUDE.md](CLAUDE.md) for detailed development instructions and agent architecture.

## License

[Add your license here]
