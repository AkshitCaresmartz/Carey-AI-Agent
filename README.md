# PyMCP Chat API

A clean and powerful Python MCP client with FastAPI-based chat service using GROQ LLM and intelligent MCP (Model Context Protocol) tools integration.

## Features

- **Single chat endpoint**: `POST /chat/chat`
- **GROQ LLM integration** for natural conversation
- **Intelligent MCP tools integration** - automatically decides when to use tools
- **Conversation history management**
- **Simple configuration** via environment variables
- **FastAPI** with automatic documentation
- **Built with UV** for modern Python package management

## Quick Start

1. **Install UV and set up project:**
   ```bash
   pip install uv
   uv venv
   .\.venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   uv sync
   ```

2. **Set your GROQ API key in `.env`:**
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Start the server:**
   ```bash
   python -m mcp_client.main
   ```
   
   Or use the convenience scripts:
   - Windows: `start.bat`
   - PowerShell: `start.ps1`

4. **Use the API:**
   - Server: `http://localhost:8000`
   - Docs: `http://localhost:8000/docs`
   - Chat endpoint: `http://localhost:8000/chat/chat`

## API Usage

### Chat Endpoint
# PyMCP Chat API

PyMCP Chat API is a small FastAPI-based service that connects a GROQ LLM provider with an MCP (Model Capability Provider) server. The service:

- Converts MCP tool metadata into function descriptors consumable by GROQ's function-calling API.
- Lets the model decide whether to call MCP tools, executes requested tools, and feeds results back to the model.
- Synthesizes a concise, human-readable reply based on tool outputs.

This repository is intended for local development and testing. It uses an in-memory conversation store (replace with Redis/DB for production).

## Repository layout

- `main.py` - FastAPI application entrypoint and Uvicorn launcher
- `pyproject.toml` - Project metadata and dependencies
- `pymcp/`
  - `chat_api.py` - FastAPI router exposing `/chat` endpoints and conversation lifecycle
  - `mcp_client.py` - MCP client wrapper for listing and calling tools
  - `groq_client.py` - GROQ LLM integration and tool orchestration
  - `config.py` - pydantic-based environment settings

## Requirements

- Python 3.13+

Dependencies (declared in `pyproject.toml`):

- fastapi
- groq
- httpx
- mcp[cli]
- pydantic
- pydantic-settings
- uvicorn

## Environment variables

Set the following environment variables (or use a `.env` loader in your environment):

- `GROQ_API_KEY` (required) — API key used to authenticate with the GROQ provider.
- `GROQ_MODEL` (optional) — model name to use (the app will read a default from settings if not provided).
- `MCP_SERVER_URL` (optional) — MCP server base URL (e.g., `http://localhost:8080`).
- `API_HOST` (optional) — host for Uvicorn to bind to (default `127.0.0.1`).
- `API_PORT` (optional) — port for Uvicorn to bind to (default `8000`).
- `DEBUG` (optional) — `true` to enable reload and more verbose logging.

Important: The GROQ API key and model come from environment/settings only and must not be passed in request bodies. MCP access requires a Bearer token passed in the `Authorization` header when calling `/chat/chat`.

## Install (PowerShell)

Recommended: create and use a virtual environment.

```powershell
# From project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Alternatively install dependencies with `pip install -r requirements.txt` if you export them.

## Run the app (PowerShell)

Set environment variables and start the server.

```powershell
$env:GROQ_API_KEY = "sk-...your-key..."
$env:GROQ_MODEL = "groq-alpha-1"
$env:MCP_SERVER_URL = "http://localhost:8080"
$env:API_HOST = "127.0.0.1"
$env:API_PORT = "8000"
$env:DEBUG = "true"

- `GET /conversations` - List all conversations
python main.py
```

Visit `http://127.0.0.1:8000/docs` for OpenAPI docs and to test endpoints interactively.

## Example requests

- Health check

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

- Chat (example request with Authorization header)

```powershell
$body = @{ message = "How many open shifts exist for 2025-09-30?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/chat/chat -Method Post -Body $body -ContentType 'application/json' -Headers @{ Authorization = 'Bearer <MCP_TOKEN>' }
```

The `/chat/chat` response contains `message`, `conversation_id`, `used_tools`, and `tool_calls`.

## Logging & debugging

The app configures basic logging in `main.py`. Set `DEBUG=true` to enable reload and more verbose logs. The code logs:

- MCP client initialization and masked token presence
- Tool discovery and per-tool metadata
- Model responses and tool-call decisions
- Tool invocation arguments and results

If you hit errors related to authentication or tool payload validation, inspect logs for the precise GROQ/MCP responses — the provider enforces strict message shapes for function-calling.

## Notes & next steps

- Conversation state is currently held in memory. Use Redis or a database for production.
- Consider adding unit tests for `groq_client` tool normalization and synthesis flow.
- If you want, I can add a `.env.example`, a pinned `requirements.txt`, or an automated test script.

---

If you'd like, I can also generate a `requirements.txt` with pinned versions or add a `.env.example`. Let me know which you'd prefer.

## Project Structure

```
├── mcp_client/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   └── chat.py          # Chat API endpoints
│   └── llm/
│       ├── __init__.py
│       └── groq_client.py   # GROQ LLM client
├── requirements.txt         # Python dependencies
├── .env                     # Configuration file
├── start.bat               # Windows startup script
└── start.ps1               # PowerShell startup script
```

## Requirements

- Python 3.8+
- GROQ API key
- Internet connection for GROQ API

## Notes

- The API includes CORS middleware for frontend integration
- Conversations are stored in memory (use Redis/database for production)
- Default model is `llama-3.1-8b-instant`
- All endpoints return JSON responses