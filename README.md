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

**POST** `/chat/chat`

```json
{
  "message": "How many caregivers are available?",
  "groq_api_key": "optional_key_override",
  "access_token": "your_mcp_access_token_for_tools",
  "conversation_id": "optional_conversation_id",
  "model": "optional_model_override"
}
```

**Response:**
```json
{
  "message": "Based on the current data, there are 25 active caregivers available in your system.",
  "conversation_id": "conv_0",
  "used_tools": true,
  "tool_calls": [
    {
      "id": "call_123",
      "function": {
        "name": "get_caregiver_count",
        "arguments": "{\"status\":\"active\"}"
      }
    }
  ]
}
```

### Example with MCP Tools

```bash
curl -X POST "http://localhost:8000/chat/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "How many caregivers are in the system?",
       "access_token": "your_mcp_access_token"
     }'
```

### Example without MCP Tools

```bash
curl -X POST "http://localhost:8000/chat/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello, world!"}'
```

### Example with Python

```python
import requests

# Simple chat without MCP tools
response = requests.post(
    "http://localhost:8000/chat/chat",
    json={
        "message": "What is the capital of France?",
        "groq_api_key": "your_api_key_here"  # Optional if set in .env
    }
)

# Chat with MCP tools for data queries
response = requests.post(
    "http://localhost:8000/chat/chat",
    json={
        "message": "Show me the active caregivers",
        "access_token": "your_mcp_access_token",  # Enables MCP tools
        "groq_api_key": "your_api_key_here"  # Optional if set in .env
    }
)

print(response.json())
```

## Configuration

Create a `.env` file in the project root:

```env
# GROQ LLM Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# MCP Server Configuration (optional - for advanced tool usage)
MCP_SERVER_URL=https://aimcpserver.caresmartz360.net/mcp

# Web API Configuration
API_HOST=localhost
API_PORT=8000
DEBUG=true
```

## How It Works

The system intelligently determines when to use MCP tools:

1. **Simple Conversation**: Questions like "Hello", "How are you?", "What is 2+2?" are handled directly by GROQ LLM
2. **Tool-Enhanced Queries**: Questions about data, searching, counting, or specific information automatically use MCP tools when an `access_token` is provided

### Tool Detection Examples:
- ✅ **Uses Tools**: "How many caregivers?", "Find patient data", "Show appointments"
- ❌ **Direct LLM**: "Hello", "Thank you", "What is Python?"

## Additional Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `GET /conversation/{conversation_id}` - Get conversation history
- `DELETE /conversation/{conversation_id}` - Clear conversation
- `GET /conversations` - List all conversations

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