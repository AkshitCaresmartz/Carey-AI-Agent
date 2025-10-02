"""Chat API implementation."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from .mcp_client import MCPClient
from .config import Settings, get_settings
from .groq_client import GroqClient, ConversationManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# In-memory conversation storage (in production, use Redis or database)
conversations: Dict[str, ConversationManager] = {}

bearer_scheme = HTTPBearer(auto_error=False)


@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Chat API is running"}


class ChatRequest(BaseModel):
    """Request model for chat completion.

    Note: GROQ API key and model are read from environment/settings and must not be
    supplied in the request body. Access token must be provided in the
    Authorization header using the Bearer scheme: "Authorization: Bearer <token>".
    """
    message: str = Field(..., description="User message to send to the LLM")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")


class ChatResponse(BaseModel):
    """Response model for chat completion."""
    message: str = Field(..., description="LLM response message")
    conversation_id: str = Field(..., description="Conversation ID for future requests")
    used_tools: bool = Field(..., description="Whether MCP tools were used")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Details of tool calls made")


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    conversation_id: str
    messages: List[Dict[str, Any]]


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(bearer_scheme)])
async def chat(
    chat_request: ChatRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
):
    """
    Generate a chat completion using GROQ LLM with intelligent MCP tool integration.
    
    Flow:
    1. Try to initialize MCP session with access_token
    2. If MCP fails → fallback to simple LLM response
    3. If MCP succeeds → intelligent tool detection (use tools when needed, LLM otherwise)
    """
    # Log that a request arrived and basic header presence for debugging
    logger.info(f"Chat request received: {chat_request.message[:50]}...")
    logger.debug(f"Chat request full payload: conversation_id={chat_request.conversation_id}")
    try:
        header_keys = list(request.headers.keys())
        logger.debug(f"Request headers present: {header_keys}")
        if credentials is None:
            logger.info("No Authorization header present in request (credentials is None)")
            # If there's no token, treat as unauthorized per user's request
            raise HTTPException(status_code=401, detail="Unauthorized: missing Authorization Bearer token")
        # Mask token for logs
        auth_raw = credentials.scheme + " " + credentials.credentials
        try:
            masked = auth_raw[:8] + '...' + auth_raw[-8:]
        except Exception:
            masked = auth_raw[:6] + '...'
        logger.info(f"Authorization header received (masked): {masked}")
        # Use the credential token
        header_token = credentials.credentials
    except HTTPException:
        raise
    except Exception:
        logger.debug("Failed to inspect request headers for debugging", exc_info=True)
    try:
        # Get or create conversation manager
        conversation_id = chat_request.conversation_id or f"conv_{len(conversations)}"
        logger.info(f"Using conversation_id: {conversation_id}")

        if conversation_id not in conversations:
            conversations[conversation_id] = ConversationManager()
            logger.info(f"Created new conversation manager for {conversation_id}")

        conversation_manager = conversations[conversation_id]

        # GROQ configuration must come from environment/settings only
        groq_api_key = settings.groq_api_key
        groq_model = settings.groq_model

        if not groq_api_key:
            raise HTTPException(
                status_code=400,
                detail="GROQ API key is required. Set GROQ_API_KEY in environment or .env file."
            )

        # Try to initialize MCP session first
        mcp_client = None
        use_mcp_tools = False

        # header_token has been set above from HTTPBearer credentials
        if header_token:
            logger.info("Authorization Bearer token provided - initializing MCP session")
            try:
                # Create MCP client
                mcp_client = MCPClient(
                    server_url=settings.mcp_server_url,
                    access_token=header_token,
                    settings=settings
                )

                # Test the MCP connection by trying to list tools
                logger.info("Testing MCP connection with provided token...")
                tools = await mcp_client.list_tools()

                # Create GROQ client with MCP integration
                groq_client = GroqClient(
                    api_key=groq_api_key,
                    model=groq_model,
                    mcp_client=mcp_client
                )

                # Initialize tools in GROQ client
                await groq_client.initialize_tools()
                # If tools are available, add a short system instruction to the
                # conversation so the model knows to use tools for data lookups
                # (counts, searches, client-specific queries).
                if groq_client.tools:
                    # Add system guidance at the start of the conversation
                    conversation_manager.add_system_message(
                        "You have access to agent tools for retrieving data. "
                        "When the user asks for counts, lookups, or any information "
                        "that requires querying the backend (for example: number of offices, caregivers, appointments), "
                        "prefer calling the appropriate tool and return the tool results to the user."
                    )

                if groq_client.tools:
                    use_mcp_tools = True
                    logger.info(f"MCP session initialized successfully with {len(groq_client.tools)} tools")
                else:
                    logger.warning("MCP session initialized but no tools available")
                    use_mcp_tools = True

            except Exception as mcp_error:
                logger.error(f"MCP connection failed: {mcp_error}")
                error_str = str(mcp_error).lower()

                # Check for authentication failures
                auth_indicators = [
                    "unauthorized", "authentication", "401", "403", "forbidden", 
                    "invalid token", "token expired", "access denied", "permission denied",
                    "jwt", "bearer", "signature", "expired", "invalid", "malformed"
                ]

                if any(indicator in error_str for indicator in auth_indicators):
                    logger.info(f"Treating as authentication failure: {mcp_error}")
                    raise HTTPException(
                        status_code=401,
                        detail="Unauthorized: Invalid or expired access token"
                    )
                else:
                    # For network errors, try creating MCP client anyway
                    logger.warning(f"Network error, will attempt MCP execution anyway: {mcp_error}")
                    mcp_client = MCPClient(
                        server_url=settings.mcp_server_url,
                        access_token=header_token,
                        settings=settings
                    )
                    groq_client = GroqClient(
                        api_key=groq_api_key,
                        model=groq_model,
                        mcp_client=mcp_client
                    )
                    use_mcp_tools = True
        else:
            logger.info("No Authorization Bearer token provided, using simple LLM mode")

        # Create GROQ client with or without MCP
        if not use_mcp_tools:
            groq_client = GroqClient(
                api_key=groq_api_key,
                model=groq_model,
                mcp_client=None
            )
            logger.info("GROQ client created in simple LLM mode")

        logger.debug(f"Entering response generation path: use_mcp_tools={use_mcp_tools}")

        logger.info(f"GROQ client ready - MCP tools: {use_mcp_tools}")

        # Generate response with intelligent tool usage
        logger.info("Generating response...")
        if use_mcp_tools:
            # Use MCP tools
            try:
                response_content, used_tools, tool_calls_info = await groq_client.chat_completion(
                    conversation_manager,
                    chat_request.message
                )
            except Exception as mcp_runtime_error:
                # If MCP fails during execution, return the error
                logger.error(f"MCP execution failed: {mcp_runtime_error}")
                if "unauthorized" in str(mcp_runtime_error).lower() or "authentication" in str(mcp_runtime_error).lower():
                    raise HTTPException(
                        status_code=401,
                        detail=f"MCP authentication failed: {str(mcp_runtime_error)}"
                    )
                else:
                    raise HTTPException(
                        status_code=503,
                        detail=f"MCP service error: {str(mcp_runtime_error)}"
                    )
        else:
            # Simple LLM fallback
            conversation_manager.add_user_message(chat_request.message)
            response = groq_client.client.chat.completions.create(
                model=groq_model,
                messages=conversation_manager.get_messages(),
                temperature=0.7,
                max_tokens=1000
            )
            response_content = response.choices[0].message.content or "I don't have a response to that."
            conversation_manager.add_assistant_message(response_content)
            used_tools = False
            tool_calls_info = None

        logger.info("Response generated successfully")
        logger.debug(f"Returning response: used_tools={used_tools} conversation_id={conversation_id}")

        return ChatResponse(
            message=response_content,
            conversation_id=conversation_id,
            used_tools=used_tools,
            tool_calls=tool_calls_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


@router.get("/conversation/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(conversation_id: str):
    """Get conversation history for a specific conversation ID."""
    try:
        if conversation_id not in conversations:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation {conversation_id} not found"
            )
        
        conversation_manager = conversations[conversation_id]
        
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=conversation_manager.get_messages()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation history: {str(e)}"
        )


@router.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear a specific conversation history."""
    try:
        if conversation_id in conversations:
            conversations[conversation_id].clear_messages()
            return {"message": f"Conversation {conversation_id} cleared"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation {conversation_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear conversation: {str(e)}"
        )


@router.get("/conversations")
async def list_conversations():
    """List all active conversation IDs."""
    return {
        "conversations": list(conversations.keys()),
        "total": len(conversations)
    }


@router.get("/debug/headers")
async def debug_headers(request: Request):
    """Return incoming request headers for debugging (Authorization is masked)."""
    try:
        headers = {k: v for k, v in request.headers.items()}
        auth = headers.get('authorization')
        masked = None
        if auth:
            try:
                masked = auth[:8] + '...' + auth[-8:]
            except Exception:
                masked = auth[:6] + '...'

        return {
            "has_authorization": auth is not None,
            "authorization_masked": masked,
            "headers": headers
        }
    except Exception as e:
        logger.error("Failed to read request headers for debug: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read headers")