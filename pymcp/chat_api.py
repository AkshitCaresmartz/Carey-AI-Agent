"""Chat API implementation."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from .mcp_client import MCPClient
from .config import Settings, get_settings
from .groq_client import GroqClient, ConversationManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# In-memory conversation storage (in production, use Redis or database)
conversations: Dict[str, ConversationManager] = {}


@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Chat API is running"}


class ChatRequest(BaseModel):
    """Request model for chat completion."""
    message: str = Field(..., description="User message to send to the LLM")
    groq_api_key: Optional[str] = Field(None, description="GROQ API key for authentication (optional if set in environment)")
    access_token: Optional[str] = Field(None, description="Access token for MCP server authentication")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    model: Optional[str] = Field(None, description="GROQ model to use (optional if set in environment)")


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


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Generate a chat completion using GROQ LLM with intelligent MCP tool integration.
    
    Flow:
    1. Try to initialize MCP session with access_token
    2. If MCP fails → fallback to simple LLM response
    3. If MCP succeeds → intelligent tool detection (use tools when needed, LLM otherwise)
    """
    logger.info(f"Chat request received: {request.message[:50]}...")
    try:
        # Get or create conversation manager
        conversation_id = request.conversation_id or f"conv_{len(conversations)}"
        logger.info(f"Using conversation_id: {conversation_id}")
        
        if conversation_id not in conversations:
            conversations[conversation_id] = ConversationManager()
            logger.info(f"Created new conversation manager for {conversation_id}")
        
        conversation_manager = conversations[conversation_id]
        
        # Initialize GROQ client first
        groq_api_key = request.groq_api_key or settings.groq_api_key
        groq_model = request.model or settings.groq_model
        
        if not groq_api_key:
            raise HTTPException(
                status_code=400,
                detail="GROQ API key is required. Provide it in request or set GROQ_API_KEY environment variable."
            )
        
        # Try to initialize MCP session first
        mcp_client = None
        use_mcp_tools = False
        
        if request.access_token:
            logger.info("Access token provided - initializing MCP session")
            try:
                # Create MCP client
                mcp_client = MCPClient(
                    server_url=settings.mcp_server_url,
                    access_token=request.access_token,
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
                        access_token=request.access_token,
                        settings=settings
                    )
                    groq_client = GroqClient(
                        api_key=groq_api_key,
                        model=groq_model,
                        mcp_client=mcp_client
                    )
                    use_mcp_tools = True
        else:
            logger.info("No access token provided, using simple LLM mode")
        
        # Create GROQ client with or without MCP
        if not use_mcp_tools:
            groq_client = GroqClient(
                api_key=groq_api_key,
                model=groq_model,
                mcp_client=None
            )
            logger.info("GROQ client created in simple LLM mode")
        
        logger.info(f"GROQ client ready - MCP tools: {use_mcp_tools}")
        
        # Generate response with intelligent tool usage
        logger.info("Generating response...")
        if use_mcp_tools:
            # Use MCP tools
            try:
                response_content, used_tools, tool_calls_info = await groq_client.chat_completion(
                    conversation_manager, 
                    request.message
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
            conversation_manager.add_user_message(request.message)
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