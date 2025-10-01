"""GROQ LLM client with MCP integration."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq
from pydantic import BaseModel

from .mcp_client import MCPClient

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # "user", "assistant"
    content: str


class ConversationManager:
    """Manages conversation state and message history."""

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.conversation_id: Optional[str] = None

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add an assistant message to the conversation."""
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)

    def add_tool_message(self, tool_call_id: str, content: str) -> None:
        """Add a tool execution result to the conversation."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the conversation."""
        return self.messages.copy()

    def clear_messages(self) -> None:
        """Clear all messages from the conversation."""
        self.messages.clear()


class GroqClient:
    """GROQ LLM client with optional MCP tools integration."""

    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/llama-4-maverick-17b-128e-instruct",
        mcp_client: Optional[MCPClient] = None
    ):
        """Initialize GROQ client.
        
        Args:
            api_key: GROQ API key
            model: GROQ model to use
            mcp_client: Optional MCP client for tool integration
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        self.mcp_client = mcp_client
        self.tools: List[Dict[str, Any]] = []

    async def initialize_tools(self) -> None:
        """Initialize MCP tools if MCP client is available."""
        if not self.mcp_client:
            logger.info("No MCP client provided, running in simple LLM mode")
            return

        try:
            mcp_tools = await self.mcp_client.list_tools()
            
            # Convert MCP tools to OpenAI format
            for tool in mcp_tools:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                self.tools.append(openai_tool)
            
            logger.info(f"Initialized {len(self.tools)} MCP tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}")
            raise

    async def chat_completion(
        self,
        conversation_manager: ConversationManager,
        message: str
    ) -> Tuple[str, bool, Optional[List[Dict[str, Any]]]]:
        """Generate chat completion with optional tool usage.
        
        Args:
            conversation_manager: Conversation state manager
            message: User message
            
        Returns:
            Tuple of (response_content, used_tools, tool_calls_info)
        """
        conversation_manager.add_user_message(message)
        
        # If no tools available, use simple completion
        if not self.tools:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation_manager.get_messages(),
                temperature=0.7,
                max_tokens=1000
            )
            response_content = response.choices[0].message.content or "I don't have a response to that."
            conversation_manager.add_assistant_message(response_content)
            return response_content, False, None

        # Use tools if available
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation_manager.get_messages(),
                tools=self.tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000
            )

            response_message = response.choices[0].message
            tool_calls_info = []
            used_tools = False

            # Handle tool calls if present
            if response_message.tool_calls:
                used_tools = True
                conversation_manager.add_assistant_message(
                    response_message.content or "",
                    [{"id": tc.id, "function": tc.function.name, "arguments": tc.function.arguments} 
                     for tc in response_message.tool_calls]
                )

                # Execute each tool call
                for tool_call in response_message.tool_calls:
                    try:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        logger.info(f"Calling MCP tool: {function_name} with args: {function_args}")
                        
                        # Call the MCP tool
                        result = await self.mcp_client.call_tool(function_name, function_args)
                        
                        # Extract content from the result
                        if hasattr(result, 'content') and result.content:
                            tool_result = str(result.content[0].text if result.content[0].text else result.content[0])
                        else:
                            tool_result = str(result)
                        
                        conversation_manager.add_tool_message(tool_call.id, tool_result)
                        
                        tool_calls_info.append({
                            "function": function_name,
                            "arguments": function_args,
                            "result": tool_result
                        })
                        
                    except Exception as e:
                        error_msg = f"Tool execution failed: {str(e)}"
                        logger.error(f"Tool {function_name} failed: {e}")
                        conversation_manager.add_tool_message(tool_call.id, error_msg)
                        
                        tool_calls_info.append({
                            "function": function_name,
                            "arguments": function_args,
                            "result": error_msg,
                            "error": True
                        })

                # Generate final response after tool execution
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=conversation_manager.get_messages(),
                    temperature=0.7,
                    max_tokens=1000
                )
                
                final_content = final_response.choices[0].message.content or "I processed your request."
                conversation_manager.add_assistant_message(final_content)
                
                return final_content, used_tools, tool_calls_info

            else:
                # No tool calls, just add the response
                response_content = response_message.content or "I don't have a response to that."
                conversation_manager.add_assistant_message(response_content)
                return response_content, False, None

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise