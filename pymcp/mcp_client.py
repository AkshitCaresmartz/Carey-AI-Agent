"""MCP Client implementation."""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import (
    CallToolResult,
    InitializeResult,
    ListToolsResult,
    Tool,
)

from .config import Settings

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP client that communicates with MCP servers."""

    def __init__(
        self,
        server_url: str,
        settings: Settings,
        access_token: Optional[str] = None,
    ):
        """Initialize MCP client.
        
        Args:
            server_url: MCP server endpoint URL
            settings: Application settings
            access_token: Optional access token for server authentication
        """
        self.server_url = server_url.rstrip('/')
        self.settings = settings
        self.access_token = access_token

    @asynccontextmanager
    async def _get_client_session(self) -> AsyncGenerator[ClientSession, None]:
        """Get MCP client session with optional authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "PyMCP-Client/1.0.0"
        }
        
        # Add authorization header if access token is provided
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
            logger.info("Added Bearer token to MCP request headers")
        
        timeout = timedelta(seconds=30)
        sse_timeout = timedelta(seconds=300)
        
        logger.info(f"Connecting to MCP server: {self.server_url}")
        
        try:
            async with streamablehttp_client(
                url=self.server_url,
                headers=headers,
                timeout=timeout,
                sse_read_timeout=sse_timeout,
            ) as (read_stream, write_stream, get_session_id):
                
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize the session
                    logger.info("Initializing MCP session")
                    init_result = await session.initialize()
                    logger.info(f"MCP session initialized: {init_result.serverInfo.name}")
                    
                    yield session
        except Exception as e:
            # Enhance error information for better debugging
            error_msg = str(e)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                raise Exception(f"Unauthorized access to MCP server - invalid or expired token: {e}")
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                raise Exception(f"Access forbidden by MCP server: {e}")
            elif "404" in error_msg:
                raise Exception(f"MCP server not found: {e}")
            elif "timeout" in error_msg.lower():
                raise Exception(f"MCP server connection timeout: {e}")
            else:
                raise Exception(f"MCP server connection failed: {e}")

    async def initialize(self) -> InitializeResult:
        """Initialize connection to MCP server."""
        async with self._get_client_session() as session:
            return await session.initialize()

    async def list_tools(self) -> List[Tool]:
        """List available tools from MCP server."""
        try:
            async with self._get_client_session() as session:
                result: ListToolsResult = await session.list_tools()
                return result.tools
        except Exception as e:
            # Re-raise with more context for better error handling
            logger.error(f"Failed to list tools from MCP server: {e}")
            if "401" in str(e) or "unauthorized" in str(e).lower():
                raise Exception(f"Unauthorized access to MCP server: {e}")
            elif "403" in str(e) or "forbidden" in str(e).lower():
                raise Exception(f"Access forbidden by MCP server: {e}")
            else:
                raise Exception(f"MCP server connection failed: {e}")

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Call a tool on the MCP server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            async with self._get_client_session() as session:
                return await session.call_tool(name, arguments)
        except Exception as e:
            # Re-raise with more context for better error handling
            logger.error(f"Failed to call tool '{name}' on MCP server: {e}")
            if "401" in str(e) or "unauthorized" in str(e).lower():
                raise Exception(f"Unauthorized access to MCP server for tool '{name}': {e}")
            elif "403" in str(e) or "forbidden" in str(e).lower():
                raise Exception(f"Access forbidden by MCP server for tool '{name}': {e}")
            else:
                raise Exception(f"MCP server tool execution failed for '{name}': {e}")

    async def ping(self) -> bool:
        """Ping the MCP server to check connectivity.
        
        Returns:
            True if server responds to ping
        """
        try:
            async with self._get_client_session() as session:
                await session.send_ping()
                return True
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            return False