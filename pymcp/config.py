"""Configuration management for PyMCP."""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # GROQ LLM Configuration
    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY")
    groq_model: str = Field(default="meta-llama/llama-4-maverick-17b-128e-instruct", env="GROQ_MODEL")
    
    # MCP Server Configuration
    mcp_server_url: str = Field(default="https://aimcpserver.caresmartz360.net/mcp", env="MCP_SERVER_URL")
    
    # Web API Configuration
    api_host: str = Field(default="localhost", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()