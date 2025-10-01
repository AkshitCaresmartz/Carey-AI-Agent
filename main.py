"""PyMCP FastAPI application."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pymcp.chat_api import router as chat_router
from pymcp.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()
    logger.info(f"Starting PyMCP Chat API on {settings.api_host}:{settings.api_port}")
    
    yield
    
    logger.info("Shutting down PyMCP Chat API")


# Create FastAPI application
app = FastAPI(
    title="PyMCP Chat API",
    description="Python MCP Client with Chat API integration",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:6274",  # Angular dev server
        "http://localhost:4200",  # Angular dev server
        "http://localhost:3000",  # Alternative frontend port
        "http://127.0.0.1:6274",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if get_settings().debug else "An error occurred"
        }
    )


# Include routers
app.include_router(chat_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PyMCP Chat API",
        "version": "0.1.0",
        "description": "Python MCP Client with Chat API integration",
        "endpoints": {
            "chat": "/chat/chat - Chat with GROQ LLM",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "PyMCP Chat API"
    }


def main():
    """Main entry point."""
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )


if __name__ == "__main__":
    main()
