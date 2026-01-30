"""
Langfuse tracing utilities for interview observability.
"""
import os
from typing import Optional
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

# Initialize Langfuse client (will be None if credentials not provided)
_langfuse_client: Optional[Langfuse] = None

def initialize_langfuse() -> Optional[Langfuse]:
    """
    Initialize Langfuse client with credentials from environment.
    Returns None if credentials are not configured.
    """
    global _langfuse_client

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print("⚠️  Langfuse credentials not found. Tracing disabled.")
        return None

    try:
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        print("✅ Langfuse tracing enabled")
        return _langfuse_client
    except Exception as e:
        print(f"⚠️  Failed to initialize Langfuse: {e}")
        return None

def get_langfuse_client() -> Optional[Langfuse]:
    """Get the initialized Langfuse client."""
    return _langfuse_client

def is_tracing_enabled() -> bool:
    """Check if Langfuse tracing is enabled."""
    return _langfuse_client is not None
