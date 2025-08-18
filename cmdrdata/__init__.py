"""
cmdrdata: Universal AI client tracking for any provider

This package provides a universal wrapper that can track ANY AI client
(OpenAI, Anthropic, Google, Cohere, HuggingFace, etc.) with zero code changes.

Key Features:
- Works with ANY AI client that returns usage information
- Auto-detects provider type (OpenAI, Anthropic, Google, etc.)
- 100% API compatibility - use wrapped clients exactly like originals
- Transparent usage tracking with customer attribution
- Non-blocking background tracking
- Thread-safe context management
- Comprehensive metadata support

Basic Usage:
    from cmdrdata import track_ai
    from openai import OpenAI

    # Wrap any AI client
    client = track_ai(
        OpenAI(api_key="..."),
        cmdrdata_api_key="cmd-..."
    )

    # Use exactly like the original - tracking happens automatically
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}],
        customer_id="customer-123"  # Optional metadata
    )

Multi-Provider Example:
    from cmdrdata import CmdrData
    from openai import OpenAI
    from anthropic import Anthropic
    import google.generativeai as genai

    # Track multiple providers with one SDK
    openai_client = CmdrData(
        client=OpenAI(api_key="..."),
        cmdrdata_api_key="cmd-..."
    )

    anthropic_client = CmdrData(
        client=Anthropic(api_key="..."),
        cmdrdata_api_key="cmd-..."
    )

    # Let CmdrData create the client
    gemini_tracker = CmdrData(
        client_class=genai.GenerativeModel,
        client_kwargs={"model_name": "gemini-pro"},
        cmdrdata_api_key="cmd-...",
        provider="google"
    )
"""

__version__ = "0.1.0"

from .client import CmdrData, track_ai
from .context import (
    clear_customer_context,
    customer_context,
    get_customer_context,
    metadata_context,
    set_customer_context,
)
from .exceptions import (
    CMDRDataError,
    ConfigurationError,
    NetworkError,
    TrackingError,
    ValidationError,
)

__all__ = [
    # Main client
    "CmdrData",
    "track_ai",
    # Context management
    "customer_context",
    "set_customer_context",
    "get_customer_context",
    "clear_customer_context",
    "metadata_context",
    # Exceptions
    "CMDRDataError",
    "ValidationError",
    "ConfigurationError",
    "NetworkError",
    "TrackingError",
]


def get_version() -> str:
    """Get the current version of cmdrdata"""
    return __version__
