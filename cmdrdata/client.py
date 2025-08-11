"""
Universal AI client wrapper for automatic usage tracking
"""

import os
import time
import threading
import logging
from typing import Any, Dict, Optional, Union, Callable
from datetime import datetime

from .tracker import UsageTracker
from .context import get_customer_context, get_metadata_context
from .exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)


class CmdrData:
    """
    Universal wrapper for any AI client with automatic usage tracking.
    
    This class wraps ANY AI client and automatically tracks usage to CmdrData
    without requiring any code changes. It works by intercepting method calls
    and extracting usage information from responses.
    
    Attributes:
        client: The wrapped AI client instance
        tracker: UsageTracker instance for sending events
        provider: Detected or specified provider name
        default_customer_id: Default customer ID for tracking
        default_metadata: Default metadata applied to all calls
    """
    
    def __init__(
        self,
        client: Optional[Any] = None,
        client_class: Optional[type] = None,
        client_kwargs: Optional[Dict] = None,
        cmdrdata_api_key: Optional[str] = None,
        cmdrdata_url: Optional[str] = None,
        provider: Optional[str] = None,
        auto_detect_provider: bool = True,
        customer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        disable_tracking: bool = False,
    ):
        """
        Initialize the universal tracker.
        
        Args:
            client: Existing AI client instance to wrap
            client_class: AI client class to instantiate
            client_kwargs: Arguments for client class instantiation
            cmdrdata_api_key: CmdrData API key (or set CMDRDATA_API_KEY env var)
            cmdrdata_url: CmdrData API endpoint (or set CMDRDATA_URL env var)
            provider: Provider name (openai, anthropic, google, etc.)
            auto_detect_provider: Automatically detect provider from client
            customer_id: Default customer ID for all requests
            metadata: Default metadata for all requests
            disable_tracking: Disable tracking (for testing)
        
        Raises:
            ConfigurationError: If configuration is invalid
            ValidationError: If required parameters are missing
        """
        # Create or wrap the client
        if client:
            self._client = client
        elif client_class and client_kwargs:
            self._client = client_class(**client_kwargs)
        else:
            raise ValidationError(
                "Must provide either 'client' or both 'client_class' and 'client_kwargs'"
            )
        
        # Get API key from parameter or environment
        api_key = cmdrdata_api_key or os.getenv("CMDRDATA_API_KEY")
        if not api_key and not disable_tracking:
            logger.warning(
                "No CmdrData API key provided. Tracking will be disabled. "
                "Set cmdrdata_api_key parameter or CMDRDATA_API_KEY environment variable."
            )
            disable_tracking = True
        
        # Initialize tracker
        self.tracker = UsageTracker(
            api_key=api_key,
            endpoint=cmdrdata_url or os.getenv("CMDRDATA_URL", "https://api.cmdrdata.ai/api/events"),
            disabled=disable_tracking
        )
        
        # Auto-detect or set provider
        if auto_detect_provider and not provider:
            self.provider = self.__detect_provider(self._client)
        else:
            self.provider = provider or "unknown"
        
        # Store defaults
        self.default_customer_id = customer_id or os.getenv("CMDRDATA_CUSTOMER_ID")
        self.default_metadata = metadata or {}
        
        # Tracking state
        self._tracking_enabled = not disable_tracking
        self._wrapped_attrs = {}
    
    def __detect_provider(self, client: Any) -> str:
        """
        Detect the AI provider from the client type.
        
        Args:
            client: The AI client instance
        
        Returns:
            Detected provider name
        """
        client_type = type(client).__name__.lower()
        module = type(client).__module__.lower()
        
        # Check module name first (most reliable)
        if "openai" in module:
            return "openai"
        elif "anthropic" in module:
            return "anthropic"
        elif "google" in module or "genai" in module or "generativeai" in module:
            return "google"
        elif "cohere" in module:
            return "cohere"
        elif "huggingface" in module or "transformers" in module:
            return "huggingface"
        elif "replicate" in module:
            return "replicate"
        elif "together" in module:
            return "together"
        elif "perplexity" in module:
            return "perplexity"
        
        # Check class name as fallback
        elif "openai" in client_type:
            return "openai"
        elif "anthropic" in client_type or "claude" in client_type:
            return "anthropic"
        elif "gemini" in client_type or "generative" in client_type:
            return "google"
        
        return "unknown"
    
    def __getattr__(self, name: str) -> Any:
        """
        Intercept attribute access to wrap nested objects and methods.
        
        This is the magic that makes the wrapper transparent - any attribute
        access is passed through to the wrapped client, with methods being
        wrapped for tracking.
        
        Args:
            name: Attribute name
        
        Returns:
            Wrapped attribute or method
        """
        # Check if we've already wrapped this attribute
        if name in self._wrapped_attrs:
            return self._wrapped_attrs[name]
        
        # Get the attribute from the wrapped client
        try:
            attr = getattr(self._client, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self._client).__name__}' object has no attribute '{name}'"
            )
        
        # If it's a method or callable, wrap it
        if callable(attr):
            wrapped = self.__wrap_method(attr, name)
            self._wrapped_attrs[name] = wrapped
            return wrapped
        
        # If it's an object with methods (like client.chat.completions)
        # wrap it recursively
        if hasattr(attr, "__dict__") and not isinstance(attr, (str, int, float, bool, list, dict)):
            proxy = CmdrDataProxy(attr, self, f"{name}")
            self._wrapped_attrs[name] = proxy
            return proxy
        
        # Return simple attributes as-is
        return attr
    
    def __wrap_method(self, method: Callable, method_name: str) -> Callable:
        """
        Wrap a method to track its usage.
        
        Args:
            method: The method to wrap
            method_name: Name of the method for tracking
        
        Returns:
            Wrapped method that tracks usage
        """
        def wrapped(*args, **kwargs):
            # Skip tracking if disabled
            if not self._tracking_enabled:
                return method(*args, **kwargs)
            
            # Extract CmdrData-specific parameters
            customer_id = kwargs.pop("customer_id", None) or get_customer_context() or self.default_customer_id
            extra_metadata = kwargs.pop("metadata", {})
            
            # Merge metadata (defaults -> context -> call-specific)
            metadata = {
                **self.default_metadata,
                **get_metadata_context(),
                **extra_metadata
            }
            
            # Track timing
            start_time = time.time()
            
            try:
                # Call the original method
                response = method(*args, **kwargs)
                
                # Track successful usage
                if self.__should_track_method(method_name):
                    try:
                        self.__track_usage(
                            method_name=method_name,
                            args=args,
                            kwargs=kwargs,
                            response=response,
                            customer_id=customer_id,
                            metadata=metadata,
                            start_time=start_time,
                            error_occurred=False
                        )
                    except Exception as track_error:
                        # Log but don't fail the API call
                        logger.debug(f"Failed to track usage: {track_error}")
                
                return response
                
            except Exception as e:
                # Track error
                if self.__should_track_method(method_name):
                    try:
                        self.__track_usage(
                            method_name=method_name,
                            args=args,
                            kwargs=kwargs,
                            response=None,
                            customer_id=customer_id,
                            metadata=metadata,
                            start_time=start_time,
                            error_occurred=True,
                            error_info={
                                "error_type": type(e).__name__,
                                "error_message": str(e)
                            }
                        )
                    except Exception as track_error:
                        # Log but don't fail the API call
                        logger.debug(f"Failed to track error: {track_error}")
                
                # Re-raise the exception
                raise
        
        # Preserve method attributes
        wrapped.__name__ = method.__name__ if hasattr(method, "__name__") else method_name
        wrapped.__doc__ = method.__doc__ if hasattr(method, "__doc__") else None
        
        return wrapped
    
    def __should_track_method(self, method_name: str) -> bool:
        """
        Determine if a method should be tracked.
        
        We track methods that typically generate billable usage.
        
        Args:
            method_name: Name of the method
        
        Returns:
            True if method should be tracked
        """
        # Common completion/generation methods
        completion_keywords = {
            "create", "generate", "complete", "chat", "invoke",
            "run", "predict", "send", "call", "stream",
            "embed", "encode", "transcribe", "translate",
            "synthesize", "analyze", "process"
        }
        
        method_lower = method_name.lower()
        
        # Check if method name contains any completion keywords
        return any(keyword in method_lower for keyword in completion_keywords)
    
    def __track_usage(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict,
        response: Any,
        customer_id: Optional[str],
        metadata: Dict[str, Any],
        start_time: float,
        error_occurred: bool = False,
        error_info: Optional[Dict[str, Any]] = None
    ):
        """
        Extract usage information and send to CmdrData.
        
        Args:
            method_name: Name of the called method
            args: Method arguments
            kwargs: Method keyword arguments
            response: Response from the method
            customer_id: Customer ID for tracking
            metadata: Metadata for the request
            start_time: Request start time
            error_occurred: Whether an error occurred
            error_info: Error details if applicable
        """
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Extract usage and model information
        usage_data = self.__extract_usage(response) if response else {}
        model = self.__extract_model(args, kwargs, response)
        
        # Build tracking event
        event = {
            "customer_id": customer_id,
            "model": model,
            "provider": self.provider,
            "metadata": metadata,
            "request_duration_ms": duration_ms,
            **usage_data
        }
        
        if error_occurred and error_info:
            event["error_occurred"] = True
            event.update(error_info)
        
        # Send event in background
        self.tracker.track_usage_background(**event)
    
    def __extract_usage(self, response: Any) -> Dict[str, Any]:
        """
        Extract token usage from various response formats.
        
        This method handles different response formats from various providers.
        
        Args:
            response: Response object from AI provider
        
        Returns:
            Dictionary with input_tokens, output_tokens, total_tokens
        """
        usage_data = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
        
        if not response:
            return usage_data
        
        # Get usage from dict or object attribute
        usage = None
        if isinstance(response, dict) and "usage" in response:
            usage = response["usage"]
        elif hasattr(response, "usage"):
            usage = response.usage
        
        # Process usage if found
        if usage:
            # OpenAI pattern (prompt_tokens, completion_tokens)
            if hasattr(usage, "prompt_tokens"):
                usage_data["input_tokens"] = getattr(usage, "prompt_tokens", 0)
                if hasattr(usage, "completion_tokens"):
                    usage_data["output_tokens"] = getattr(usage, "completion_tokens", 0)
                if hasattr(usage, "total_tokens"):
                    usage_data["total_tokens"] = getattr(usage, "total_tokens", 0)
            
            # Anthropic pattern (input_tokens, output_tokens)
            elif hasattr(usage, "input_tokens"):
                usage_data["input_tokens"] = getattr(usage, "input_tokens", 0)
                if hasattr(usage, "output_tokens"):
                    usage_data["output_tokens"] = getattr(usage, "output_tokens", 0)
        
        # Google/Gemini pattern
        elif hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            usage_data["input_tokens"] = getattr(usage, "prompt_token_count", 0)
            usage_data["output_tokens"] = getattr(usage, "candidates_token_count", 0)
            usage_data["total_tokens"] = getattr(usage, "total_token_count", 0)
        
        # Cohere pattern
        elif hasattr(response, "meta"):
            meta = response.meta
            if hasattr(meta, "billed_units"):
                units = meta.billed_units
                usage_data["input_tokens"] = getattr(units, "input_tokens", 0)
                usage_data["output_tokens"] = getattr(units, "output_tokens", 0)
        
        # Calculate total if not provided
        if usage_data["total_tokens"] == 0:
            usage_data["total_tokens"] = usage_data["input_tokens"] + usage_data["output_tokens"]
        
        return usage_data
    
    def __extract_model(self, args: tuple, kwargs: dict, response: Any) -> str:
        """
        Extract model name from various sources.
        
        Args:
            args: Method arguments
            kwargs: Method keyword arguments
            response: Response object
        
        Returns:
            Model name or "unknown"
        """
        # Check kwargs first (most common)
        if "model" in kwargs:
            return str(kwargs["model"])
        
        # Check response for model info (as attribute)
        if response:
            if hasattr(response, "model"):
                return str(response.model)
            elif hasattr(response, "model_name"):
                return str(response.model_name)
            # Check if model is in response dict
            elif isinstance(response, dict) and "model" in response:
                return str(response["model"])
        
        # Check if first arg looks like a model name (not a prompt)
        # Model names typically contain dashes or are short identifiers
        if args and isinstance(args[0], str):
            first_arg = args[0]
            # Check if it looks like a model name (contains dash or is short)
            if "-" in first_arg or len(first_arg) < 30:
                # But not if it looks like JSON or a sentence
                if not first_arg.startswith("{") and not " " in first_arg:
                    return first_arg
        
        return "unknown"


class CmdrDataProxy:
    """
    Proxy object for nested attributes (like client.chat.completions).
    
    This allows us to maintain the chain of attribute access while
    still intercepting method calls for tracking.
    """
    
    def __init__(self, wrapped_obj: Any, parent: CmdrData, path: str):
        """
        Initialize proxy for nested object.
        
        Args:
            wrapped_obj: The object being wrapped
            parent: Parent CmdrData instance
            path: Dot-separated path to this object
        """
        self._wrapped = wrapped_obj
        self._parent = parent
        self._path = path
    
    def __getattr__(self, name: str) -> Any:
        """
        Recursively wrap nested attributes.
        
        Args:
            name: Attribute name
        
        Returns:
            Wrapped attribute or method
        """
        attr = getattr(self._wrapped, name)
        full_path = f"{self._path}.{name}"
        
        if callable(attr):
            return self._parent._CmdrData__wrap_method(attr, full_path)
        
        # Continue wrapping nested objects
        if hasattr(attr, "__dict__") and not isinstance(attr, (str, int, float, bool, list, dict)):
            return CmdrDataProxy(attr, self._parent, full_path)
        
        return attr
    
    def __call__(self, *args, **kwargs):
        """
        Handle direct calls on proxy objects.
        
        Some APIs have callable objects that aren't methods.
        """
        if callable(self._wrapped):
            wrapped_call = self._parent._CmdrData__wrap_method(self._wrapped, self._path)
            return wrapped_call(*args, **kwargs)
        else:
            raise TypeError(f"'{type(self._wrapped).__name__}' object is not callable")


def track_ai(
    client: Any,
    cmdrdata_api_key: Optional[str] = None,
    **kwargs
) -> CmdrData:
    """
    Convenience function to quickly wrap any AI client.
    
    This is the simplest way to add tracking to an existing AI client.
    
    Args:
        client: Any AI client instance
        cmdrdata_api_key: CmdrData API key
        **kwargs: Additional arguments for CmdrData
    
    Returns:
        Wrapped client with automatic tracking
    
    Example:
        from cmdrdata_universal import track_ai
        from openai import OpenAI
        
        client = track_ai(OpenAI(api_key="..."), cmdrdata_api_key="cmd-...")
        response = client.chat.completions.create(...)  # Automatically tracked
    """
    return CmdrData(
        client=client,
        cmdrdata_api_key=cmdrdata_api_key,
        **kwargs
    )