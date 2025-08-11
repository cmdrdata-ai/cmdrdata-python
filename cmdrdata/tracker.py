"""
Usage tracking implementation for CmdrData SDK
"""

import json
import logging
import threading
from typing import Any, Dict, Optional
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class UsageTracker:
    """
    Handles sending usage events to CmdrData API.
    
    This class manages the communication with CmdrData's tracking API,
    including authentication, error handling, and background processing.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://api.cmdrdata.ai/api/events",
        timeout: int = 5,
        max_retries: int = 3,
        disabled: bool = False
    ):
        """
        Initialize the usage tracker.
        
        Args:
            api_key: CmdrData API key for authentication
            endpoint: API endpoint for sending events
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            disabled: Whether tracking is disabled
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.disabled = disabled
        
        if not self.api_key and not self.disabled:
            logger.warning("No API key provided. Usage tracking disabled.")
            self.disabled = True
    
    def track_usage(
        self,
        customer_id: Optional[str],
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        provider: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        request_duration_ms: Optional[int] = None,
        error_occurred: Optional[bool] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Send a usage event to CmdrData.
        
        Args:
            customer_id: Customer identifier
            model: Model name/identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            total_tokens: Total tokens used
            provider: AI provider name
            metadata: Additional metadata
            timestamp: Event timestamp
            request_duration_ms: Request duration in milliseconds
            error_occurred: Whether an error occurred
            error_type: Type of error if occurred
            error_message: Error message if occurred
            **kwargs: Additional fields to include
        
        Returns:
            True if event was sent successfully
        """
        if self.disabled:
            return False
        
        # Build event payload
        event = {
            "customer_id": customer_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "provider": provider,
            "metadata": metadata or {},
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            **kwargs
        }
        
        # Add optional fields
        if request_duration_ms is not None:
            event["request_duration_ms"] = request_duration_ms
        
        if error_occurred:
            event["error_occurred"] = True
            if error_type:
                event["error_type"] = error_type
            if error_message:
                event["error_message"] = error_message
        
        # Send event
        return self._send_event(event)
    
    def track_usage_background(self, **kwargs) -> None:
        """
        Send a usage event in a background thread.
        
        This method returns immediately and sends the event asynchronously.
        
        Args:
            **kwargs: Arguments for track_usage
        """
        if self.disabled:
            return
        
        thread = threading.Thread(
            target=self.track_usage,
            kwargs=kwargs,
            daemon=True
        )
        thread.start()
    
    def _send_event(self, event: Dict[str, Any]) -> bool:
        """
        Send an event to the CmdrData API.
        
        Args:
            event: Event payload
        
        Returns:
            True if successful
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "cmdrdata/0.1.0"
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    json=event,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    logger.debug(f"Successfully tracked usage for customer {event.get('customer_id')}")
                    return True
                elif response.status_code == 401:
                    logger.error("Invalid CmdrData API key")
                    self.disabled = True  # Disable further attempts
                    return False
                elif response.status_code >= 500:
                    logger.warning(f"CmdrData API error {response.status_code}, attempt {attempt + 1}/{self.max_retries}")
                    continue
                else:
                    logger.error(f"Failed to track usage: {response.status_code} - {response.text}")
                    return False
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout tracking usage, attempt {attempt + 1}/{self.max_retries}")
                continue
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error tracking usage, attempt {attempt + 1}/{self.max_retries}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error tracking usage: {e}")
                return False
        
        logger.error(f"Failed to track usage after {self.max_retries} attempts")
        return False