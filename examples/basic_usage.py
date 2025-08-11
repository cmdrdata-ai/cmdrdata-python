#!/usr/bin/env python3
"""
Basic usage examples for CmdrData universal SDK
"""

import os
from cmdrdata import track_ai, CmdrData, customer_context, metadata_context


def example_openai():
    """Example: Wrapping OpenAI client"""
    try:
        from openai import OpenAI
        
        # Method 1: Using track_ai convenience function
        client = track_ai(
            OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
            cmdrdata_api_key=os.getenv("CMDRDATA_API_KEY")
        )
        
        # Method 2: Using CmdrData class directly
        client = CmdrData(
            client=OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
            cmdrdata_api_key=os.getenv("CMDRDATA_API_KEY"),
            customer_id="default-customer",
            metadata={"environment": "production", "version": "1.0"}
        )
        
        # Use exactly like normal OpenAI client
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}],
            customer_id="customer-123",  # Track by customer
            metadata={"feature": "greeting", "user_tier": "premium"}  # Custom metadata
        )
        
        print("OpenAI response tracked successfully!")
        
    except ImportError:
        print("OpenAI not installed. Install with: pip install openai")


def example_anthropic():
    """Example: Wrapping Anthropic client"""
    try:
        from anthropic import Anthropic
        
        # Create wrapped client using class instantiation
        client = CmdrData(
            client_class=Anthropic,
            client_kwargs={"api_key": os.getenv("ANTHROPIC_API_KEY")},
            cmdrdata_api_key=os.getenv("CMDRDATA_API_KEY")
        )
        
        # Use exactly like normal Anthropic client
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Hello Claude!"}],
            max_tokens=100,
            customer_id="customer-456",
            metadata={"feature": "chat", "department": "support"}
        )
        
        print("Anthropic response tracked successfully!")
        
    except ImportError:
        print("Anthropic not installed. Install with: pip install anthropic")


def example_context_managers():
    """Example: Using context managers for customer and metadata"""
    
    # Mock client for demonstration
    class MockAIClient:
        def generate(self, prompt):
            return {"text": "Generated response", "tokens": 100}
    
    # Wrap the mock client
    client = track_ai(
        MockAIClient(),
        cmdrdata_api_key=os.getenv("CMDRDATA_API_KEY", "test-key")
    )
    
    # Method 1: Customer context
    with customer_context("customer-789"):
        # All calls within this context will be tracked to customer-789
        response = client.generate("Hello world")
        print("Tracked with customer context")
    
    # Method 2: Metadata context
    with metadata_context({"experiment": "v2", "cohort": "control"}):
        # All calls within this context will include this metadata
        response = client.generate("Test prompt")
        print("Tracked with metadata context")
    
    # Method 3: Nested contexts
    with customer_context("customer-999"):
        with metadata_context({"feature": "advanced", "priority": "high"}):
            # Both customer and metadata are applied
            response = client.generate("Complex prompt")
            print("Tracked with nested contexts")


def example_multiple_providers():
    """Example: Tracking multiple AI providers simultaneously"""
    
    # You can track multiple providers at once
    clients = {}
    
    # Mock different AI providers
    class OpenAIMock:
        def __init__(self, api_key): pass
        def complete(self, prompt): return {"text": "OpenAI response"}
    
    class AnthropicMock:
        def __init__(self, api_key): pass
        def complete(self, prompt): return {"text": "Anthropic response"}
    
    class GoogleMock:
        def __init__(self, api_key): pass
        def complete(self, prompt): return {"text": "Google response"}
    
    # Set module names for provider detection
    OpenAIMock.__module__ = "openai.client"
    AnthropicMock.__module__ = "anthropic.client"
    GoogleMock.__module__ = "google.generativeai"
    
    # Wrap all providers with the same CmdrData API key
    cmdrdata_key = os.getenv("CMDRDATA_API_KEY", "test-key")
    
    clients["openai"] = track_ai(
        OpenAIMock(api_key="key1"),
        cmdrdata_api_key=cmdrdata_key
    )
    
    clients["anthropic"] = track_ai(
        AnthropicMock(api_key="key2"),
        cmdrdata_api_key=cmdrdata_key
    )
    
    clients["google"] = track_ai(
        GoogleMock(api_key="key3"),
        cmdrdata_api_key=cmdrdata_key
    )
    
    # Use any client - all tracked to the same CmdrData account
    for provider, client in clients.items():
        response = client.complete(f"Hello from {provider}")
        print(f"Tracked {provider} usage")


def example_custom_provider():
    """Example: Tracking a custom AI provider"""
    
    class CustomAIProvider:
        """Your custom AI provider that returns usage info"""
        
        def generate_text(self, prompt, model="custom-model-v1"):
            # Simulate API response with usage data
            return {
                "text": "Generated text response",
                "model": model,
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": 50,
                    "total_tokens": len(prompt.split()) + 50
                }
            }
    
    # Wrap your custom provider
    client = CmdrData(
        client=CustomAIProvider(),
        cmdrdata_api_key=os.getenv("CMDRDATA_API_KEY", "test-key"),
        provider="custom-ai",  # Specify provider name
        customer_id="customer-001",
        metadata={"source": "custom"}
    )
    
    # Use your custom provider - usage automatically tracked
    response = client.generate_text(
        "Generate a story about space",
        model="custom-model-v2"
    )
    
    print("Custom provider tracked successfully!")


if __name__ == "__main__":
    print("CmdrData SDK Examples")
    print("=" * 50)
    
    print("\n1. OpenAI Example:")
    example_openai()
    
    print("\n2. Anthropic Example:")
    example_anthropic()
    
    print("\n3. Context Managers Example:")
    example_context_managers()
    
    print("\n4. Multiple Providers Example:")
    example_multiple_providers()
    
    print("\n5. Custom Provider Example:")
    example_custom_provider()
    
    print("\n" + "=" * 50)
    print("Examples completed!")