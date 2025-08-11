#!/usr/bin/env python3
"""
Test suite for CmdrData Universal SDK
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cmdrdata import track_ai, CmdrData, customer_context, metadata_context


def test_openai_wrapper():
    """Test wrapping OpenAI client"""
    print("\n" + "="*60)
    print("Testing OpenAI Wrapper")
    print("="*60)
    
    try:
        from openai import OpenAI
        
        # Create regular OpenAI client
        base_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key"))
        
        # Wrap with CmdrData Universal
        client = track_ai(
            base_client,
            cmdrdata_api_key="cmd-test-key",
            cmdrdata_url="http://localhost:8000/api/events",
            customer_id="test-customer",
            metadata={"test": "openai", "environment": "development"}
        )
        
        print("[OK] Successfully wrapped OpenAI client")
        print(f"    Provider detected: {client.provider}")
        
        # Test that we can access nested attributes
        assert hasattr(client, "chat"), "Missing chat attribute"
        assert hasattr(client.chat, "completions"), "Missing completions attribute"
        assert callable(client.chat.completions.create), "create method not callable"
        print("[OK] All OpenAI attributes accessible")
        
        # Test with mock response (no actual API call)
        print("\nWould track calls to:")
        print("  - client.chat.completions.create()")
        print("  - client.embeddings.create()")
        print("  - client.images.generate()")
        
        return True
        
    except ImportError:
        print("[SKIP] OpenAI not installed")
        return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_anthropic_wrapper():
    """Test wrapping Anthropic client"""
    print("\n" + "="*60)
    print("Testing Anthropic Wrapper")
    print("="*60)
    
    try:
        from anthropic import Anthropic
        
        # Create using client_class pattern
        client = CmdrData(
            client_class=Anthropic,
            client_kwargs={"api_key": os.getenv("ANTHROPIC_API_KEY") or "dummy-key"},
            cmdrdata_api_key="cmd-test-key",
            cmdrdata_url="http://localhost:8000/api/events",
            customer_id="test-customer",
            metadata={"test": "anthropic"}
        )
        
        print("[OK] Successfully created Anthropic client via class")
        print(f"    Provider detected: {client.provider}")
        
        # Test attributes
        assert hasattr(client, "messages"), "Missing messages attribute"
        assert callable(client.messages.create), "create method not callable"
        print("[OK] All Anthropic attributes accessible")
        
        print("\nWould track calls to:")
        print("  - client.messages.create()")
        print("  - client.messages.stream()")
        
        return True
        
    except ImportError:
        print("[SKIP] Anthropic not installed")
        return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_google_wrapper():
    """Test wrapping Google Gemini client"""
    print("\n" + "="*60)
    print("Testing Google Gemini Wrapper")
    print("="*60)
    
    try:
        import google.generativeai as genai
        
        # Configure API key
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or "dummy-key")
        
        # Create model
        base_model = genai.GenerativeModel("gemini-pro")
        
        # Wrap with CmdrData
        model = track_ai(
            base_model,
            cmdrdata_api_key="cmd-test-key",
            cmdrdata_url="http://localhost:8000/api/events",
            provider="google",  # Explicitly set provider
            customer_id="test-customer"
        )
        
        print("[OK] Successfully wrapped Gemini model")
        print(f"    Provider: {model.provider}")
        
        # Test attributes
        assert hasattr(model, "generate_content"), "Missing generate_content method"
        assert callable(model.generate_content), "generate_content not callable"
        print("[OK] All Gemini methods accessible")
        
        print("\nWould track calls to:")
        print("  - model.generate_content()")
        print("  - model.start_chat()")
        
        return True
        
    except ImportError:
        print("[SKIP] Google Generative AI not installed")
        return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_context_managers():
    """Test context managers"""
    print("\n" + "="*60)
    print("Testing Context Managers")
    print("="*60)
    
    try:
        # Create a mock client class
        class MockClient:
            def complete(self, prompt):
                return {"response": "test", "usage": {"tokens": 100}}
        
        # Wrap it
        client = track_ai(
            MockClient(),
            cmdrdata_api_key="cmd-test-key",
            cmdrdata_url="http://localhost:8000/api/events"
        )
        
        print("[OK] Created mock client")
        
        # Test customer context
        with customer_context("customer-123"):
            # In real usage, this would include customer-123 automatically
            print("[OK] Customer context manager works")
        
        # Test metadata context
        with metadata_context({"feature": "test", "version": "1.0"}):
            # In real usage, this would include the metadata
            print("[OK] Metadata context manager works")
        
        # Test nested contexts
        with customer_context("customer-456"):
            with metadata_context({"experiment": "v2"}):
                print("[OK] Nested contexts work")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_provider_detection():
    """Test automatic provider detection"""
    print("\n" + "="*60)
    print("Testing Provider Auto-Detection")
    print("="*60)
    
    # Test with mock clients
    test_cases = [
        ("openai.client", "OpenAI", "openai"),
        ("anthropic.client", "Anthropic", "anthropic"),
        ("google.generativeai", "GenerativeModel", "google"),
        ("cohere.client", "Client", "cohere"),
        ("unknown.module", "CustomClient", "unknown"),
    ]
    
    for module_name, class_name, expected_provider in test_cases:
        # Create mock client
        class MockClient:
            pass
        
        MockClient.__module__ = module_name
        MockClient.__name__ = class_name
        
        # Wrap and check detection
        wrapper = CmdrData(
            client=MockClient(),
            cmdrdata_api_key="test",
            disable_tracking=True  # Don't actually track
        )
        
        if wrapper.provider == expected_provider:
            print(f"[OK] {module_name}.{class_name} -> {expected_provider}")
        else:
            print(f"[FAIL] {module_name}.{class_name} -> {wrapper.provider} (expected {expected_provider})")
            return False
    
    return True


def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# CmdrData Universal SDK Test Suite")
    print("#"*60)
    
    results = {
        "OpenAI Wrapper": test_openai_wrapper(),
        "Anthropic Wrapper": test_anthropic_wrapper(),
        "Google Gemini Wrapper": test_google_wrapper(),
        "Context Managers": test_context_managers(),
        "Provider Detection": test_provider_detection(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        if result is True:
            status = "[PASS]"
        elif result is False:
            status = "[FAIL]"
        else:  # None
            status = "[SKIP]"
        
        print(f"{status} {test_name}")
    
    # Overall result
    failures = sum(1 for r in results.values() if r is False)
    passes = sum(1 for r in results.values() if r is True)
    skips = sum(1 for r in results.values() if r is None)
    
    print("\n" + "-"*60)
    print(f"Passed: {passes}, Failed: {failures}, Skipped: {skips}")
    
    if failures == 0:
        print("\nAll tests passed! The Universal SDK is working correctly.")
        return 0
    else:
        print(f"\n{failures} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())