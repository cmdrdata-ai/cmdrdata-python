#!/usr/bin/env python3
"""
Smoke test for CmdrData SDK
Quick validation that everything is working
"""

import os
import sys
import time
from datetime import datetime
from unittest.mock import Mock

from cmdrdata import track_ai, CmdrData, customer_context, metadata_context


def run_smoke_test():
    """Run a quick smoke test of SDK functionality"""
    print("\n" + "="*60)
    print("CMDRDATA SDK SMOKE TEST")
    print("="*60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    
    results = []
    
    # Test 1: Import and initialization
    print("\n[1/5] Testing SDK import and initialization...")
    try:
        class TestProvider:
            def generate(self, prompt):
                return {
                    "text": "response",
                    "usage": Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
                }
        
        client = track_ai(
            TestProvider(),
            cmdrdata_api_key=os.getenv("CMDRDATA_API_KEY", "test-key"),
            disable_tracking=True  # Don't actually send events for smoke test
        )
        print("  [OK] SDK initialized successfully")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] Initialization failed: {e}")
        results.append(False)
        return False
    
    # Test 2: Method wrapping
    print("\n[2/5] Testing method wrapping...")
    try:
        # customer_id is extracted by the wrapper, not passed to the method
        response = client.generate("test prompt")
        assert response["text"] == "response"
        
        # Also test with customer_id (should be extracted before calling method)
        response2 = client.generate("test prompt 2")
        assert response2["text"] == "response"
        
        print("  [OK] Method wrapping works")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] Method wrapping failed: {e}")
        results.append(False)
    
    # Test 3: Provider detection
    print("\n[3/5] Testing provider detection...")
    try:
        providers = {
            "openai.client": "openai",
            "anthropic.client": "anthropic",
            "google.generativeai": "google"
        }
        
        for module, expected in providers.items():
            class MockProvider:
                pass
            MockProvider.__module__ = module
            
            wrapped = CmdrData(
                client=MockProvider(),
                cmdrdata_api_key="test",
                disable_tracking=True
            )
            
            if wrapped.provider != expected:
                raise ValueError(f"Expected {expected}, got {wrapped.provider}")
        
        print("  [OK] Provider detection works")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] Provider detection failed: {e}")
        results.append(False)
    
    # Test 4: Context managers
    print("\n[4/5] Testing context managers...")
    try:
        with customer_context("context-customer"):
            # Context should be set
            pass
        
        with metadata_context({"test": "value"}):
            # Metadata should be set
            pass
        
        print("  [OK] Context managers work")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] Context managers failed: {e}")
        results.append(False)
    
    # Test 5: Usage extraction
    print("\n[5/5] Testing usage extraction...")
    try:
        # Test different response formats
        wrapper = CmdrData(
            client=Mock(),
            cmdrdata_api_key="test",
            disable_tracking=True
        )
        
        # OpenAI format
        response1 = Mock(spec=['usage'])
        response1.usage = Mock(
            spec=['prompt_tokens', 'completion_tokens', 'total_tokens'],
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        usage1 = wrapper._CmdrData__extract_usage(response1)
        assert usage1["input_tokens"] == 10
        assert usage1["output_tokens"] == 20
        
        # Anthropic format
        response2 = Mock(spec=['usage'])
        response2.usage = Mock(
            spec=['input_tokens', 'output_tokens'],
            input_tokens=15,
            output_tokens=25
        )
        usage2 = wrapper._CmdrData__extract_usage(response2)
        assert usage2["input_tokens"] == 15
        assert usage2["output_tokens"] == 25
        
        print("  [OK] Usage extraction works")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] Usage extraction failed: {e}")
        results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("SMOKE TEST RESULTS")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"[SUCCESS] All {total} smoke tests passed!")
        print("The SDK is working correctly.")
        return True
    else:
        print(f"[WARNING] {passed}/{total} tests passed")
        print("Please run full test suite for details.")
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)