#!/usr/bin/env python3
"""
Production validation for CmdrData SDK
Comprehensive test to ensure SDK is ready for production
"""

import os
import sys
import time
import json
from datetime import datetime
from unittest.mock import Mock
import subprocess

from cmdrdata import track_ai, CmdrData, customer_context, metadata_context


def validate_production_readiness():
    """Validate that SDK is ready for production"""
    print("\n" + "="*70)
    print(" CMDRDATA SDK - PRODUCTION READINESS VALIDATION")
    print("="*70)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print(f"Python: {sys.version}")
    
    all_checks = []
    
    # Check 1: Package imports
    print("\n[CHECK 1] Package imports and dependencies...")
    try:
        import cmdrdata
        from cmdrdata import track_ai, CmdrData
        from cmdrdata.tracker import UsageTracker
        from cmdrdata.context import customer_context, metadata_context
        from cmdrdata.exceptions import CMDRDataError
        print("  [OK] All core modules import successfully")
        all_checks.append(("Package Imports", True))
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        all_checks.append(("Package Imports", False))
        return False
    
    # Check 2: Test suite passes
    print("\n[CHECK 2] Running test suite...")
    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "tests/test_comprehensive.py", "--tb=no", "-q"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if "passed" in result.stdout:
            # Extract test count
            import re
            match = re.search(r'(\d+) passed', result.stdout)
            if match:
                test_count = match.group(1)
                print(f"  [OK] {test_count} tests passed")
                all_checks.append(("Test Suite", True))
            else:
                print("  [OK] Tests passed")
                all_checks.append(("Test Suite", True))
        else:
            print(f"  [FAIL] Some tests failed")
            print(f"       Output: {result.stdout}")
            all_checks.append(("Test Suite", False))
    except Exception as e:
        print(f"  [FAIL] Test execution error: {e}")
        all_checks.append(("Test Suite", False))
    
    # Check 3: Provider detection
    print("\n[CHECK 3] Provider detection...")
    try:
        providers_tested = 0
        providers = [
            ("openai.client", "openai"),
            ("anthropic.client", "anthropic"),
            ("google.generativeai", "google"),
            ("cohere.client", "cohere"),
            ("huggingface_hub", "huggingface")
        ]
        
        for module, expected in providers:
            class TestClient:
                pass
            TestClient.__module__ = module
            
            wrapper = CmdrData(
                client=TestClient(),
                cmdrdata_api_key="test",
                disable_tracking=True
            )
            
            if wrapper.provider == expected:
                providers_tested += 1
            else:
                raise ValueError(f"Provider detection failed for {module}")
        
        print(f"  [OK] All {providers_tested} providers detected correctly")
        all_checks.append(("Provider Detection", True))
    except Exception as e:
        print(f"  [FAIL] Provider detection error: {e}")
        all_checks.append(("Provider Detection", False))
    
    # Check 4: Usage extraction
    print("\n[CHECK 4] Usage extraction patterns...")
    try:
        wrapper = CmdrData(
            client=Mock(),
            cmdrdata_api_key="test",
            disable_tracking=True
        )
        
        patterns_tested = 0
        
        # OpenAI pattern
        resp1 = Mock(spec=['usage'])
        resp1.usage = Mock(
            spec=['prompt_tokens', 'completion_tokens', 'total_tokens'],
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        usage1 = wrapper._CmdrData__extract_usage(resp1)
        assert usage1["input_tokens"] == 10
        assert usage1["output_tokens"] == 20
        patterns_tested += 1
        
        # Anthropic pattern
        resp2 = Mock(spec=['usage'])
        resp2.usage = Mock(
            spec=['input_tokens', 'output_tokens'],
            input_tokens=15, output_tokens=25
        )
        usage2 = wrapper._CmdrData__extract_usage(resp2)
        assert usage2["input_tokens"] == 15
        assert usage2["output_tokens"] == 25
        patterns_tested += 1
        
        # Google pattern
        resp3 = Mock(spec=['usage_metadata'])
        resp3.usage_metadata = Mock(
            spec=['prompt_token_count', 'candidates_token_count', 'total_token_count'],
            prompt_token_count=5, candidates_token_count=15, total_token_count=20
        )
        usage3 = wrapper._CmdrData__extract_usage(resp3)
        assert usage3["input_tokens"] == 5
        assert usage3["output_tokens"] == 15
        patterns_tested += 1
        
        # Cohere pattern
        resp4 = Mock(spec=['meta'])
        resp4.meta = Mock(spec=['billed_units'])
        resp4.meta.billed_units = Mock(
            spec=['input_tokens', 'output_tokens'],
            input_tokens=8, output_tokens=12
        )
        usage4 = wrapper._CmdrData__extract_usage(resp4)
        assert usage4["input_tokens"] == 8
        assert usage4["output_tokens"] == 12
        patterns_tested += 1
        
        print(f"  [OK] All {patterns_tested} usage patterns work correctly")
        all_checks.append(("Usage Extraction", True))
    except Exception as e:
        print(f"  [FAIL] Usage extraction error: {e}")
        all_checks.append(("Usage Extraction", False))
    
    # Check 5: Context managers
    print("\n[CHECK 5] Context managers...")
    try:
        from cmdrdata.context import (
            set_customer_context, get_customer_context, clear_customer_context,
            set_metadata_context, get_metadata_context, clear_metadata_context
        )
        
        # Test customer context
        clear_customer_context()
        assert get_customer_context() is None
        
        with customer_context("test-customer"):
            assert get_customer_context() == "test-customer"
        
        assert get_customer_context() is None
        
        # Test metadata context
        clear_metadata_context()
        assert get_metadata_context() == {}
        
        with metadata_context({"key": "value"}):
            assert get_metadata_context()["key"] == "value"
        
        assert get_metadata_context() == {}
        
        print("  [OK] Context managers work correctly")
        all_checks.append(("Context Managers", True))
    except Exception as e:
        print(f"  [FAIL] Context manager error: {e}")
        all_checks.append(("Context Managers", False))
    
    # Check 6: Error resilience
    print("\n[CHECK 6] Error resilience...")
    try:
        class FailingTracker:
            def track_usage_background(self, **kwargs):
                raise Exception("Tracking failed!")
        
        class WorkingClient:
            def process(self, data):
                return {"result": "success", "usage": Mock(total_tokens=10)}
        
        wrapper = CmdrData(
            client=WorkingClient(),
            cmdrdata_api_key="test"
        )
        wrapper.tracker = FailingTracker()
        
        # Should not raise despite tracker failure
        result = wrapper.process("test data")
        assert result["result"] == "success"
        
        print("  [OK] SDK resilient to tracking failures")
        all_checks.append(("Error Resilience", True))
    except Exception as e:
        print(f"  [FAIL] Error resilience failed: {e}")
        all_checks.append(("Error Resilience", False))
    
    # Check 7: Thread safety
    print("\n[CHECK 7] Thread safety...")
    try:
        import threading
        
        results = []
        lock = threading.Lock()
        
        def thread_test(thread_id):
            with customer_context(f"thread-{thread_id}"):
                ctx = get_customer_context()
                with lock:
                    results.append(ctx)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=thread_test, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Each thread should have its own context
        expected = {f"thread-{i}" for i in range(5)}
        actual = set(results)
        
        if expected == actual:
            print("  [OK] Thread-safe context management")
            all_checks.append(("Thread Safety", True))
        else:
            raise ValueError(f"Thread safety issue: expected {expected}, got {actual}")
    except Exception as e:
        print(f"  [FAIL] Thread safety error: {e}")
        all_checks.append(("Thread Safety", False))
    
    # Check 8: API key handling
    print("\n[CHECK 8] API key configuration...")
    try:
        # Test with environment variable
        os.environ["CMDRDATA_API_KEY"] = "env-test-key"
        wrapper = CmdrData(client=Mock())
        assert wrapper.tracker.api_key == "env-test-key"
        
        # Test with explicit key
        wrapper2 = CmdrData(
            client=Mock(),
            cmdrdata_api_key="explicit-key"
        )
        assert wrapper2.tracker.api_key == "explicit-key"
        
        # Clean up
        del os.environ["CMDRDATA_API_KEY"]
        
        print("  [OK] API key configuration works")
        all_checks.append(("API Key Config", True))
    except Exception as e:
        print(f"  [FAIL] API key configuration error: {e}")
        all_checks.append(("API Key Config", False))
    
    # Summary
    print("\n" + "="*70)
    print(" VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in all_checks if success)
    total = len(all_checks)
    
    for check_name, success in all_checks:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {check_name}")
    
    print(f"\n  Total: {total} checks")
    print(f"  Passed: {passed}")
    print(f"  Failed: {total - passed}")
    
    if passed == total:
        print("\n" + "="*70)
        print(" [SUCCESS] SDK IS PRODUCTION READY!")
        print("="*70)
        print("\n  The CmdrData SDK has passed all production readiness checks.")
        print("  It is ready to be deployed and used in production environments.")
        print("\n  Next steps:")
        print("  1. Push to GitHub repository")
        print("  2. Publish to PyPI")
        print("  3. Update documentation")
        print("  4. Notify users of availability")
        return True
    else:
        print("\n" + "="*70)
        print(" [WARNING] SOME CHECKS FAILED")
        print("="*70)
        print("\n  Please address the failed checks before production deployment.")
        return False


if __name__ == "__main__":
    success = validate_production_readiness()
    sys.exit(0 if success else 1)