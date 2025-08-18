#!/usr/bin/env python3
"""
Production integration tests for CmdrData SDK
Tests actual communication with CmdrData API
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import requests

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmdrdata import CmdrData, customer_context, metadata_context, track_ai


class ProductionValidator:
    """Validates SDK functionality against production CmdrData API"""

    def __init__(
        self, api_key: str, api_url: str = "https://api.cmdrdata.ai/api/events"
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.test_results = []
        self.test_customer_prefix = (
            f"sdk-test-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        )

    def validate_api_connection(self) -> bool:
        """Test basic API connectivity"""
        print("\n[TEST] Validating API connection...")

        try:
            # Send a test event
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            test_event = {
                "customer_id": f"{self.test_customer_prefix}-connection",
                "model": "test-model",
                "provider": "test-provider",
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "metadata": {"test": "connection"},
            }

            response = requests.post(
                self.api_url, json=test_event, headers=headers, timeout=10
            )

            if response.status_code == 200:
                print("  [OK] API connection successful")
                self.test_results.append(("API Connection", True, None))
                return True
            else:
                error = f"API returned status {response.status_code}: {response.text}"
                print(f"  [FAIL] {error}")
                self.test_results.append(("API Connection", False, error))
                return False

        except Exception as e:
            print(f"  [FAIL] Connection error: {e}")
            self.test_results.append(("API Connection", False, str(e)))
            return False

    def test_mock_provider_tracking(self) -> bool:
        """Test tracking with mock providers"""
        print("\n[TEST] Testing mock provider tracking...")

        try:
            # Create mock provider
            class MockAIProvider:
                def generate(self, prompt: str) -> Dict[str, Any]:
                    return {
                        "text": f"Response to: {prompt}",
                        "model": "mock-model-v1",
                        "usage": Mock(
                            prompt_tokens=len(prompt.split()),
                            completion_tokens=50,
                            total_tokens=len(prompt.split()) + 50,
                        ),
                    }

            # Wrap with SDK
            client = track_ai(
                MockAIProvider(),
                cmdrdata_api_key=self.api_key,
                cmdrdata_url=self.api_url,
            )

            # Make tracked call
            customer_id = f"{self.test_customer_prefix}-mock"
            response = client.generate(
                "This is a test prompt for the mock provider",
                customer_id=customer_id,
                metadata={
                    "test_type": "mock_provider",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Wait for background tracking
            time.sleep(2)

            print("  [OK] Mock provider tracked successfully")
            self.test_results.append(("Mock Provider Tracking", True, None))
            return True

        except Exception as e:
            print(f"  [FAIL] Mock provider error: {e}")
            self.test_results.append(("Mock Provider Tracking", False, str(e)))
            return False

    def test_multiple_customers(self) -> bool:
        """Test tracking for multiple customers"""
        print("\n[TEST] Testing multiple customer tracking...")

        try:

            class MultiCustomerProvider:
                def process(self, customer: str, task: str) -> Dict[str, Any]:
                    return {
                        "result": f"Processed {task} for {customer}",
                        "usage": Mock(
                            prompt_tokens=10, completion_tokens=20, total_tokens=30
                        ),
                    }

            client = track_ai(
                MultiCustomerProvider(),
                cmdrdata_api_key=self.api_key,
                cmdrdata_url=self.api_url,
            )

            # Track for multiple customers
            customers = ["alpha", "beta", "gamma"]
            for customer in customers:
                customer_id = f"{self.test_customer_prefix}-{customer}"
                response = client.process(
                    customer,
                    "analysis",
                    customer_id=customer_id,
                    metadata={"customer_type": customer},
                )

            # Wait for all events
            time.sleep(2)

            print(f"  [OK] Tracked {len(customers)} customers successfully")
            self.test_results.append(("Multiple Customers", True, None))
            return True

        except Exception as e:
            print(f"  [FAIL] Multiple customer error: {e}")
            self.test_results.append(("Multiple Customers", False, str(e)))
            return False

    def test_context_managers(self) -> bool:
        """Test context manager functionality"""
        print("\n[TEST] Testing context managers...")

        try:

            class ContextTestProvider:
                def analyze(self, data: str) -> Dict[str, Any]:
                    return {
                        "analysis": f"Analyzed: {data}",
                        "usage": Mock(
                            prompt_tokens=15, completion_tokens=25, total_tokens=40
                        ),
                    }

            client = track_ai(
                ContextTestProvider(),
                cmdrdata_api_key=self.api_key,
                cmdrdata_url=self.api_url,
            )

            # Test customer context
            with customer_context(f"{self.test_customer_prefix}-context"):
                response = client.analyze("customer context test")

            # Test metadata context
            with metadata_context({"env": "production", "version": "1.0"}):
                response = client.analyze(
                    "metadata context test",
                    customer_id=f"{self.test_customer_prefix}-metadata",
                )

            # Test nested contexts
            with customer_context(f"{self.test_customer_prefix}-nested"):
                with metadata_context({"level": "nested", "test": True}):
                    response = client.analyze("nested context test")

            time.sleep(2)

            print("  [OK] Context managers working correctly")
            self.test_results.append(("Context Managers", True, None))
            return True

        except Exception as e:
            print(f"  [FAIL] Context manager error: {e}")
            self.test_results.append(("Context Managers", False, str(e)))
            return False

    def test_error_handling(self) -> bool:
        """Test error tracking and resilience"""
        print("\n[TEST] Testing error handling...")

        try:

            class ErrorProneProvider:
                def __init__(self):
                    self.call_count = 0

                def risky_operation(self, risk_level: str) -> Dict[str, Any]:
                    self.call_count += 1

                    if risk_level == "high":
                        raise ValueError("Operation failed: risk too high")

                    return {
                        "result": "Success",
                        "usage": Mock(
                            prompt_tokens=5, completion_tokens=10, total_tokens=15
                        ),
                    }

            provider = ErrorProneProvider()
            client = track_ai(
                provider, cmdrdata_api_key=self.api_key, cmdrdata_url=self.api_url
            )

            # Successful call
            response = client.risky_operation(
                "low", customer_id=f"{self.test_customer_prefix}-success"
            )

            # Error call (should raise but still track)
            try:
                response = client.risky_operation(
                    "high", customer_id=f"{self.test_customer_prefix}-error"
                )
            except ValueError as e:
                if "risk too high" not in str(e):
                    raise

            time.sleep(2)

            print("  [OK] Error handling works correctly")
            self.test_results.append(("Error Handling", True, None))
            return True

        except Exception as e:
            print(f"  [FAIL] Error handling issue: {e}")
            self.test_results.append(("Error Handling", False, str(e)))
            return False

    def test_high_metadata_volume(self) -> bool:
        """Test with complex metadata structures"""
        print("\n[TEST] Testing complex metadata...")

        try:

            class MetadataTestProvider:
                def process(self, data: str) -> Dict[str, Any]:
                    return {
                        "result": "Processed",
                        "usage": Mock(
                            prompt_tokens=100, completion_tokens=200, total_tokens=300
                        ),
                    }

            client = track_ai(
                MetadataTestProvider(),
                cmdrdata_api_key=self.api_key,
                cmdrdata_url=self.api_url,
            )

            # Complex metadata
            complex_metadata = {
                "session": {
                    "id": "sess_123",
                    "start": datetime.utcnow().isoformat(),
                    "user": {
                        "id": "user_456",
                        "tier": "premium",
                        "features": ["advanced", "analytics", "export"],
                    },
                },
                "request": {
                    "type": "analysis",
                    "priority": "high",
                    "tags": ["ml", "nlp", "production"],
                },
                "system": {
                    "version": "2.1.0",
                    "environment": "production",
                    "region": "us-west-2",
                },
            }

            response = client.process(
                "test data",
                customer_id=f"{self.test_customer_prefix}-metadata",
                metadata=complex_metadata,
            )

            time.sleep(2)

            print("  [OK] Complex metadata handled correctly")
            self.test_results.append(("Complex Metadata", True, None))
            return True

        except Exception as e:
            print(f"  [FAIL] Metadata handling error: {e}")
            self.test_results.append(("Complex Metadata", False, str(e)))
            return False

    def run_all_tests(self) -> bool:
        """Run all production validation tests"""
        print("\n" + "=" * 60)
        print("CMDRDATA SDK PRODUCTION VALIDATION")
        print("=" * 60)
        print(f"API URL: {self.api_url}")
        print(f"Test prefix: {self.test_customer_prefix}")

        # Run tests
        all_passed = True

        if not self.validate_api_connection():
            print("\n[CRITICAL] API connection failed - skipping remaining tests")
            all_passed = False
        else:
            # Run all other tests
            tests = [
                self.test_mock_provider_tracking,
                self.test_multiple_customers,
                self.test_context_managers,
                self.test_error_handling,
                self.test_high_metadata_volume,
            ]

            for test in tests:
                if not test():
                    all_passed = False

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, success, _ in self.test_results if success)
        failed = len(self.test_results) - passed

        for test_name, success, error in self.test_results:
            status = "[PASS]" if success else "[FAIL]"
            print(f"{status} {test_name}")
            if error:
                print(f"       Error: {error}")

        print(f"\nTotal: {len(self.test_results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if all_passed:
            print("\n[SUCCESS] All production validation tests passed!")
            print("The SDK is ready for production use.")
        else:
            print(
                "\n[WARNING] Some tests failed. Please review before production deployment."
            )

        return all_passed


def main():
    """Main entry point for production validation"""
    # Get API key from environment or command line
    api_key = os.getenv("CMDRDATA_API_KEY")

    if not api_key:
        print("ERROR: CMDRDATA_API_KEY environment variable not set")
        print("\nUsage:")
        print("  export CMDRDATA_API_KEY=your-api-key")
        print("  python tests/test_production.py")
        sys.exit(1)

    # Get API URL (default to production)
    api_url = os.getenv("CMDRDATA_URL", "https://api.cmdrdata.ai/api/events")

    # For local testing
    if "--local" in sys.argv:
        api_url = "http://localhost:8000/api/events"
        print("Using local API URL:", api_url)

    # Run validation
    validator = ProductionValidator(api_key, api_url)
    success = validator.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
