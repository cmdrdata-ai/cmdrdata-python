#!/usr/bin/env python
"""
Integration tests for CmdrData SDK with all supported providers.

This test suite verifies that the SDK correctly:
1. Wraps each provider's client
2. Intercepts API calls
3. Extracts token usage
4. Sends tracking events

Run with: pytest tests/test_all_providers_integration.py -v
Or: python tests/test_all_providers_integration.py
"""

import json
import os
import sys
import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmdrdata import CmdrData


class TestProviderIntegration(unittest.TestCase):
    """Test integration with all supported AI providers"""

    def setUp(self):
        """Set up test fixtures"""
        # Clear any leftover context from previous tests
        from cmdrdata.context import clear_customer_context, clear_metadata_context

        clear_customer_context()
        clear_metadata_context()

        self.test_customer_id = "test-customer"
        self.test_metadata = {"test": True, "suite": "integration"}
        # Mock the tracker to prevent actual API calls
        self.tracker_patch = patch("cmdrdata.client.UsageTracker")
        self.mock_tracker_class = self.tracker_patch.start()
        self.mock_tracker = Mock()
        self.mock_tracker.track_usage_background = Mock()
        self.mock_tracker_class.return_value = self.mock_tracker

    def tearDown(self):
        """Clean up"""
        self.tracker_patch.stop()

    def test_openai_integration(self):
        """Test OpenAI client integration"""
        # Create response with OpenAI structure
        mock_response = Mock()
        mock_response.usage = Mock(
            spec=["prompt_tokens", "completion_tokens", "total_tokens"]
        )
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.model = "gpt-3.5-turbo"
        mock_response.choices = [Mock(message=Mock(content="Test response"))]

        # Create proper nested structure
        class MockCompletions:
            def create(self, **kwargs):
                return mock_response

        class MockChat:
            def __init__(self):
                self.completions = MockCompletions()

        class MockOpenAI:
            def __init__(self):
                self.chat = MockChat()

        MockOpenAI.__module__ = "openai"
        mock_openai = MockOpenAI()

        # Wrap with CmdrData
        client = CmdrData(
            client=mock_openai, cmdrdata_api_key="test-key", provider="openai"
        )

        # Make a tracked call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            customer_id=self.test_customer_id,
            metadata=self.test_metadata,
        )

        # Verify response is returned
        self.assertEqual(response, mock_response)

        # Verify tracking was called
        self.mock_tracker.track_usage_background.assert_called_once()

        # Check tracked data
        tracked_data = self.mock_tracker.track_usage_background.call_args[1]
        self.assertEqual(tracked_data["customer_id"], self.test_customer_id)
        self.assertEqual(tracked_data["input_tokens"], 10)
        self.assertEqual(tracked_data["output_tokens"], 20)
        self.assertEqual(tracked_data["total_tokens"], 30)
        self.assertEqual(tracked_data["model"], "gpt-3.5-turbo")
        self.assertEqual(tracked_data["provider"], "openai")
        self.assertEqual(tracked_data["metadata"], self.test_metadata)

    def test_anthropic_integration(self):
        """Test Anthropic client integration"""
        # Create response with Anthropic structure
        mock_response = Mock()
        mock_response.usage = Mock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 25
        mock_response.model = "claude-3-haiku"
        mock_response.content = [Mock(text="Test response")]

        # Create proper nested structure
        class MockMessages:
            def create(self, **kwargs):
                return mock_response

        class MockAnthropic:
            def __init__(self):
                self.messages = MockMessages()

        MockAnthropic.__module__ = "anthropic"
        mock_anthropic = MockAnthropic()

        # Wrap with CmdrData
        client = CmdrData(
            client=mock_anthropic, cmdrdata_api_key="test-key", provider="anthropic"
        )

        # Make a tracked call
        response = client.messages.create(
            model="claude-3-haiku",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            customer_id=self.test_customer_id,
            metadata=self.test_metadata,
        )

        # Verify response is returned
        self.assertEqual(response, mock_response)

        # Verify tracking was called
        self.mock_tracker.track_usage_background.assert_called_once()

        # Check tracked data
        tracked_data = self.mock_tracker.track_usage_background.call_args[1]
        self.assertEqual(tracked_data["customer_id"], self.test_customer_id)
        self.assertEqual(tracked_data["input_tokens"], 15)
        self.assertEqual(tracked_data["output_tokens"], 25)
        self.assertEqual(tracked_data["total_tokens"], 40)
        self.assertEqual(tracked_data["model"], "claude-3-haiku")
        self.assertEqual(tracked_data["provider"], "anthropic")

    def test_cohere_v2_integration(self):
        """Test Cohere V2 client integration"""
        # Mock Cohere client
        mock_cohere = Mock()
        mock_cohere.__module__ = "cohere"

        # Mock chat()
        mock_chat = Mock()

        # Create response with Cohere V2 structure (dict)
        mock_response = {
            "id": "test-123",
            "finish_reason": "COMPLETE",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Test response"}],
            },
            "usage": {
                "billed_units": {"input_tokens": 5, "output_tokens": 10},
                "tokens": {"input_tokens": 71, "output_tokens": 10},
            },
        }

        mock_chat.return_value = mock_response
        mock_cohere.chat = mock_chat

        # Wrap with CmdrData
        client = CmdrData(
            client=mock_cohere, cmdrdata_api_key="test-key", provider="cohere"
        )

        # Make a tracked call
        response = client.chat(
            model="command-r",
            messages=[{"role": "user", "content": "Hello"}],
            customer_id=self.test_customer_id,
            metadata=self.test_metadata,
        )

        # Verify response is returned
        self.assertEqual(response, mock_response)

        # Verify tracking was called
        self.mock_tracker.track_usage_background.assert_called_once()

        # Check tracked data
        tracked_data = self.mock_tracker.track_usage_background.call_args[1]
        self.assertEqual(tracked_data["customer_id"], self.test_customer_id)
        self.assertEqual(tracked_data["input_tokens"], 5)  # Uses billed_units
        self.assertEqual(tracked_data["output_tokens"], 10)
        self.assertEqual(tracked_data["total_tokens"], 15)
        self.assertEqual(tracked_data["provider"], "cohere")

    def test_google_gemini_integration(self):
        """Test Google Gemini integration"""
        # Create response with Gemini structure (no 'usage' attribute)
        mock_response = Mock(spec=["usage_metadata", "text"])
        mock_response.usage_metadata = Mock(
            spec=["prompt_token_count", "candidates_token_count", "total_token_count"]
        )
        mock_response.usage_metadata.prompt_token_count = 12
        mock_response.usage_metadata.candidates_token_count = 18
        mock_response.usage_metadata.total_token_count = 30
        mock_response.text = "Test response"

        # Create proper mock Gemini model
        class MockGeminiModel:
            def generate_content(self, *args, **kwargs):
                return mock_response

        MockGeminiModel.__module__ = "google.generativeai"
        mock_model = MockGeminiModel()

        # Wrap with CmdrData
        client = CmdrData(
            client=mock_model, cmdrdata_api_key="test-key", provider="google"
        )

        # Make a tracked call
        response = client.generate_content(
            "Hello", customer_id=self.test_customer_id, metadata=self.test_metadata
        )

        # Verify response is returned
        self.assertEqual(response, mock_response)

        # Verify tracking was called
        self.mock_tracker.track_usage_background.assert_called_once()

        # Check tracked data
        tracked_data = self.mock_tracker.track_usage_background.call_args[1]
        self.assertEqual(tracked_data["customer_id"], self.test_customer_id)
        self.assertEqual(tracked_data["input_tokens"], 12)
        self.assertEqual(tracked_data["output_tokens"], 18)
        self.assertEqual(tracked_data["total_tokens"], 30)
        self.assertEqual(tracked_data["provider"], "google")

    def test_provider_auto_detection(self):
        """Test automatic provider detection"""
        test_cases = [
            ("openai", "openai"),
            ("anthropic", "anthropic"),
            ("google.generativeai", "google"),
            ("cohere", "cohere"),
            ("unknown.module", "unknown"),
        ]

        for module_name, expected_provider in test_cases:
            # Create a dynamic class with the correct module
            ClientClass = type("MockClient", (), {})
            ClientClass.__module__ = module_name
            mock_client = ClientClass()

            wrapper = CmdrData(
                client=mock_client,
                cmdrdata_api_key="test-key",
                auto_detect_provider=True,
            )

            self.assertEqual(
                wrapper.provider,
                expected_provider,
                f"Failed to detect provider from module {module_name}",
            )

    def test_error_tracking(self):
        """Test that errors are tracked properly"""

        # Create mock client that raises an error
        class MockFailingCompletions:
            def create(self, **kwargs):
                raise Exception("API Error")

        class MockFailingChat:
            def __init__(self):
                self.completions = MockFailingCompletions()

        class MockFailingClient:
            def __init__(self):
                self.chat = MockFailingChat()

        MockFailingClient.__module__ = "openai"
        mock_client = MockFailingClient()

        # Wrap with CmdrData
        client = CmdrData(client=mock_client, cmdrdata_api_key="test-key")

        # Make a call that will error
        with self.assertRaises(Exception) as context:
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                customer_id=self.test_customer_id,
            )

        self.assertEqual(str(context.exception), "API Error")

        # Verify error tracking was called
        self.mock_tracker.track_usage_background.assert_called_once()

        # Check error was tracked
        tracked_data = self.mock_tracker.track_usage_background.call_args[1]
        self.assertTrue(tracked_data["error_occurred"])
        self.assertEqual(tracked_data["error_type"], "Exception")
        self.assertEqual(tracked_data["error_message"], "API Error")

    def test_context_usage(self):
        """Test thread-local context for customer_id and metadata"""
        from cmdrdata.context import set_customer_context, set_metadata_context

        # Set context
        set_customer_context("context-customer")
        set_metadata_context({"from": "context"})

        # Mock client
        mock_client = Mock()
        mock_client.__module__ = "openai"
        mock_method = Mock(return_value={"usage": {"prompt_tokens": 5}})
        mock_client.complete = mock_method

        # Wrap with CmdrData
        client = CmdrData(client=mock_client, cmdrdata_api_key="test-key")

        # Call without explicit customer_id
        client.complete("test")

        # Verify context was used
        tracked_data = self.mock_tracker.track_usage_background.call_args[1]
        self.assertEqual(tracked_data["customer_id"], "context-customer")
        self.assertEqual(tracked_data["metadata"]["from"], "context")

    def test_metadata_merging(self):
        """Test metadata merging from multiple sources"""
        from cmdrdata.context import clear_metadata_context, set_metadata_context

        # Set context metadata
        set_metadata_context({"source": "context", "level": 1})

        # Mock client
        mock_client = Mock()
        mock_client.__module__ = "openai"
        mock_method = Mock(return_value={"usage": {"prompt_tokens": 5}})
        mock_client.complete = mock_method

        # Wrap with default metadata
        client = CmdrData(
            client=mock_client,
            cmdrdata_api_key="test-key",
            metadata={"source": "default", "type": "test"},
        )

        # Call with call-specific metadata
        client.complete(
            "test", customer_id="test", metadata={"source": "call", "extra": "value"}
        )

        # Verify metadata was merged correctly (call > context > default)
        tracked_data = self.mock_tracker.track_usage_background.call_args[1]
        self.assertEqual(tracked_data["metadata"]["source"], "call")  # Call overrides
        self.assertEqual(tracked_data["metadata"]["level"], 1)  # From context
        self.assertEqual(tracked_data["metadata"]["type"], "test")  # From default
        self.assertEqual(tracked_data["metadata"]["extra"], "value")  # From call

        # Clean up context
        clear_metadata_context()


def run_integration_tests():
    """Run all integration tests and print results"""
    print("=" * 60)
    print("CmdrData SDK Integration Tests")
    print("=" * 60)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestProviderIntegration)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")

    if not result.wasSuccessful():
        print("\nFailed tests:")
        for test, traceback in result.failures + result.errors:
            print(f"  - {test}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
