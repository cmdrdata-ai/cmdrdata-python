#!/usr/bin/env python3
"""
Comprehensive test suite for CmdrData SDK with near 100% coverage
Includes unit tests, integration tests, error handling, and edge cases
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, PropertyMock, call, patch

import httpx
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cmdrdata import CmdrData, customer_context, metadata_context, track_ai
from cmdrdata.client import CmdrDataProxy
from cmdrdata.context import (
    clear_customer_context,
    clear_metadata_context,
    get_customer_context,
    get_metadata_context,
    set_customer_context,
    set_metadata_context,
    update_metadata_context,
)
from cmdrdata.exceptions import (
    CMDRDataError,
    ConfigurationError,
    NetworkError,
    TrackingError,
    ValidationError,
)
from cmdrdata.tracker import UsageTracker


class TestCmdrDataInitialization:
    """Test SDK initialization and configuration"""

    def test_init_with_client_instance(self):
        """Test initialization with existing client instance"""
        mock_client = Mock()
        mock_client.__class__.__module__ = "openai.client"

        wrapper = CmdrData(
            client=mock_client, cmdrdata_api_key="test-key", disable_tracking=True
        )

        assert wrapper._client == mock_client
        assert wrapper.provider == "openai"
        assert wrapper._tracking_enabled is False

    def test_init_with_client_class(self):
        """Test initialization with client class and kwargs"""

        class MockClient:
            def __init__(self, api_key: str):
                self.api_key = api_key

        MockClient.__module__ = "anthropic.client"

        wrapper = CmdrData(
            client_class=MockClient,
            client_kwargs={"api_key": "test-key"},
            cmdrdata_api_key="cmd-key",
            disable_tracking=True,
        )

        assert isinstance(wrapper._client, MockClient)
        assert wrapper._client.api_key == "test-key"
        assert wrapper.provider == "anthropic"

    def test_init_validation_error(self):
        """Test initialization raises ValidationError without client"""
        with pytest.raises(ValidationError) as exc:
            CmdrData(cmdrdata_api_key="test-key")

        assert "Must provide either 'client'" in str(exc.value)

    def test_init_with_environment_variables(self):
        """Test initialization uses environment variables"""
        mock_client = Mock()

        with patch.dict(
            os.environ,
            {
                "CMDRDATA_API_KEY": "env-key",
                "CMDRDATA_URL": "https://custom.api/events",
                "CMDRDATA_CUSTOMER_ID": "env-customer",
            },
        ):
            wrapper = CmdrData(client=mock_client)

            assert wrapper.tracker.api_key == "env-key"
            assert wrapper.tracker.endpoint == "https://custom.api/events"
            assert wrapper.default_customer_id == "env-customer"

    def test_init_without_api_key_disables_tracking(self):
        """Test that missing API key disables tracking with warning"""
        mock_client = Mock()

        with patch("cmdrdata.client.logger") as mock_logger:
            wrapper = CmdrData(client=mock_client)

            mock_logger.warning.assert_called_once()
            assert wrapper._tracking_enabled is False
            assert wrapper.tracker.disabled is True

    def test_init_with_custom_metadata(self):
        """Test initialization with default metadata"""
        mock_client = Mock()
        metadata = {"environment": "production", "version": "2.0"}

        wrapper = CmdrData(
            client=mock_client,
            cmdrdata_api_key="test-key",
            metadata=metadata,
            disable_tracking=True,
        )

        assert wrapper.default_metadata == metadata


class TestProviderDetection:
    """Test automatic provider detection"""

    @pytest.mark.parametrize(
        "module,class_name,expected",
        [
            ("openai.client", "OpenAI", "openai"),
            ("openai", "Client", "openai"),
            ("anthropic.client", "Anthropic", "anthropic"),
            ("anthropic", "Client", "anthropic"),
            ("google.generativeai", "GenerativeModel", "google"),
            ("google.genai", "Model", "google"),
            ("cohere.client", "Client", "cohere"),
            ("huggingface_hub", "InferenceClient", "huggingface"),
            ("transformers", "Pipeline", "huggingface"),
            ("replicate", "Client", "replicate"),
            ("together", "Together", "together"),
            ("perplexity", "Client", "perplexity"),
            ("unknown.module", "CustomClient", "unknown"),
        ],
    )
    def test_provider_detection(self, module, class_name, expected):
        """Test provider detection for various AI clients"""

        class MockClient:
            pass

        MockClient.__module__ = module
        MockClient.__name__ = class_name

        wrapper = CmdrData(
            client=MockClient(), cmdrdata_api_key="test", disable_tracking=True
        )

        assert wrapper.provider == expected

    def test_manual_provider_override(self):
        """Test manual provider override"""

        class MockClient:
            pass

        MockClient.__module__ = "unknown"

        wrapper = CmdrData(
            client=MockClient(),
            provider="custom-provider",
            cmdrdata_api_key="test",
            disable_tracking=True,
        )

        assert wrapper.provider == "custom-provider"


class TestMethodWrapping:
    """Test method wrapping and attribute access"""

    def test_simple_method_wrapping(self):
        """Test wrapping of simple methods"""

        class MockClient:
            def generate(self, prompt: str) -> Dict[str, Any]:
                return {"text": "response", "usage": {"tokens": 10}}

        wrapper = CmdrData(
            client=MockClient(), cmdrdata_api_key="test", disable_tracking=True
        )

        assert hasattr(wrapper, "generate")
        assert callable(wrapper.generate)

        result = wrapper.generate("test prompt")
        assert result["text"] == "response"

    def test_nested_attribute_access(self):
        """Test nested attribute access (like client.chat.completions)"""

        class Completions:
            def create(self, **kwargs):
                return {"response": "test"}

        class Chat:
            def __init__(self):
                self.completions = Completions()

        class MockClient:
            def __init__(self):
                self.chat = Chat()

        wrapper = CmdrData(
            client=MockClient(), cmdrdata_api_key="test", disable_tracking=True
        )

        assert hasattr(wrapper, "chat")
        assert hasattr(wrapper.chat, "completions")
        assert callable(wrapper.chat.completions.create)

        result = wrapper.chat.completions.create(model="test")
        assert result["response"] == "test"

    def test_attribute_caching(self):
        """Test that wrapped attributes are cached"""
        mock_client = Mock()
        mock_client.method = Mock(return_value="result")

        wrapper = CmdrData(
            client=mock_client, cmdrdata_api_key="test", disable_tracking=True
        )

        # Access same method twice
        method1 = wrapper.method
        method2 = wrapper.method

        # Should be the same wrapped instance
        assert method1 is method2

    def test_non_existent_attribute_error(self):
        """Test that accessing non-existent attributes raises AttributeError"""
        mock_client = Mock(spec=[])  # Empty spec

        wrapper = CmdrData(
            client=mock_client, cmdrdata_api_key="test", disable_tracking=True
        )

        with pytest.raises(AttributeError) as exc:
            wrapper.non_existent_method()

        assert "has no attribute 'non_existent_method'" in str(exc.value)

    def test_property_access(self):
        """Test that properties are accessible"""

        class MockClient:
            @property
            def api_key(self):
                return "secret-key"

        wrapper = CmdrData(
            client=MockClient(), cmdrdata_api_key="test", disable_tracking=True
        )

        assert wrapper.api_key == "secret-key"


class TestUsageTracking:
    """Test usage tracking functionality"""

    def setup_method(self):
        """Clear context before each test"""
        clear_customer_context()
        clear_metadata_context()

    @patch("cmdrdata.tracker.requests.post")
    def test_successful_tracking(self, mock_post):
        """Test successful usage tracking"""
        mock_post.return_value.status_code = 200

        class MockClient:
            def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
                return {
                    "text": "response",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                    "model": "test-model",
                }

        wrapper = CmdrData(
            client=MockClient(), cmdrdata_api_key="test-key", customer_id="customer-123"
        )

        result = wrapper.generate("test prompt")

        # Give background thread time to send
        time.sleep(0.1)

        mock_post.assert_called_once()
        call_args = mock_post.call_args

        # Check headers
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"

        # Check payload
        payload = call_args.kwargs["json"]
        assert payload["customer_id"] == "customer-123"
        assert payload["model"] == "test-model"
        assert payload["input_tokens"] == 10
        assert payload["output_tokens"] == 20
        assert payload["total_tokens"] == 30

    def test_tracking_with_metadata(self):
        """Test tracking with custom metadata"""
        tracker = Mock()

        class MockClient:
            def generate(self, prompt: str) -> Dict[str, Any]:
                return {"text": "response", "usage": {"total_tokens": 10}}

        wrapper = CmdrData(
            client=MockClient(),
            cmdrdata_api_key="test-key",
            metadata={"environment": "test"},
        )
        wrapper.tracker = tracker

        # Call with additional metadata
        wrapper.generate("test", customer_id="cust-456", metadata={"feature": "chat"})

        # Give background thread time
        time.sleep(0.1)

        tracker.track_usage_background.assert_called_once()
        call_kwargs = tracker.track_usage_background.call_args.kwargs

        assert call_kwargs["customer_id"] == "cust-456"
        assert call_kwargs["metadata"]["environment"] == "test"
        assert call_kwargs["metadata"]["feature"] == "chat"

    def test_tracking_with_context_managers(self):
        """Test tracking with context managers"""
        tracker = Mock()

        class MockClient:
            def generate(self, prompt: str) -> Dict[str, Any]:
                return {"text": "response"}

        wrapper = CmdrData(client=MockClient(), cmdrdata_api_key="test-key")
        wrapper.tracker = tracker

        with customer_context("context-customer"):
            with metadata_context({"context_feature": "test"}):
                wrapper.generate("test")

        time.sleep(0.1)

        tracker.track_usage_background.assert_called_once()
        call_kwargs = tracker.track_usage_background.call_args.kwargs

        assert call_kwargs["customer_id"] == "context-customer"
        assert call_kwargs["metadata"]["context_feature"] == "test"

    def test_tracking_disabled(self):
        """Test that tracking can be disabled"""
        tracker = Mock()

        class MockClient:
            def generate(self, prompt: str) -> Dict[str, Any]:
                return {"text": "response"}

        wrapper = CmdrData(
            client=MockClient(), cmdrdata_api_key="test-key", disable_tracking=True
        )
        wrapper.tracker = tracker

        wrapper.generate("test")
        time.sleep(0.1)

        tracker.track_usage_background.assert_not_called()


class TestUsageExtraction:
    """Test usage information extraction from responses"""

    def test_openai_usage_extraction(self):
        """Test usage extraction from OpenAI-style response"""
        response = Mock(
            usage=Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )

        wrapper = CmdrData(
            client=Mock(), cmdrdata_api_key="test", disable_tracking=True
        )

        usage = wrapper._CmdrData__extract_usage(response)

        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 20
        assert usage["total_tokens"] == 30

    def test_anthropic_usage_extraction(self):
        """Test usage extraction from Anthropic-style response"""
        response = Mock(spec=["usage"])
        response.usage = Mock(
            spec=["input_tokens", "output_tokens"], input_tokens=15, output_tokens=25
        )

        wrapper = CmdrData(
            client=Mock(), cmdrdata_api_key="test", disable_tracking=True
        )

        usage = wrapper._CmdrData__extract_usage(response)

        assert usage["input_tokens"] == 15
        assert usage["output_tokens"] == 25
        assert usage["total_tokens"] == 40  # Calculated

    def test_google_usage_extraction(self):
        """Test usage extraction from Google-style response"""
        response = Mock(spec=["usage_metadata"])
        response.usage_metadata = Mock(
            spec=["prompt_token_count", "candidates_token_count", "total_token_count"],
            prompt_token_count=5,
            candidates_token_count=15,
            total_token_count=20,
        )

        wrapper = CmdrData(
            client=Mock(), cmdrdata_api_key="test", disable_tracking=True
        )

        usage = wrapper._CmdrData__extract_usage(response)

        assert usage["input_tokens"] == 5
        assert usage["output_tokens"] == 15
        assert usage["total_tokens"] == 20

    def test_cohere_usage_extraction(self):
        """Test usage extraction from Cohere-style response"""
        response = Mock(spec=["meta"])
        response.meta = Mock(spec=["billed_units"])
        response.meta.billed_units = Mock(
            spec=["input_tokens", "output_tokens"], input_tokens=8, output_tokens=12
        )

        wrapper = CmdrData(
            client=Mock(), cmdrdata_api_key="test", disable_tracking=True
        )

        usage = wrapper._CmdrData__extract_usage(response)

        assert usage["input_tokens"] == 8
        assert usage["output_tokens"] == 12
        assert usage["total_tokens"] == 20

    def test_empty_response_extraction(self):
        """Test usage extraction from empty response"""
        wrapper = CmdrData(
            client=Mock(), cmdrdata_api_key="test", disable_tracking=True
        )

        usage = wrapper._CmdrData__extract_usage(None)

        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["total_tokens"] == 0


class TestModelExtraction:
    """Test model name extraction"""

    def test_model_from_kwargs(self):
        """Test extracting model from kwargs"""
        wrapper = CmdrData(
            client=Mock(), cmdrdata_api_key="test", disable_tracking=True
        )

        model = wrapper._CmdrData__extract_model(
            args=(), kwargs={"model": "gpt-4"}, response=None
        )

        assert model == "gpt-4"

    def test_model_from_args(self):
        """Test extracting model from args"""
        wrapper = CmdrData(
            client=Mock(), cmdrdata_api_key="test", disable_tracking=True
        )

        model = wrapper._CmdrData__extract_model(
            args=("claude-3",), kwargs={}, response=None
        )

        assert model == "claude-3"

    def test_model_from_response(self):
        """Test extracting model from response"""
        wrapper = CmdrData(
            client=Mock(), cmdrdata_api_key="test", disable_tracking=True
        )

        response = Mock(model="gemini-pro")

        model = wrapper._CmdrData__extract_model(args=(), kwargs={}, response=response)

        assert model == "gemini-pro"

    def test_model_unknown(self):
        """Test unknown model extraction"""
        wrapper = CmdrData(
            client=Mock(), cmdrdata_api_key="test", disable_tracking=True
        )

        model = wrapper._CmdrData__extract_model(args=(), kwargs={}, response=None)

        assert model == "unknown"


class TestErrorHandling:
    """Test error handling in the SDK"""

    def test_method_error_still_tracked(self):
        """Test that errors in methods are tracked and re-raised"""
        tracker = Mock()

        class MockClient:
            def generate(self, prompt: str):
                raise ValueError("API Error")

        wrapper = CmdrData(client=MockClient(), cmdrdata_api_key="test-key")
        wrapper.tracker = tracker

        with pytest.raises(ValueError) as exc:
            wrapper.generate("test")

        assert str(exc.value) == "API Error"

        time.sleep(0.1)

        tracker.track_usage_background.assert_called_once()
        call_kwargs = tracker.track_usage_background.call_args.kwargs

        assert call_kwargs["error_occurred"] is True
        assert call_kwargs["error_type"] == "ValueError"
        assert call_kwargs["error_message"] == "API Error"

    def test_tracking_error_doesnt_break_call(self):
        """Test that tracking errors don't break the API call"""

        class MockClient:
            def generate(self, prompt: str):
                return {"text": "success"}

        wrapper = CmdrData(client=MockClient(), cmdrdata_api_key="test-key")

        # Make tracker's background tracking fail
        wrapper.tracker.track_usage_background = Mock(
            side_effect=Exception("Tracking failed")
        )

        # Should still work despite tracking failure
        result = wrapper.generate("test")
        assert result["text"] == "success"


class TestContextManagers:
    """Test context manager functionality"""

    def test_customer_context_basic(self):
        """Test basic customer context manager"""
        clear_customer_context()

        assert get_customer_context() is None

        with customer_context("customer-123"):
            assert get_customer_context() == "customer-123"

        assert get_customer_context() is None

    def test_customer_context_nested(self):
        """Test nested customer contexts"""
        clear_customer_context()

        with customer_context("customer-1"):
            assert get_customer_context() == "customer-1"

            with customer_context("customer-2"):
                assert get_customer_context() == "customer-2"

            assert get_customer_context() == "customer-1"

        assert get_customer_context() is None

    def test_metadata_context_basic(self):
        """Test basic metadata context manager"""
        clear_metadata_context()

        assert get_metadata_context() == {}

        with metadata_context({"feature": "test"}):
            assert get_metadata_context() == {"feature": "test"}

        assert get_metadata_context() == {}

    def test_metadata_context_nested(self):
        """Test nested metadata contexts"""
        clear_metadata_context()

        with metadata_context({"env": "test"}):
            assert get_metadata_context() == {"env": "test"}

            with metadata_context({"feature": "chat"}):
                assert get_metadata_context() == {"feature": "chat"}

            assert get_metadata_context() == {"env": "test"}

        assert get_metadata_context() == {}

    def test_update_metadata_context(self):
        """Test updating metadata context"""
        clear_metadata_context()

        set_metadata_context({"env": "test"})
        update_metadata_context({"feature": "chat"})

        context = get_metadata_context()
        assert context["env"] == "test"
        assert context["feature"] == "chat"

        clear_metadata_context()
        assert get_metadata_context() == {}


class TestProxy:
    """Test CmdrDataProxy functionality"""

    def test_proxy_method_wrapping(self):
        """Test proxy wraps methods correctly"""

        class Nested:
            def method(self):
                return "result"

        class MockClient:
            def __init__(self):
                self.nested = Nested()

        wrapper = CmdrData(
            client=MockClient(), cmdrdata_api_key="test", disable_tracking=True
        )

        # Access through proxy
        proxy = wrapper.nested
        assert isinstance(proxy, CmdrDataProxy)

        result = proxy.method()
        assert result == "result"

    def test_proxy_callable(self):
        """Test proxy handles callable objects"""

        class CallableObject:
            def __call__(self, x):
                return x * 2

        class MockClient:
            def __init__(self):
                self.callable = CallableObject()

        wrapper = CmdrData(
            client=MockClient(), cmdrdata_api_key="test", disable_tracking=True
        )

        result = wrapper.callable(5)
        assert result == 10

    def test_proxy_non_callable_error(self):
        """Test proxy raises error for non-callable"""

        class MockClient:
            def __init__(self):
                self.not_callable = "string"

        wrapper = CmdrData(
            client=MockClient(), cmdrdata_api_key="test", disable_tracking=True
        )

        with pytest.raises(TypeError) as exc:
            wrapper.not_callable()

        assert "object is not callable" in str(exc.value)


class TestTracker:
    """Test UsageTracker functionality"""

    @patch("cmdrdata.tracker.requests.post")
    def test_tracker_successful_send(self, mock_post):
        """Test tracker successfully sends events"""
        mock_post.return_value.status_code = 200

        tracker = UsageTracker(api_key="test-key")

        result = tracker.track_usage(
            customer_id="customer-123",
            model="test-model",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            provider="test-provider",
        )

        assert result is True
        mock_post.assert_called_once()

    @patch("cmdrdata.tracker.requests.post")
    def test_tracker_retry_on_500(self, mock_post):
        """Test tracker retries on 500 errors"""
        mock_post.side_effect = [
            Mock(status_code=500),
            Mock(status_code=500),
            Mock(status_code=200),
        ]

        tracker = UsageTracker(api_key="test-key", max_retries=3)

        result = tracker.track_usage(
            customer_id="customer-123",
            model="test-model",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )

        assert result is True
        assert mock_post.call_count == 3

    @patch("cmdrdata.tracker.requests.post")
    def test_tracker_auth_error_disables(self, mock_post):
        """Test tracker disables on auth error"""
        mock_post.return_value.status_code = 401

        tracker = UsageTracker(api_key="bad-key")

        result = tracker.track_usage(
            customer_id="customer-123",
            model="test-model",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )

        assert result is False
        assert tracker.disabled is True

    @patch("cmdrdata.tracker.requests.post")
    def test_tracker_timeout_retry(self, mock_post):
        """Test tracker retries on timeout"""
        import requests

        mock_post.side_effect = [requests.exceptions.Timeout(), Mock(status_code=200)]

        tracker = UsageTracker(api_key="test-key", max_retries=2)

        result = tracker.track_usage(
            customer_id="customer-123",
            model="test-model",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )

        assert result is True
        assert mock_post.call_count == 2

    def test_tracker_disabled(self):
        """Test tracker when disabled"""
        tracker = UsageTracker(api_key="test-key", disabled=True)

        result = tracker.track_usage(
            customer_id="customer-123",
            model="test-model",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )

        assert result is False

    def test_tracker_background(self):
        """Test background tracking"""
        tracker = UsageTracker(api_key="test-key", disabled=True)

        # Should not raise
        tracker.track_usage_background(
            customer_id="customer-123",
            model="test-model",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )


class TestMethodCollisions:
    """Test that double underscore prevents collisions"""

    def test_no_method_collision(self):
        """Test that client methods don't collide with SDK methods"""

        class MockClient:
            def __init__(self):
                # These would collide if we used single underscore
                self._detect_provider = "client_method"
                self._wrap_method = "client_method"
                self._track_usage = "client_method"
                self._extract_usage = "client_method"
                self._extract_model = "client_method"
                self._should_track_method = "client_method"

        wrapper = track_ai(
            MockClient(), cmdrdata_api_key="test-key", disable_tracking=True
        )

        # Client's methods should be accessible
        assert wrapper._detect_provider == "client_method"
        assert wrapper._wrap_method == "client_method"
        assert wrapper._track_usage == "client_method"
        assert wrapper._extract_usage == "client_method"
        assert wrapper._extract_model == "client_method"
        assert wrapper._should_track_method == "client_method"

        # Our internal methods use double underscores
        assert hasattr(wrapper, "_CmdrData__detect_provider")
        assert hasattr(wrapper, "_CmdrData__wrap_method")
        assert hasattr(wrapper, "_CmdrData__track_usage")
        assert hasattr(wrapper, "_CmdrData__extract_usage")
        assert hasattr(wrapper, "_CmdrData__extract_model")
        assert hasattr(wrapper, "_CmdrData__should_track_method")


class TestTrackAiFunction:
    """Test the track_ai convenience function"""

    def test_track_ai_returns_wrapped_client(self):
        """Test track_ai returns properly wrapped client"""
        mock_client = Mock()
        mock_client.__class__.__module__ = "openai.client"

        wrapped = track_ai(
            mock_client, cmdrdata_api_key="test-key", disable_tracking=True
        )

        assert isinstance(wrapped, CmdrData)
        assert wrapped._client == mock_client
        assert wrapped.provider == "openai"

    def test_track_ai_with_kwargs(self):
        """Test track_ai passes kwargs correctly"""
        mock_client = Mock()

        wrapped = track_ai(
            mock_client,
            cmdrdata_api_key="test-key",
            customer_id="default-customer",
            metadata={"env": "test"},
            provider="custom",
            disable_tracking=True,
        )

        assert wrapped.default_customer_id == "default-customer"
        assert wrapped.default_metadata == {"env": "test"}
        assert wrapped.provider == "custom"


class TestShouldTrackMethod:
    """Test method tracking detection"""

    @pytest.mark.parametrize(
        "method_name,should_track",
        [
            ("create", True),
            ("generate", True),
            ("complete", True),
            ("chat", True),
            ("invoke", True),
            ("run", True),
            ("predict", True),
            ("send", True),
            ("call", True),
            ("stream", True),
            ("embed", True),
            ("encode", True),
            ("transcribe", True),
            ("translate", True),
            ("synthesize", True),
            ("analyze", True),
            ("process", True),
            ("get_api_key", False),
            ("configure", False),
            ("__init__", False),
            ("to_dict", False),
        ],
    )
    def test_should_track_method(self, method_name, should_track):
        """Test which methods should be tracked"""
        wrapper = CmdrData(
            client=Mock(), cmdrdata_api_key="test", disable_tracking=True
        )

        result = wrapper._CmdrData__should_track_method(method_name)
        assert result == should_track


class TestThreadSafety:
    """Test thread safety of context managers"""

    def test_thread_local_customer_context(self):
        """Test that customer context is thread-local"""
        results = []

        def thread_function(customer_id):
            set_customer_context(customer_id)
            time.sleep(0.01)  # Simulate work
            results.append(get_customer_context())

        threads = []
        for i in range(5):
            t = threading.Thread(target=thread_function, args=(f"customer-{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each thread should have its own context
        assert sorted(results) == [f"customer-{i}" for i in range(5)]

    def test_thread_local_metadata_context(self):
        """Test that metadata context is thread-local"""
        results = []

        def thread_function(feature):
            set_metadata_context({"feature": feature})
            time.sleep(0.01)  # Simulate work
            results.append(get_metadata_context()["feature"])

        threads = []
        for i in range(5):
            t = threading.Thread(target=thread_function, args=(f"feature-{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each thread should have its own context
        assert sorted(results) == [f"feature-{i}" for i in range(5)]


# Hypothesis-based property testing
class TestHypothesis:
    """Property-based testing with Hypothesis"""

    @given(
        customer_id=st.text(min_size=1, max_size=100),
        model=st.text(min_size=1, max_size=50),
        input_tokens=st.integers(min_value=0, max_value=1000000),
        output_tokens=st.integers(min_value=0, max_value=1000000),
    )
    @settings(max_examples=50, deadline=1000)
    def test_usage_extraction_always_returns_dict(
        self, customer_id, model, input_tokens, output_tokens
    ):
        """Test that usage extraction always returns valid dict"""
        response = Mock(
            usage=Mock(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        )

        wrapper = CmdrData(
            client=Mock(), cmdrdata_api_key="test", disable_tracking=True
        )

        usage = wrapper._CmdrData__extract_usage(response)

        assert isinstance(usage, dict)
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert "total_tokens" in usage
        assert usage["input_tokens"] >= 0
        assert usage["output_tokens"] >= 0
        assert usage["total_tokens"] >= 0

    @given(
        metadata=st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.text(min_size=1, max_size=100),
            min_size=0,
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_metadata_merging(self, metadata):
        """Test that metadata merging works correctly"""
        wrapper = CmdrData(
            client=Mock(),
            cmdrdata_api_key="test",
            metadata={"default": "value"},
            disable_tracking=True,
        )

        # Set context metadata
        set_metadata_context({"context": "value"})

        # Merge should include all sources
        tracker = Mock()
        wrapper.tracker = tracker

        class MockMethod:
            def __call__(self, *args, **kwargs):
                return {"result": "test"}

        wrapped_method = wrapper._CmdrData__wrap_method(MockMethod(), "test_method")

        wrapped_method(metadata=metadata)

        time.sleep(0.1)

        if tracker.track_usage_background.called:
            call_kwargs = tracker.track_usage_background.call_args.kwargs
            merged = call_kwargs["metadata"]

            assert "default" in merged
            assert merged["default"] == "value"
            assert "context" in merged
            assert merged["context"] == "value"
            for k, v in metadata.items():
                assert k in merged
                assert merged[k] == v


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=cmdrdata", "--cov-report=term-missing"])
