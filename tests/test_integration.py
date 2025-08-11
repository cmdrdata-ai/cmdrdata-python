#!/usr/bin/env python3
"""
Integration tests for CmdrData SDK
Tests actual data collection and API communication
"""

import os
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
import httpx
import pytest

from cmdrdata import track_ai, CmdrData, customer_context, metadata_context
from cmdrdata.tracker import UsageTracker


class MockCmdrDataServer:
    """Mock CmdrData API server for testing"""
    
    def __init__(self):
        self.received_events: List[Dict[str, Any]] = []
        self.response_status = 200
        self.response_delay = 0
        self.auth_tokens = {"test-api-key", "valid-key", "cmd-test-key"}
    
    def handle_request(self, request_data: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Simulate CmdrData API handling"""
        # Check authentication
        auth_header = headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return {"status": 401, "error": "Missing authorization"}
        
        token = auth_header.replace("Bearer ", "")
        if token not in self.auth_tokens:
            return {"status": 401, "error": "Invalid API key"}
        
        # Validate required fields
        required_fields = ["customer_id", "model", "provider"]
        for field in required_fields:
            if field not in request_data:
                return {"status": 400, "error": f"Missing required field: {field}"}
        
        # Store the event
        event = {
            **request_data,
            "timestamp": datetime.utcnow().isoformat(),
            "api_key": token
        }
        self.received_events.append(event)
        
        # Simulate delay if configured
        if self.response_delay > 0:
            time.sleep(self.response_delay)
        
        return {"status": self.response_status, "event_id": f"evt_{len(self.received_events)}"}
    
    def get_events_for_customer(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get all events for a specific customer"""
        return [e for e in self.received_events if e.get("customer_id") == customer_id]
    
    def clear_events(self):
        """Clear all received events"""
        self.received_events = []


class TestIntegrationBasic:
    """Basic integration tests with mock server"""
    
    @pytest.fixture
    def mock_server(self):
        """Create a mock CmdrData server"""
        return MockCmdrDataServer()
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI-like client"""
        class MockOpenAI:
            def __init__(self, api_key=None):
                self.api_key = api_key
                self.chat = Mock()
                self.chat.completions = Mock()
                self.chat.completions.create = self._create_completion
            
            def _create_completion(self, **kwargs):
                response = Mock()
                response.model = kwargs.get("model", "gpt-3.5-turbo")
                response.usage = Mock(
                    prompt_tokens=50,
                    completion_tokens=100,
                    total_tokens=150
                )
                response.choices = [Mock(message=Mock(content="Test response"))]
                return response
        
        MockOpenAI.__module__ = "openai"
        return MockOpenAI
    
    @patch("cmdrdata.tracker.requests.post")
    def test_basic_tracking_flow(self, mock_post, mock_server, mock_openai_client):
        """Test basic tracking flow with mock server"""
        # Configure mock to use our server
        def post_handler(url, json=None, headers=None, timeout=None):
            response = Mock()
            result = mock_server.handle_request(json, headers)
            response.status_code = result["status"]
            response.json.return_value = result
            return response
        
        mock_post.side_effect = post_handler
        
        # Create wrapped client
        client = track_ai(
            mock_openai_client(api_key="test-openai-key"),
            cmdrdata_api_key="test-api-key"
        )
        
        # Make API call
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            customer_id="customer-123",
            metadata={"session": "abc", "feature": "chat"}
        )
        
        # Wait for background tracking
        time.sleep(0.2)
        
        # Verify event was received
        events = mock_server.get_events_for_customer("customer-123")
        assert len(events) == 1
        
        event = events[0]
        assert event["model"] == "gpt-4"
        assert event["provider"] == "openai"
        assert event["input_tokens"] == 50
        assert event["output_tokens"] == 100
        assert event["total_tokens"] == 150
        assert event["metadata"]["session"] == "abc"
        assert event["metadata"]["feature"] == "chat"
    
    @patch("cmdrdata.tracker.requests.post")
    def test_multiple_providers(self, mock_post, mock_server):
        """Test tracking multiple AI providers"""
        def post_handler(url, json=None, headers=None, timeout=None):
            response = Mock()
            result = mock_server.handle_request(json, headers)
            response.status_code = result["status"]
            return response
        
        mock_post.side_effect = post_handler
        
        # Create different mock providers
        providers = {
            "openai": ("openai.client", "gpt-4"),
            "anthropic": ("anthropic.client", "claude-3"),
            "google": ("google.generativeai", "gemini-pro")
        }
        
        for provider_name, (module, model) in providers.items():
            # Create mock client
            class MockClient:
                def generate(self, prompt):
                    return {
                        "text": f"Response from {provider_name}",
                        "model": model,
                        "usage": Mock(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30
                        )
                    }
            
            MockClient.__module__ = module
            
            # Wrap and track
            client = track_ai(
                MockClient(),
                cmdrdata_api_key="test-api-key"
            )
            
            # Make call
            response = client.generate(
                f"Test {provider_name}",
                customer_id=f"customer-{provider_name}",
                metadata={"provider": provider_name}
            )
            
            # Wait for tracking
            time.sleep(0.1)
        
        # Verify all events received
        time.sleep(0.2)
        assert len(mock_server.received_events) == 3
        
        # Check each provider was tracked correctly
        for event in mock_server.received_events:
            provider = event["metadata"]["provider"]
            assert event["provider"] in ["openai", "anthropic", "google"]
            assert event["customer_id"] == f"customer-{provider}"
            assert event["total_tokens"] == 30
    
    @patch("cmdrdata.tracker.requests.post")
    def test_context_managers(self, mock_post, mock_server, mock_openai_client):
        """Test context managers for customer and metadata"""
        def post_handler(url, json=None, headers=None, timeout=None):
            response = Mock()
            result = mock_server.handle_request(json, headers)
            response.status_code = result["status"]
            return response
        
        mock_post.side_effect = post_handler
        
        client = track_ai(
            mock_openai_client(),
            cmdrdata_api_key="test-api-key"
        )
        
        # Test customer context
        with customer_context("context-customer"):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}]
            )
        
        # Test metadata context
        with metadata_context({"env": "production", "version": "2.0"}):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                customer_id="metadata-customer"
            )
        
        # Test nested contexts
        with customer_context("nested-customer"):
            with metadata_context({"level": "nested"}):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Test"}],
                    metadata={"additional": "data"}
                )
        
        time.sleep(0.3)
        
        # Verify all contexts worked
        assert len(mock_server.received_events) == 3
        
        # Check first event (customer context)
        assert mock_server.received_events[0]["customer_id"] == "context-customer"
        
        # Check second event (metadata context)
        assert mock_server.received_events[1]["customer_id"] == "metadata-customer"
        assert mock_server.received_events[1]["metadata"]["env"] == "production"
        assert mock_server.received_events[1]["metadata"]["version"] == "2.0"
        
        # Check third event (nested contexts)
        assert mock_server.received_events[2]["customer_id"] == "nested-customer"
        assert mock_server.received_events[2]["metadata"]["level"] == "nested"
        assert mock_server.received_events[2]["metadata"]["additional"] == "data"
    
    @patch("cmdrdata.tracker.requests.post")
    def test_error_tracking(self, mock_post, mock_server):
        """Test that errors are tracked properly"""
        def post_handler(url, json=None, headers=None, timeout=None):
            response = Mock()
            result = mock_server.handle_request(json, headers)
            response.status_code = result["status"]
            return response
        
        mock_post.side_effect = post_handler
        
        class FailingClient:
            def generate(self, prompt):
                raise ValueError("API Error: Rate limit exceeded")
        
        client = track_ai(
            FailingClient(),
            cmdrdata_api_key="test-api-key"
        )
        
        # Call should raise but still track
        with pytest.raises(ValueError) as exc:
            client.generate("test", customer_id="error-customer")
        
        assert "Rate limit exceeded" in str(exc.value)
        
        time.sleep(0.2)
        
        # Verify error was tracked
        events = mock_server.get_events_for_customer("error-customer")
        assert len(events) == 1
        
        event = events[0]
        assert event["error_occurred"] is True
        assert event["error_type"] == "ValueError"
        assert "Rate limit exceeded" in event["error_message"]
    
    @patch("cmdrdata.tracker.requests.post")
    def test_retry_mechanism(self, mock_post, mock_server, mock_openai_client):
        """Test retry mechanism for failed requests"""
        call_count = [0]
        
        def post_handler(url, json=None, headers=None, timeout=None):
            call_count[0] += 1
            response = Mock()
            
            # Fail first two attempts, succeed on third
            if call_count[0] < 3:
                response.status_code = 500
                response.json.return_value = {"error": "Server error"}
            else:
                result = mock_server.handle_request(json, headers)
                response.status_code = result["status"]
                response.json.return_value = result
            
            return response
        
        mock_post.side_effect = post_handler
        
        client = track_ai(
            mock_openai_client(),
            cmdrdata_api_key="test-api-key"
        )
        
        # Make API call
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            customer_id="retry-customer"
        )
        
        time.sleep(0.5)  # Allow time for retries
        
        # Should have retried and succeeded
        assert call_count[0] == 3
        events = mock_server.get_events_for_customer("retry-customer")
        assert len(events) == 1


class TestIntegrationRealProviders:
    """Integration tests with real provider response formats"""
    
    @patch("cmdrdata.tracker.requests.post")
    def test_openai_response_format(self, mock_post):
        """Test with realistic OpenAI response format"""
        received_events = []
        
        def capture_event(url, json=None, headers=None, timeout=None):
            received_events.append(json)
            response = Mock()
            response.status_code = 200
            return response
        
        mock_post.side_effect = capture_event
        
        # Create realistic OpenAI response
        class RealisticOpenAI:
            class Chat:
                class Completions:
                    def create(self, **kwargs):
                        response = Mock()
                        response.id = "chatcmpl-123"
                        response.model = "gpt-4-0613"
                        response.created = 1234567890
                        response.usage = Mock()
                        response.usage.prompt_tokens = 127
                        response.usage.completion_tokens = 453
                        response.usage.total_tokens = 580
                        response.choices = [
                            Mock(
                                index=0,
                                message=Mock(
                                    role="assistant",
                                    content="This is a detailed response from GPT-4."
                                ),
                                finish_reason="stop"
                            )
                        ]
                        return response
                
                def __init__(self):
                    self.completions = self.Completions()
            
            def __init__(self):
                self.chat = self.Chat()
        
        RealisticOpenAI.__module__ = "openai"
        
        client = track_ai(
            RealisticOpenAI(),
            cmdrdata_api_key="test-api-key"
        )
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing in simple terms."}
            ],
            temperature=0.7,
            customer_id="quantum-customer",
            metadata={
                "feature": "education",
                "topic": "quantum",
                "user_level": "beginner"
            }
        )
        
        time.sleep(0.2)
        
        # Verify tracked data
        assert len(received_events) == 1
        event = received_events[0]
        
        assert event["customer_id"] == "quantum-customer"
        assert event["model"] == "gpt-4"
        assert event["provider"] == "openai"
        assert event["input_tokens"] == 127
        assert event["output_tokens"] == 453
        assert event["total_tokens"] == 580
        assert event["metadata"]["feature"] == "education"
        assert event["metadata"]["topic"] == "quantum"
        assert event["metadata"]["user_level"] == "beginner"
    
    @patch("cmdrdata.tracker.requests.post")
    def test_anthropic_response_format(self, mock_post):
        """Test with realistic Anthropic response format"""
        received_events = []
        
        def capture_event(url, json=None, headers=None, timeout=None):
            received_events.append(json)
            response = Mock()
            response.status_code = 200
            return response
        
        mock_post.side_effect = capture_event
        
        # Create realistic Anthropic response
        class RealisticAnthropic:
            class Messages:
                def create(self, **kwargs):
                    response = Mock(spec=['id', 'type', 'role', 'content', 'model', 'usage'])
                    response.id = "msg_123"
                    response.type = "message"
                    response.role = "assistant"
                    response.content = [
                        Mock(type="text", text="Claude's detailed response here.")
                    ]
                    response.model = "claude-3-opus-20240229"
                    response.usage = Mock(spec=['input_tokens', 'output_tokens'])
                    response.usage.input_tokens = 245
                    response.usage.output_tokens = 189
                    return response
            
            def __init__(self):
                self.messages = self.Messages()
        
        RealisticAnthropic.__module__ = "anthropic"
        
        client = track_ai(
            RealisticAnthropic(),
            cmdrdata_api_key="test-api-key"
        )
        
        response = client.messages.create(
            model="claude-3-opus-20240229",
            messages=[
                {"role": "user", "content": "What are the key principles of clean code?"}
            ],
            max_tokens=1000,
            customer_id="clean-code-customer",
            metadata={
                "feature": "code-review",
                "language": "python",
                "experience": "senior"
            }
        )
        
        time.sleep(0.2)
        
        # Verify tracked data
        assert len(received_events) == 1
        event = received_events[0]
        
        assert event["customer_id"] == "clean-code-customer"
        assert event["model"] == "claude-3-opus-20240229"
        assert event["provider"] == "anthropic"
        assert event["input_tokens"] == 245
        assert event["output_tokens"] == 189
        assert event["total_tokens"] == 434
        assert event["metadata"]["feature"] == "code-review"


class TestIntegrationPerformance:
    """Performance and reliability tests"""
    
    @patch("cmdrdata.tracker.requests.post")
    def test_high_volume_tracking(self, mock_post):
        """Test tracking high volume of requests"""
        received_events = []
        
        def capture_event(url, json=None, headers=None, timeout=None):
            received_events.append(json)
            response = Mock()
            response.status_code = 200
            return response
        
        mock_post.side_effect = capture_event
        
        class FastClient:
            def generate(self, i):
                return {
                    "text": f"Response {i}",
                    "usage": Mock(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30
                    )
                }
        
        client = track_ai(
            FastClient(),
            cmdrdata_api_key="test-api-key"
        )
        
        # Generate many requests quickly
        num_requests = 50
        for i in range(num_requests):
            client.generate(
                i,
                customer_id=f"customer-{i % 5}",  # 5 different customers
                metadata={"batch": i // 10}  # Group in batches
            )
        
        # Wait for all background tracking
        time.sleep(1.0)
        
        # Verify all events were tracked
        assert len(received_events) == num_requests
        
        # Verify customer distribution
        customer_counts = {}
        for event in received_events:
            cid = event["customer_id"]
            customer_counts[cid] = customer_counts.get(cid, 0) + 1
        
        # Should have 10 events per customer
        for count in customer_counts.values():
            assert count == 10
    
    @patch("cmdrdata.tracker.requests.post")
    def test_concurrent_tracking(self, mock_post):
        """Test concurrent tracking from multiple threads"""
        received_events = []
        lock = threading.Lock()
        
        def capture_event(url, json=None, headers=None, timeout=None):
            with lock:
                received_events.append(json)
            response = Mock()
            response.status_code = 200
            return response
        
        mock_post.side_effect = capture_event
        
        class ThreadSafeClient:
            def generate(self, thread_id, call_id):
                return {
                    "text": f"Response from thread {thread_id}, call {call_id}",
                    "usage": Mock(
                        prompt_tokens=5,
                        completion_tokens=10,
                        total_tokens=15
                    )
                }
        
        client = track_ai(
            ThreadSafeClient(),
            cmdrdata_api_key="test-api-key"
        )
        
        def make_requests(thread_id):
            for i in range(10):
                client.generate(
                    thread_id,
                    i,
                    customer_id=f"thread-{thread_id}",
                    metadata={"thread": thread_id, "call": i}
                )
        
        # Start multiple threads
        threads = []
        num_threads = 5
        for i in range(num_threads):
            t = threading.Thread(target=make_requests, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Wait for background tracking
        time.sleep(1.0)
        
        # Verify all events tracked
        assert len(received_events) == num_threads * 10
        
        # Verify no data corruption
        for event in received_events:
            thread_id = event["metadata"]["thread"]
            assert event["customer_id"] == f"thread-{thread_id}"
            assert event["total_tokens"] == 15


class TestIntegrationEdgeCases:
    """Test edge cases and error conditions"""
    
    @patch("cmdrdata.tracker.requests.post")
    def test_missing_usage_data(self, mock_post):
        """Test handling responses without usage data"""
        received_events = []
        
        def capture_event(url, json=None, headers=None, timeout=None):
            received_events.append(json)
            response = Mock()
            response.status_code = 200
            return response
        
        mock_post.side_effect = capture_event
        
        class NoUsageClient:
            def generate(self, prompt):
                # Response without usage data
                return {"text": "Response without usage"}
        
        client = track_ai(
            NoUsageClient(),
            cmdrdata_api_key="test-api-key"
        )
        
        response = client.generate("test", customer_id="no-usage-customer")
        
        time.sleep(0.2)
        
        # Should still track with zero tokens
        assert len(received_events) == 1
        event = received_events[0]
        assert event["customer_id"] == "no-usage-customer"
        assert event["input_tokens"] == 0
        assert event["output_tokens"] == 0
        assert event["total_tokens"] == 0
    
    @patch("cmdrdata.tracker.requests.post")
    def test_network_timeout(self, mock_post):
        """Test handling network timeouts"""
        import requests
        
        call_count = [0]
        
        def timeout_then_succeed(url, json=None, headers=None, timeout=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise requests.exceptions.Timeout("Connection timeout")
            response = Mock()
            response.status_code = 200
            return response
        
        mock_post.side_effect = timeout_then_succeed
        
        class TimeoutTestClient:
            def generate(self, prompt):
                return {
                    "text": "Success after timeout",
                    "usage": Mock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
                }
        
        client = track_ai(
            TimeoutTestClient(),
            cmdrdata_api_key="test-api-key"
        )
        
        # Should not raise despite timeout
        response = client.generate("test", customer_id="timeout-customer")
        
        time.sleep(0.5)
        
        # Should have retried
        assert call_count[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])