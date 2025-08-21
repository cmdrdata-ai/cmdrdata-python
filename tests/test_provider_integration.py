"""
Provider Integration Tests for CmdrData SDK

These tests verify that CmdrData correctly wraps and tracks usage for various AI providers.
They are designed to run in CI/CD with real API keys when available.
"""

import os
import sys
import pytest
from datetime import datetime
from unittest.mock import Mock

from cmdrdata import CmdrData


class TestProviderIntegration:
    """Test individual provider integrations with CmdrData wrapper"""

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key")
    def test_openai_integration(self):
        """Test OpenAI SDK integration"""
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        test_id = f"ci-test-openai-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        wrapped_client = CmdrData(
            client=client,
            cmdrdata_api_key=os.getenv("CMDRDATA_API_KEY", "test-key"),
            customer_id=test_id,
            metadata={
                "test_type": "ci_integration",
                "provider": "OpenAI",
                "github_run_id": os.getenv("GITHUB_RUN_ID", "local"),
            },
        )

        # Make a minimal API call
        response = wrapped_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Say 'OK'"}], max_tokens=5
        )

        assert response.choices[0].message.content
        assert response.usage.total_tokens > 0
        print(f"[OK] OpenAI: {response.usage.total_tokens} tokens used")

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No Anthropic API key")
    def test_anthropic_integration(self):
        """Test Anthropic SDK integration"""
        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        test_id = f"ci-test-anthropic-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        wrapped_client = CmdrData(
            client=client,
            cmdrdata_api_key=os.getenv("CMDRDATA_API_KEY", "test-key"),
            customer_id=test_id,
            metadata={
                "test_type": "ci_integration",
                "provider": "Anthropic",
                "github_run_id": os.getenv("GITHUB_RUN_ID", "local"),
            },
        )

        # Make a minimal API call
        response = wrapped_client.messages.create(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Say 'OK'"}],
            max_tokens=5,
        )

        assert response.content[0].text
        total_tokens = response.usage.input_tokens + response.usage.output_tokens
        assert total_tokens > 0
        print(f"[OK] Anthropic: {total_tokens} tokens used")

    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="No Google API key")
    def test_google_integration(self):
        """Test Google Generative AI SDK integration"""
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        client = genai.GenerativeModel("gemini-1.5-flash")
        test_id = f"ci-test-google-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        wrapped_client = CmdrData(
            client=client,
            cmdrdata_api_key=os.getenv("CMDRDATA_API_KEY", "test-key"),
            customer_id=test_id,
            metadata={
                "test_type": "ci_integration",
                "provider": "Google",
                "github_run_id": os.getenv("GITHUB_RUN_ID", "local"),
            },
        )

        # Make a minimal API call
        response = wrapped_client.generate_content("Say 'OK'")

        assert response.text
        assert response.usage_metadata.total_token_count > 0
        print(f"[OK] Google: {response.usage_metadata.total_token_count} tokens used")

    @pytest.mark.skipif(not os.getenv("COHERE_API_KEY"), reason="No Cohere API key")
    def test_cohere_integration(self):
        """Test Cohere SDK integration"""
        import cohere

        # Try both V2 and V1 clients
        try:
            client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
            is_v2 = True
        except:
            client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
            is_v2 = False

        test_id = f"ci-test-cohere-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        wrapped_client = CmdrData(
            client=client,
            cmdrdata_api_key=os.getenv("CMDRDATA_API_KEY", "test-key"),
            customer_id=test_id,
            metadata={
                "test_type": "ci_integration",
                "provider": "Cohere",
                "version": "v2" if is_v2 else "v1",
                "github_run_id": os.getenv("GITHUB_RUN_ID", "local"),
            },
        )

        # Make a minimal API call based on client version
        if is_v2 and hasattr(wrapped_client, "chat"):
            response = wrapped_client.chat(
                model="command-r", messages=[{"role": "user", "content": "Say 'OK'"}]
            )
        else:
            response = wrapped_client.generate(prompt="Say 'OK'", model="command", max_tokens=5)

        print(f"[OK] Cohere ({'v2' if is_v2 else 'v1'}): Response received")



if __name__ == "__main__":
    # Run tests with pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))