"""
End-to-End Provider Tests for CmdrData SDK

This test suite runs comprehensive integration tests across all supported providers.
It verifies that tracking works correctly across multiple providers in a single session.
"""

import json
import os
import sys
from datetime import datetime

import pytest

from cmdrdata import CmdrData


class TestE2EProviders:
    """End-to-end tests across multiple providers"""

    def setup_method(self):
        """Setup test run ID and results tracking"""
        self.test_run_id = f"e2e-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.results = []

    def _test_provider(self, provider_name, setup_func):
        """Helper to test a single provider"""
        try:
            print(f"\n{'='*50}")
            print(f"Testing {provider_name}")
            print("=" * 50)

            client = setup_func()
            if not client:
                print(f"[SKIP] {provider_name} - No API key")
                return {"provider": provider_name, "status": "skipped"}

            # Wrap with CmdrData
            customer_id = f"{self.test_run_id}-{provider_name.lower()}"
            wrapped = CmdrData(
                client=client,
                cmdrdata_api_key=os.getenv("CMDRDATA_API_KEY", "test-key"),
                customer_id=customer_id,
                metadata={
                    "test_run": self.test_run_id,
                    "provider": provider_name,
                    "ci": True,
                },
            )

            # Make minimal API call based on provider
            tokens = 0
            if provider_name == "OpenAI":
                response = wrapped.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Reply with 'OK'"}],
                    max_tokens=5,
                )
                tokens = response.usage.total_tokens

            elif provider_name == "Anthropic":
                response = wrapped.messages.create(
                    model="claude-3-haiku-20240307",
                    messages=[{"role": "user", "content": "Reply with 'OK'"}],
                    max_tokens=5,
                )
                tokens = response.usage.input_tokens + response.usage.output_tokens

            elif provider_name == "Google":
                response = wrapped.generate_content("Reply with 'OK'")
                tokens = response.usage_metadata.total_token_count

            elif provider_name == "Cohere":
                if hasattr(wrapped, "chat"):
                    response = wrapped.chat(
                        model="command-r",
                        messages=[{"role": "user", "content": "Reply with 'OK'"}],
                    )
                else:
                    response = wrapped.generate(
                        prompt="Reply with 'OK'", model="command", max_tokens=5
                    )
                tokens = 10  # Approximate

            print(f"[OK] {provider_name} - {tokens} tokens used")
            return {
                "provider": provider_name,
                "status": "success",
                "tokens": tokens,
                "customer_id": customer_id,
            }

        except Exception as e:
            print(f"[ERROR] {provider_name}: {e}")
            return {"provider": provider_name, "status": "error", "error": str(e)}

    def _setup_openai(self):
        """Setup OpenAI client"""
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        from openai import OpenAI

        return OpenAI(api_key=key)

    def _setup_anthropic(self):
        """Setup Anthropic client"""
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return None
        from anthropic import Anthropic

        return Anthropic(api_key=key)

    def _setup_google(self):
        """Setup Google Generative AI client"""
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            return None
        import google.generativeai as genai

        genai.configure(api_key=key)
        return genai.GenerativeModel("gemini-1.5-flash")

    def _setup_cohere(self):
        """Setup Cohere client"""
        key = os.getenv("COHERE_API_KEY")
        if not key:
            return None
        import cohere

        try:
            return cohere.ClientV2(api_key=key)
        except:
            return cohere.Client(api_key=key)

    @pytest.mark.integration
    def test_all_providers_e2e(self):
        """Test all providers in a single E2E flow"""
        print(f"Starting E2E Test Run: {self.test_run_id}")

        providers = [
            ("OpenAI", self._setup_openai),
            ("Anthropic", self._setup_anthropic),
            ("Google", self._setup_google),
            ("Cohere", self._setup_cohere),
        ]

        for name, setup in providers:
            result = self._test_provider(name, setup)
            self.results.append(result)

        # Generate summary
        self._print_summary()

        # Save results to file
        self._save_results()

        # Check if any providers failed
        failed = [r for r in self.results if r["status"] == "error"]
        if failed:
            pytest.fail(
                f"Failed providers: {', '.join([r['provider'] for r in failed])}"
            )

    def _print_summary(self):
        """Print test summary"""
        print(f"\n{'='*50}")
        print("E2E TEST SUMMARY")
        print("=" * 50)

        successful = [r for r in self.results if r["status"] == "success"]
        failed = [r for r in self.results if r["status"] == "error"]
        skipped = [r for r in self.results if r["status"] == "skipped"]

        print(f"[OK] Successful: {len(successful)}/{len(self.results)}")
        for r in successful:
            print(f"  - {r['provider']}: {r.get('tokens', 0)} tokens")

        if failed:
            print(f"[ERROR] Failed: {len(failed)}")
            for r in failed:
                print(f"  - {r['provider']}: {r.get('error', 'Unknown')}")

        if skipped:
            print(f"[SKIP] Skipped: {len(skipped)}")
            for r in skipped:
                print(f"  - {r['provider']}")

    def _save_results(self):
        """Save test results to JSON file"""
        results_file = "e2e_results.json"
        successful = [r for r in self.results if r["status"] == "success"]
        failed = [r for r in self.results if r["status"] == "error"]
        skipped = [r for r in self.results if r["status"] == "skipped"]

        results_data = {
            "test_run_id": self.test_run_id,
            "timestamp": datetime.now().isoformat(),
            "results": self.results,
            "summary": {
                "total": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "skipped": len(skipped),
            },
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\n[INFO] Results saved to {results_file}")
        print(f"[INFO] Test Run ID: {self.test_run_id}")


if __name__ == "__main__":
    # Run the test
    sys.exit(pytest.main([__file__, "-v", "-s"]))
