# Contributing to CmdrData SDK

Thank you for your interest in contributing to the CmdrData SDK! This universal AI tracking library helps developers monitor and analyze their AI usage across any provider.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cmdrdata/cmdrdata-sdk.git
   cd cmdrdata-sdk
   ```

2. **Install uv (recommended package manager):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies:**
   ```bash
   uv pip install -e ".[dev]"
   ```

## Running Tests

We maintain comprehensive test coverage. Please ensure all tests pass before submitting a PR:

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=cmdrdata --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_comprehensive.py -v

# Run tests for specific provider patterns
uv run pytest tests/test_comprehensive.py::TestProviderDetection -v
```

## Code Quality

We use several tools to maintain code quality:

```bash
# Format code with black
uv run black cmdrdata tests

# Sort imports
uv run isort cmdrdata tests

# Type checking
uv run mypy cmdrdata --strict

# Run all checks
uv run black cmdrdata tests && uv run isort cmdrdata tests && uv run mypy cmdrdata --strict
```

## Adding Support for New AI Providers

To add support for a new AI provider:

1. **Update provider detection** in `cmdrdata/client.py`:
   ```python
   # In __detect_provider method
   elif "newprovider" in module:
       return "newprovider"
   ```

2. **Add usage extraction pattern** in `cmdrdata/client.py`:
   ```python
   # In __extract_usage method
   elif hasattr(response, "new_usage_field"):
       # Extract usage based on provider's response format
   ```

3. **Add tests** in `tests/test_comprehensive.py`:
   - Provider detection test
   - Usage extraction test
   - Integration test

## Testing with Real AI Providers

While most tests use mocks, you can test with real providers:

```python
# Example: test_real_providers.py
from cmdrdata import track_ai
from openai import OpenAI  # or any other provider

client = track_ai(
    OpenAI(api_key="your-key"),
    cmdrdata_api_key="your-cmdrdata-key"
)

# Make actual API calls for testing
```

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Write tests** for any new functionality
3. **Ensure all tests pass** with `uv run pytest tests/`
4. **Format your code** with `uv run black cmdrdata tests`
5. **Update documentation** if needed
6. **Submit a pull request** with a clear description

## Commit Message Convention

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Examples:
```
feat: add support for Mistral AI provider
fix: correct usage extraction for Anthropic responses
docs: update README with new provider examples
test: add comprehensive tests for thread safety
```

## Python Version Support

We support Python 3.9 through 3.13. Please ensure your code works across all versions.

## Questions?

Feel free to open an issue for any questions or discussions about potential contributions.