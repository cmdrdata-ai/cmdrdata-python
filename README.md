# CmdrData - Universal AI Usage Tracking

**One SDK. Any AI Provider. Complete Usage Tracking.**

CmdrData is a universal SDK that wraps ANY AI client (OpenAI, Anthropic, Google, Cohere, etc.) to automatically track usage for billing, analytics, and cost management - with zero code changes required.

## Installation

```bash
pip install cmdrdata
```

## Quick Start

```python
from cmdrdata import track_ai
from openai import OpenAI

# Wrap any AI client
client = track_ai(
    OpenAI(api_key="..."),
    cmdrdata_api_key="cmd-..."
)

# Use exactly like the original - tracking happens automatically
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    customer_id="customer-123",  # Optional: Track by customer
    metadata={"feature": "chat"}  # Optional: Custom metadata
)
```

## Features

### Universal Compatibility
Works with ANY AI provider that returns usage information:
- OpenAI (GPT-4, GPT-3.5, DALL-E, Whisper)
- Anthropic (Claude 3, Claude 2)
- Google (Gemini Pro, PaLM)
- Cohere
- HuggingFace
- Replicate
- Together AI
- Perplexity
- Any custom AI client

### Zero Code Changes
Your existing code works exactly as before:

```python
# Before
from anthropic import Anthropic
client = Anthropic(api_key="...")

# After - Just wrap it!
from cmdrdata import track_ai
client = track_ai(Anthropic(api_key="..."), cmdrdata_api_key="cmd-...")

# All your existing code continues to work unchanged
response = client.messages.create(...)  # Automatically tracked!
```

### Automatic Provider Detection
CmdrData automatically detects the provider type:

```python
from cmdrdata import CmdrData

# Auto-detects as OpenAI
openai_client = CmdrData(client=OpenAI())

# Auto-detects as Anthropic  
anthropic_client = CmdrData(client=Anthropic())

# Auto-detects as Google
gemini_client = CmdrData(client=genai.GenerativeModel("gemini-pro"))
```

### Rich Metadata Support
Track usage by any dimension:

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    customer_id="enterprise-customer",
    metadata={
        "feature": "customer-support",
        "department": "sales",
        "experiment": "prompt-v2",
        "user_tier": "premium",
        "session_id": "abc-123"
    }
)
```

### Thread-Safe Context Management
Perfect for multi-tenant applications:

```python
from cmdrdata import customer_context, metadata_context

# Set context for all API calls in a thread
with customer_context("customer-123"):
    with metadata_context({"feature": "bulk-processing"}):
        # All API calls here automatically include this context
        response1 = client.chat.completions.create(...)
        response2 = client.embeddings.create(...)
```

## Advanced Usage

### Creating Clients
Three ways to initialize:

```python
from cmdrdata import CmdrData, track_ai

# Method 1: Wrap existing client
client = track_ai(OpenAI(api_key="..."), cmdrdata_api_key="cmd-...")

# Method 2: Let CmdrData create the client
client = CmdrData(
    client_class=OpenAI,
    client_kwargs={"api_key": "..."},
    cmdrdata_api_key="cmd-..."
)

# Method 3: Full control
client = CmdrData(
    client=OpenAI(api_key="..."),
    cmdrdata_api_key="cmd-...",
    cmdrdata_url="https://your-instance.cmdrdata.com/api/events",
    provider="openai",  # Optional: manually specify
    customer_id="default-customer",  # Default for all requests
    metadata={"environment": "production"}  # Default metadata
)
```

### Environment Variables
Configure defaults via environment:

```bash
export CMDRDATA_API_KEY="cmd-..."
export CMDRDATA_URL="https://api.cmdrdata.com/api/events"
export CMDRDATA_CUSTOMER_ID="default-customer"
```

Then simply:
```python
from cmdrdata import track_ai
client = track_ai(OpenAI())  # Uses env vars automatically
```

### Error Handling
Tracking failures never break your application:

```python
# Even if CmdrData is unreachable, your AI calls continue working
response = client.chat.completions.create(...)  # Always succeeds

# Tracking errors are logged but not raised
# Check logs for tracking issues
```

### Async Support
Full async/await support:

```python
import asyncio
from openai import AsyncOpenAI
from cmdrdata import track_ai

async def main():
    client = track_ai(AsyncOpenAI(), cmdrdata_api_key="cmd-...")
    
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )

asyncio.run(main())
```

## What Gets Tracked

CmdrData automatically extracts and tracks:
- **Token Usage**: Input, output, and total tokens
- **Model Information**: Which model was used
- **Provider**: OpenAI, Anthropic, Google, etc.
- **Customer ID**: For billing and analytics
- **Custom Metadata**: Any dimensions you specify
- **Timing**: Request duration
- **Errors**: Failed requests with error details

## Use Cases

### Usage-Based Billing
```python
# Track usage per customer for accurate billing
response = client.chat.completions.create(
    ...,
    customer_id="customer-123"
)
```

### Feature Analytics
```python
# Understand which features consume the most tokens
response = client.chat.completions.create(
    ...,
    metadata={"feature": "document-summary", "feature_version": "2.0"}
)
```

### Department Cost Allocation
```python
# Allocate AI costs to specific departments
response = client.chat.completions.create(
    ...,
    metadata={"department": "engineering", "team": "ml-platform"}
)
```

### A/B Testing
```python
# Track token usage across experiments
response = client.chat.completions.create(
    ...,
    metadata={"experiment": "prompt-optimization", "variant": "b"}
)
```

### Multi-Tenancy
```python
# Track usage in multi-tenant applications
with customer_context(tenant_id):
    response = client.chat.completions.create(...)
```

## Supported Providers

CmdrData works with any AI provider that returns usage information. Tested with:

- **OpenAI**: GPT-4, GPT-3.5, DALL-E, Whisper, Embeddings
- **Anthropic**: Claude 3 Opus/Sonnet/Haiku, Claude 2
- **Google**: Gemini Pro, Gemini Ultra, PaLM
- **Cohere**: Command, Embed, Rerank
- **HuggingFace**: Inference API, Transformers
- **Replicate**: Any model on Replicate
- **Together AI**: Llama, Mistral, and other models
- **Perplexity**: pplx-api models
- **Custom**: Any client that returns token usage

## Migration Guide

### From Individual SDKs
If you're using cmdrdata-openai, cmdrdata-anthropic, or cmdrdata-gemini:

```python
# Old way (provider-specific SDKs)
from cmdrdata_openai import TrackedOpenAI
from cmdrdata_anthropic import TrackedAnthropic

# New way (universal SDK)
from cmdrdata import track_ai
openai_client = track_ai(OpenAI())
anthropic_client = track_ai(Anthropic())
```

### From Direct Clients
Zero changes to your application code:

```python
# Your existing code
client = OpenAI(api_key="...")
response = client.chat.completions.create(...)

# Just wrap the client - everything else stays the same
from cmdrdata import track_ai
client = track_ai(OpenAI(api_key="..."), cmdrdata_api_key="cmd-...")
response = client.chat.completions.create(...)  # Now with tracking!
```

## Configuration

### API Endpoint
Default: `https://api.cmdrdata.com/api/events`

Custom endpoint:
```python
client = track_ai(
    OpenAI(),
    cmdrdata_api_key="cmd-...",
    cmdrdata_url="https://your-instance.cmdrdata.com/api/events"
)
```

### Disable Tracking
For testing or development:
```python
client = CmdrData(
    client=OpenAI(),
    disable_tracking=True  # AI calls work, no tracking occurs
)
```

## License

MIT

## Support

- GitHub Issues: [github.com/cmdrdata-ai/cmdrdata](https://github.com/cmdrdata-ai/cmdrdata)
- Documentation: [docs.cmdrdata.ai](https://docs.cmdrdata.ai)
- Email: support@cmdrdata.ai

---

Built with ❤️ for developers who need to track AI usage without the hassle.