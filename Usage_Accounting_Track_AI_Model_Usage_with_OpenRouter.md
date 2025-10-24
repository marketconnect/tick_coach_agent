# Usage Accounting | Track AI Model Usage with OpenRouter | OpenRouter | Documentation

[![Logo](https://files.buildwithfern.com/openrouter.docs.buildwithfern.com/docs/2025-10-24T05:21:00.626Z/content/assets/logo.svg)![Logo](https://files.buildwithfern.com/openrouter.docs.buildwithfern.com/docs/2025-10-24T05:21:00.626Z/content/assets/logo-white.svg)](https://openrouter.ai/)

Search

/

Ask AI

-   Overview
    
    -   [Quickstart](https://openrouter.ai/docs/quickstart)
    -   [FAQ](https://openrouter.ai/docs/faq)
    -   [Principles](https://openrouter.ai/docs/overview/principles)
    -   [Models](https://openrouter.ai/docs/overview/models)
    -   [Enterprise](https://openrouter.ai/enterprise)
-   Features
    
    -   [Privacy and Logging](https://openrouter.ai/docs/features/privacy-and-logging)
    -   [Zero Data Retention (ZDR)](https://openrouter.ai/docs/features/zdr)
    -   [Model Routing](https://openrouter.ai/docs/features/model-routing)
    -   [Provider Routing](https://openrouter.ai/docs/features/provider-routing)
    -   [Exacto Variant](https://openrouter.ai/docs/features/exacto-variant)
    -   [Latency and Performance](https://openrouter.ai/docs/features/latency-and-performance)
    -   [Presets](https://openrouter.ai/docs/features/presets)
    -   [Prompt Caching](https://openrouter.ai/docs/features/prompt-caching)
    -   [Structured Outputs](https://openrouter.ai/docs/features/structured-outputs)
    -   [Tool Calling](https://openrouter.ai/docs/features/tool-calling)
    -   Multimodal
        
    -   [Message Transforms](https://openrouter.ai/docs/features/message-transforms)
    -   [Uptime Optimization](https://openrouter.ai/docs/features/uptime-optimization)
    -   [Web Search](https://openrouter.ai/docs/features/web-search)
    -   [Zero Completion Insurance](https://openrouter.ai/docs/features/zero-completion-insurance)
    -   [Provisioning API Keys](https://openrouter.ai/docs/features/provisioning-api-keys)
    -   [App Attribution](https://openrouter.ai/docs/app-attribution)
-   API Reference
    
    -   [Overview](https://openrouter.ai/docs/api-reference/overview)
    -   [Streaming](https://openrouter.ai/docs/api-reference/streaming)
    -   [Limits](https://openrouter.ai/docs/api-reference/limits)
    -   [Authentication](https://openrouter.ai/docs/api-reference/authentication)
    -   [Parameters](https://openrouter.ai/docs/api-reference/parameters)
    -   [Errors](https://openrouter.ai/docs/api-reference/errors)
    -   Responses API
        
    -   beta.responses
        
    -   Analytics
        
    -   Credits
        
    -   Generations
        
    -   Models
        
    -   Endpoints
        
    -   Parameters
        
    -   Providers
        
    -   API Keys
        
    -   Chat
        
    -   Completions
        
-   Use Cases
    
    -   [BYOK](https://openrouter.ai/docs/use-cases/byok)
    -   [Crypto API](https://openrouter.ai/docs/use-cases/crypto-api)
    -   [OAuth PKCE](https://openrouter.ai/docs/use-cases/oauth-pkce)
    -   [MCP Servers](https://openrouter.ai/docs/use-cases/mcp-servers)
    -   [Organization Management](https://openrouter.ai/docs/use-cases/organization-management)
    -   [For Providers](https://openrouter.ai/docs/use-cases/for-providers)
    -   [Reasoning Tokens](https://openrouter.ai/docs/use-cases/reasoning-tokens)
    -   [Usage Accounting](https://openrouter.ai/docs/use-cases/usage-accounting)
    -   [User Tracking](https://openrouter.ai/docs/use-cases/user-tracking)
-   Community
    
    -   [Frameworks and Integrations Overview](https://openrouter.ai/docs/community/frameworks-and-integrations-overview)
    -   [Effect AI SDK](https://openrouter.ai/docs/community/effect-ai-sdk)
    -   [Arize](https://openrouter.ai/docs/community/arize)
    -   [LangChain](https://openrouter.ai/docs/community/lang-chain)
    -   [Langfuse](https://openrouter.ai/docs/community/langfuse)
    -   [Mastra](https://openrouter.ai/docs/community/mastra)
    -   [OpenAI SDK](https://openrouter.ai/docs/community/open-ai-sdk)
    -   [PydanticAI](https://openrouter.ai/docs/community/pydantic-ai)
    -   [Vercel AI SDK](https://openrouter.ai/docs/community/vercel-ai-sdk)
    -   [Xcode](https://openrouter.ai/docs/community/xcode)
    -   [Zapier](https://openrouter.ai/docs/community/zapier)
    -   [Discord](https://discord.gg/openrouter)

[API](https://openrouter.ai/docs/api-reference/overview)[Models](https://openrouter.ai/models)[Chat](https://openrouter.ai/chat)[Ranking](https://openrouter.ai/rankings)

System

On this page

-   [Usage Information](https://openrouter.ai/docs/use-cases/usage-accounting#usage-information)

-   [Enabling Usage Accounting](https://openrouter.ai/docs/use-cases/usage-accounting#enabling-usage-accounting)
-   [Response Format](https://openrouter.ai/docs/use-cases/usage-accounting#response-format)

-   [Cost Breakdown](https://openrouter.ai/docs/use-cases/usage-accounting#cost-breakdown)
-   [Benefits](https://openrouter.ai/docs/use-cases/usage-accounting#benefits)

-   [Best Practices](https://openrouter.ai/docs/use-cases/usage-accounting#best-practices)
-   [Alternative: Getting Usage via Generation ID](https://openrouter.ai/docs/use-cases/usage-accounting#alternative-getting-usage-via-generation-id)

-   [Examples](https://openrouter.ai/docs/use-cases/usage-accounting#examples)
-   [Basic Usage with Token Tracking](https://openrouter.ai/docs/use-cases/usage-accounting#basic-usage-with-token-tracking)

-   [Streaming with Usage Information](https://openrouter.ai/docs/use-cases/usage-accounting#streaming-with-usage-information)

[Use Cases](https://openrouter.ai/docs/use-cases/byok)

# Usage Accounting

Copy page

The OpenRouter API provides built-in **Usage Accounting** that allows you to track AI model usage without making additional API calls. This feature provides detailed information about token counts, costs, and caching status directly in your API responses.

## Usage Information

When enabled, the API will return detailed usage information including:

1.  Prompt and completion token counts using the model’s native tokenizer
2.  Cost in credits
3.  Reasoning token counts (if applicable)
4.  Cached token counts (if available)

This information is included in the last SSE message for streaming responses, or in the complete response for non-streaming requests.

## Enabling Usage Accounting

You can enable usage accounting in your requests by including the `usage` parameter:

`   1  {  2    "model": "your-model",  3    "messages": [],  4    "usage": {  5      "include": true  6    }  7  }           `

## Response Format

When usage accounting is enabled, the response will include a `usage` object with detailed token information:

`   1  {  2    "object": "chat.completion.chunk",  3    "usage": {  4      "completion_tokens": 2,  5      "completion_tokens_details": {  6        "reasoning_tokens": 0  7      },  8      "cost": 0.95,  9      "cost_details": {  10        "upstream_inference_cost": 19  11      },  12      "prompt_tokens": 194,  13      "prompt_tokens_details": {  14        "cached_tokens": 0,  15        "audio_tokens": 0  16      },  17      "total_tokens": 196  18    }  19  }           `

`cached_tokens` is the number of tokens that were *read* from the cache. At this point in time, we do not support retrieving the number of tokens that were *written* to the cache.

## Cost Breakdown

The usage response includes detailed cost information:

-   `cost`: The total amount charged to your account

-   `cost_details.upstream_inference_cost`: The actual cost charged by the upstream AI provider

**Note:** The `upstream_inference_cost` field only applies to BYOK (Bring Your Own Key) requests.

##### Performance Impact

Enabling usage accounting will add a few hundred milliseconds to the last response as the API calculates token counts and costs. This only affects the final message and does not impact overall streaming performance.

## Benefits

1.  **Efficiency**: Get usage information without making separate API calls
2.  **Accuracy**: Token counts are calculated using the model’s native tokenizer
3.  **Transparency**: Track costs and cached token usage in real-time
4.  **Detailed Breakdown**: Separate counts for prompt, completion, reasoning, and cached tokens

## Best Practices

1.  Enable usage tracking when you need to monitor token consumption or costs
2.  Account for the slight delay in the final response when usage accounting is enabled
3.  Consider implementing usage tracking in development to optimize token usage before production
4.  Use the cached token information to optimize your application’s performance

## Alternative: Getting Usage via Generation ID

You can also retrieve usage information asynchronously by using the generation ID returned from your API calls. This is particularly useful when you want to fetch usage statistics after the completion has finished or when you need to audit historical usage.

To use this method:

1.  Make your chat completion request as normal
2.  Note the `id` field in the response
3.  Use that ID to fetch usage information via the `/generation` endpoint

For more details on this approach, see the [Get a Generation](https://openrouter.ai/docs/api-reference/get-a-generation) documentation.

## Examples

### Basic Usage with Token Tracking

PythonTypeScript

`   1  from openai import OpenAI  2  3  client = OpenAI(  4      base_url="https://openrouter.ai/api/v1",  5      api_key="<OPENROUTER_API_KEY>",  6  )  7  8  response = client.chat.completions.create(  9      model="anthropic/claude-3-opus",  10      messages=[  11          {"role": "user", "content": "What is the capital of France?"}  12      ],  13      extra_body={  14          "usage": {  15              "include": True  16          }  17      }  18  )  19  20  print("Response:", response.choices[0].message.content)  21  print("Usage Stats:", getattr(response, "usage", None))           `

### Streaming with Usage Information

This example shows how to handle usage information in streaming mode:

PythonTypeScript

`   1  from openai import OpenAI  2  3  client = OpenAI(  4      base_url="https://openrouter.ai/api/v1",  5      api_key="<OPENROUTER_API_KEY>",  6  )  7  8  def chat_completion_with_usage(messages):  9      response = client.chat.completions.create(  10          model="anthropic/claude-3-opus",  11          messages=messages,  12          extra_body={  13              "usage": {  14                  "include": True  15              }  16          },  17          stream=True  18      )  19      return response  20  21  for chunk in chat_completion_with_usage([  22      {"role": "user", "content": "Write a haiku about Paris."}  23  ]):  24      if hasattr(chunk, 'usage'):  25          if hasattr(chunk.usage, 'total_tokens'):  26              print(f"\nUsage Statistics:")  27              print(f"Total Tokens: {chunk.usage.total_tokens}")  28              print(f"Prompt Tokens: {chunk.usage.prompt_tokens}")  29              print(f"Completion Tokens: {chunk.usage.completion_tokens}")  30              print(f"Cost: {chunk.usage.cost} credits")  31      elif chunk.choices[0].delta.content:  32          print(chunk.choices[0].delta.content, end="")           `


---
Source: [Usage Accounting | Track AI Model Usage with OpenRouter | OpenRouter | Documentation](https://openrouter.ai/docs/use-cases/usage-accounting)