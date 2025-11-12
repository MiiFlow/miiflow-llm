# miiflow-llm

Unified Python interface for multiple LLM providers.

```python
from miiflow_llm import LLMClient
from miiflow_llm.core import Message

client = LLMClient.create("openai", model="gpt-4o-mini")
response = client.chat([Message.user("Hello")])

# Switch providers without code changes
client = LLMClient.create("anthropic", model="claude-3-5-sonnet-20241022")
response = client.chat([Message.user("Hello")])
```

## Install

```bash
pip install miiflow-llm
```

## Quick Start

```python
from miiflow_llm import LLMClient, Agent
from miiflow_llm.core import Message, AgentType
from miiflow_llm.core.tools import tool

# Basic chat
client = LLMClient.create("anthropic", model="claude-3-5-sonnet-20241022")
response = client.chat([Message.user("What is Rust?")])
print(response.message.content)

# Agent with tools
@tool("calculate", "Do math")
def calculate(expression: str) -> str:
    return str(eval(expression))

agent = Agent(client=client, agent_type=AgentType.REACT)
agent.add_tool(calculate)

import asyncio
result = asyncio.run(agent.run("What is 25 * 4 + 100?", deps={}))
print(result.data)
```

## Supported Providers

OpenAI, Anthropic, Google Gemini, Groq, TogetherAI, OpenRouter, Mistral, XAI

## Features

- Unified streaming interface
- Multi-modal (text + images)
- ReAct agents with tool calling
- Async and sync methods
- Type hints

## Documentation

**Get Started:**
- [Quickstart](docs/quickstart.md) - 5 minute intro
- [Tool Tutorial](docs/tutorial-tools.md) - Build tools step-by-step
- [Agent Tutorial](docs/tutorial-agents.md) - Build ReAct agents

**Reference:**
- [API Reference](docs/api.md) - Complete API
- [Providers](docs/providers.md) - Provider configs
- [Observability](docs/observability.md) - Phoenix tracing

**Examples:**
- [examples/](examples/) - Runnable code samples

## Environment Variables

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```

## License

MIT
