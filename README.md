# miiflow-llm

A lightweight, unified Python interface for LLM providers with built-in agentic patterns.

```python
from miiflow_llm import LLMClient, Message

# Unified interface across providers
client = LLMClient.create("openai", model="gpt-4o-mini")
response = client.chat([Message.user("Hello")])

# Switch providers with one line
client = LLMClient.create("anthropic", model="claude-3-5-sonnet-20241022")
```

## Features

- **Unified API** - Same interface for OpenAI, Anthropic, Google, Groq, and more
- **Agentic Patterns** - Built-in ReAct and Plan & Execute orchestrators
- **Tool Calling** - Simple `@tool` decorator with automatic schema generation
- **Streaming** - Real-time token streaming with event callbacks
- **Type Safety** - Full type hints and Pydantic integration
- **Lightweight** - Minimal dependencies, fast startup

## Installation

```bash
pip install miiflow-llm
```

## Quick Start

### Basic Chat

```python
from miiflow_llm import LLMClient, Message

client = LLMClient.create("openai", model="gpt-4o-mini")
response = client.chat([Message.user("What is Python?")])
print(response.message.content)
```

### Streaming

```python
async for chunk in client.astream_chat([Message.user("Tell me a story")]):
    print(chunk.delta, end="", flush=True)
```

### ReAct Agent with Tools

```python
from miiflow_llm import LLMClient, Agent, AgentType, tool
import asyncio

# Create client
client = LLMClient.create("openai", model="gpt-4o-mini")

# Define tools with the @tool decorator
@tool("calculate", "Evaluate mathematical expressions")
def calculate(expression: str) -> str:
    """Calculate the result of a math expression."""
    return str(eval(expression))

@tool("search", "Search for information")
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for '{query}': Python is a programming language..."

# Create agent with tools
agent = Agent(client, agent_type=AgentType.REACT, max_iterations=5)
agent.add_tool(calculate)
agent.add_tool(search)

# Run agent
result = asyncio.run(agent.run("What is 25 * 4 + 100?"))
print(result.data)  # "The answer is 200."
```

### Context Injection (Pydantic AI Style)

```python
from dataclasses import dataclass
from miiflow_llm import Agent, RunContext, tool

@dataclass
class UserContext:
    user_id: str
    role: str

@tool("get_profile")
def get_profile(ctx: RunContext[UserContext]) -> str:
    """Get current user's profile."""
    return f"User {ctx.deps.user_id} has role {ctx.deps.role}"

agent = Agent(client, deps_type=UserContext)
agent.add_tool(get_profile)

result = await agent.run(
    "Who am I?",
    deps=UserContext(user_id="alice", role="admin")
)
```

## Supported Providers

| Provider | Models | Streaming | Tool Calling |
|----------|--------|-----------|--------------|
| OpenAI | GPT-4o, GPT-4, GPT-3.5 | ✅ | ✅ |
| Anthropic | Claude 3.5, Claude 3 | ✅ | ✅ |
| Google | Gemini 1.5, Gemini 2 | ✅ | ✅ |
| Groq | Llama 3, Mixtral | ✅ | ✅ |
| Mistral | Mistral Large, Medium | ✅ | ✅ |
| OpenRouter | Multiple | ✅ | ✅ |
| Ollama | Local models | ✅ | ✅ |
| Amazon Bedrock | Claude, Llama | ✅ | ✅ |

## Agentic Patterns

### ReAct (Reasoning + Acting)

The ReAct pattern interleaves thinking and action:

```python
from miiflow_llm import Agent, AgentType

agent = Agent(
    client,
    agent_type=AgentType.REACT,
    max_iterations=10,
    system_prompt="You are a helpful research assistant."
)
```

### Plan & Execute

For complex multi-step tasks:

```python
from miiflow_llm.core.react import PlanAndExecuteOrchestrator, ReActFactory

# Create orchestrator with planning capabilities
orchestrator = ReActFactory.create_plan_execute_orchestrator(
    agent=agent,
    max_replans=2
)

result = await orchestrator.execute(
    "Research Python web frameworks and create a comparison table",
    context
)
```

## Event Streaming

Subscribe to real-time events during agent execution:

```python
from miiflow_llm.core.react import EventBus, ReActEventType

def on_event(event):
    if event.event_type == ReActEventType.THINKING_CHUNK:
        print(f"Thinking: {event.data['delta']}", end="")
    elif event.event_type == ReActEventType.OBSERVATION:
        print(f"\nTool result: {event.data['observation']}")

agent.event_bus.subscribe(on_event)
```

## Observability

Optional Phoenix tracing for debugging:

```python
from miiflow_llm.core import setup_tracing

# Enable tracing (requires PHOENIX_ENDPOINT env var)
setup_tracing()

# Or specify endpoint
setup_tracing(phoenix_endpoint="http://localhost:6006")
```

## Environment Variables

```bash
# Provider API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...

# Optional: Observability
PHOENIX_ENDPOINT=http://localhost:6006
PHOENIX_ENABLED=true
```

## Documentation

- [Quickstart Guide](docs/quickstart.md) - Get started in 5 minutes
- [Tool Tutorial](docs/tutorial-tools.md) - Build custom tools
- [Agent Tutorial](docs/tutorial-agents.md) - Build ReAct agents
- [API Reference](docs/api.md) - Complete API documentation
- [Provider Guide](docs/providers.md) - Provider-specific configuration
- [Observability](docs/observability.md) - Tracing and debugging

## Examples

See the [examples/](examples/) directory for runnable code samples:

- `basic_chat.py` - Simple chat completion
- `streaming.py` - Real-time streaming
- `react_agent.py` - ReAct agent with tools
- `plan_execute.py` - Plan & Execute for complex tasks
- `context_injection.py` - Pydantic AI style context

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT
