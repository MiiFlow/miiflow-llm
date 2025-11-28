"""ReAct Agent example.

This example demonstrates the ReAct (Reasoning + Acting) pattern:
- Creating an agent with tools
- Using the @tool decorator
- Running autonomous reasoning loops
- Handling tool results
"""

import asyncio
from miiflow_llm import LLMClient, Agent, AgentType, tool


# Define tools using the @tool decorator
@tool("calculate", "Evaluate mathematical expressions safely")
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression.

    Args:
        expression: A mathematical expression like "2 + 2" or "sqrt(16)"
    """
    import math

    # Safe evaluation with limited scope
    allowed = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool("get_weather", "Get current weather for a location")
def get_weather(location: str) -> str:
    """Get weather information for a location.

    Args:
        location: City name or location
    """
    # Simulated weather data
    weather_data = {
        "new york": "Sunny, 72°F (22°C)",
        "london": "Cloudy, 59°F (15°C)",
        "tokyo": "Rainy, 68°F (20°C)",
        "paris": "Partly cloudy, 65°F (18°C)",
    }

    location_lower = location.lower()
    if location_lower in weather_data:
        return f"Weather in {location}: {weather_data[location_lower]}"
    return f"Weather data not available for {location}"


@tool("search", "Search for information on a topic")
def search(query: str, limit: int = 3) -> str:
    """Search for information.

    Args:
        query: Search query
        limit: Maximum number of results (default 3)
    """
    # Simulated search results
    results = [
        f"Result 1: Information about {query} from Wikipedia",
        f"Result 2: Recent news about {query}",
        f"Result 3: Academic paper on {query}",
    ]
    return "\n".join(results[:limit])


async def basic_agent_example():
    """Basic ReAct agent with tools."""
    # Create client
    client = LLMClient.create("openai", model="gpt-4o-mini")

    # Create agent with tools
    agent = Agent(
        client,
        agent_type=AgentType.REACT,
        max_iterations=5,
        system_prompt="You are a helpful assistant with access to tools.",
    )

    # Add tools
    agent.add_tool(calculate)
    agent.add_tool(get_weather)
    agent.add_tool(search)

    # Run agent
    print("Query: What is 25 * 4 + 100?")
    result = await agent.run("What is 25 * 4 + 100?")
    print(f"Answer: {result.data}\n")


async def multi_step_reasoning():
    """Agent performing multi-step reasoning."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    agent = Agent(
        client,
        agent_type=AgentType.REACT,
        max_iterations=10,
    )

    agent.add_tool(calculate)
    agent.add_tool(search)

    # Complex query requiring multiple steps
    query = """
    I need to calculate the area of a circle with radius 5,
    then multiply that by 3. What's the final result?
    """

    print(f"Query: {query.strip()}")
    result = await agent.run(query)
    print(f"Answer: {result.data}\n")


async def agent_with_conversation_history():
    """Agent that maintains conversation context."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    agent = Agent(
        client,
        agent_type=AgentType.REACT,
        max_iterations=5,
    )

    agent.add_tool(get_weather)

    # First query
    print("Query 1: What's the weather in Tokyo?")
    result1 = await agent.run("What's the weather in Tokyo?")
    print(f"Answer: {result1.data}\n")

    # Second query (with context from first)
    print("Query 2: How about in London?")
    result2 = await agent.run(
        "How about in London?",
        messages=result1.messages,  # Pass conversation history
    )
    print(f"Answer: {result2.data}\n")


async def agent_with_event_streaming():
    """Agent with real-time event streaming."""
    from miiflow_llm.core.react import ReActEventType

    client = LLMClient.create("openai", model="gpt-4o-mini")

    agent = Agent(
        client,
        agent_type=AgentType.REACT,
        max_iterations=5,
    )

    agent.add_tool(calculate)

    # Subscribe to events
    def on_event(event):
        if event.event_type == ReActEventType.THINKING_CHUNK:
            print(f"[Thinking] {event.data.get('delta', '')}", end="")
        elif event.event_type == ReActEventType.ACTION_PLANNED:
            print(f"\n[Action] Using tool: {event.data.get('action')}")
        elif event.event_type == ReActEventType.OBSERVATION:
            print(f"[Observation] {event.data.get('observation')}")

    # Note: Event bus subscription requires orchestrator-level access
    # This is a simplified example

    print("Query: Calculate sqrt(144) + 8")
    result = await agent.run("Calculate sqrt(144) + 8")
    print(f"\nFinal Answer: {result.data}")


if __name__ == "__main__":
    print("=== Basic Agent Example ===")
    asyncio.run(basic_agent_example())

    print("=== Multi-step Reasoning ===")
    asyncio.run(multi_step_reasoning())

    print("=== Agent with Conversation History ===")
    asyncio.run(agent_with_conversation_history())

    print("=== Agent with Event Streaming ===")
    asyncio.run(agent_with_event_streaming())
