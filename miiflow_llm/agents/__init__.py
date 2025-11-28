"""
Agents module - DEPRECATED.

The AgentClient, AgentConfig, create_agent, and AgentContext have been removed.
Use miiflow_llm.core.agent.Agent directly instead.

Migration example:
    # Old way (removed):
    # from miiflow_llm.agents import create_agent, AgentConfig, AgentContext
    # agent = create_agent(AgentConfig(provider="openai", model="gpt-4", agent_type=AgentType.REACT))
    # result = await agent.run("prompt", context=AgentContext())

    # New way:
    from miiflow_llm import LLMClient, Agent, AgentType, RunContext

    client = LLMClient.create("openai", model="gpt-4")
    agent = Agent(client, agent_type=AgentType.REACT)
    result = await agent.run("prompt", deps=None)
"""

# Re-export from core for backward compatibility during transition
from ..core.agent import Agent, AgentType, RunContext, RunResult

__all__ = ["Agent", "AgentType", "RunContext", "RunResult"]
