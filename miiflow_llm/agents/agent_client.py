"""Clean interface for interacting with agents."""

import logging
from typing import Dict, Any, List, Optional, AsyncIterator, Union
from dataclasses import dataclass
from enum import Enum

from ..core import LLMClient, Message, MessageRole
from ..core.tools import FunctionTool, ToolRegistry
from ..core.agent import Agent, RunResult, AgentType, RunContext
from .context import AgentContext, ContextType

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for creating an agent."""
    
    provider: str  # "openai", "anthropic", etc.
    model: str     # "gpt-4", "claude-3", etc.
    context_type: ContextType
    tools: List[FunctionTool] = None
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_iterations: int = 10
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []


class AgentClient:
    """Clean interface for interacting with agents."""
    
    def __init__(self, config: AgentConfig, agent: Agent):
        self.config = config
        self.agent = agent
        self.tool_registry = agent.tool_registry
    
    async def run(
        self, 
        prompt: str, 
        context: AgentContext,
        **kwargs
    ) -> Dict[str, Any]:
        """Run agent with structured output."""
        agent_deps = self._build_agent_deps(context)
        message_history = self._build_message_history(context)
        
        result: RunResult = await self.agent.run(
            prompt,
            deps=agent_deps,
            message_history=message_history,
            thread_id=context.thread_id,
            **kwargs
        )
        
        tool_calls_count = sum(len(msg.tool_calls) for msg in result.messages if msg.tool_calls)
        tool_messages_count = sum(1 for msg in result.messages if msg.role == MessageRole.TOOL)
        
        logger.debug(f"Final metrics - Individual tool calls: {tool_calls_count}, Tool result messages: {tool_messages_count}")
        
        return {
            "response": result.data,
            "context_updated": True,
            "thread_id": context.thread_id,
            "reasoning_steps": len(result.all_messages),
            "tool_calls_made": tool_calls_count,
            "tool_results_received": tool_messages_count,
            "metadata": {
                "model": self.config.model,
                "provider": self.config.provider,
                "context_type": context.context_type.value
            }
        }
    
    async def stream(
        self,
        prompt: str,
        context: AgentContext,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream agent responses with structured chunks."""
        agent_deps = self._build_agent_deps(context)
        message_history = self._build_message_history(context)
        
        messages = message_history or []
        messages.append(Message(role=MessageRole.USER, content=prompt))
        
        formatted_tools = None
        if self.agent._tools or self.agent.tool_registry.tools:
            formatted_tools = self.agent.tool_registry.get_schemas(
                self.agent.client.client.provider_name
            )
        
        async for chunk in self.agent.client.astream_chat(
            messages, 
            tools=formatted_tools,
            **kwargs
        ):
            yield {
                "content": chunk.content,
                "delta": chunk.delta,
                "finish_reason": chunk.finish_reason,
                "context_type": context.context_type.value,
                "metadata": {
                    "model": self.config.model,
                    "provider": self.config.provider
                }
            }
    
    def add_tool(self, tool: FunctionTool) -> None:
        """Add a tool to this agent instance."""
        self.agent.tool_registry.register(tool)
        self.agent._tools.append(tool)
    
    def list_tools(self) -> List[str]:
        """List tools registered with this agent."""
        return self.agent.tool_registry.list_tools()
    
    def _build_agent_deps(self, context: AgentContext) -> Any:
        """Convert AgentContext to agent dependencies."""
        return context
    
    def _build_message_history(self, context: AgentContext) -> Optional[List[Message]]:
        """Build message history from context."""
        if context.get("message_history"):
            return [
                Message(
                    role=MessageRole(msg["role"]), 
                    content=msg["content"]
                )
                for msg in context.get("message_history", [])
            ]
        return None


def create_agent(config: AgentConfig) -> AgentClient:
    """Factory function to create agents."""
    if not config.provider:
        raise ValueError("Provider is required")
    if not config.model:
        raise ValueError("Model is required")
        
    try:
        llm_client = LLMClient.create(
            provider=config.provider,
            model=config.model
        )
    except Exception as e:
        raise ValueError(f"Failed to create LLM client: {e}")
    
    agent_type_mapping = {
        ContextType.USER: AgentType.CHAT,
        ContextType.EMAIL: AgentType.CHAT, 
        ContextType.DOCUMENT: AgentType.RAG,
        ContextType.WORKFLOW: AgentType.WORKFLOW,
        ContextType.RAG: AgentType.RAG
    }
    
    agent_type = agent_type_mapping.get(config.context_type, AgentType.CHAT)
    
    agent = Agent(
        llm_client,
        agent_type=agent_type,
        system_prompt=config.system_prompt,
        temperature=config.temperature,
        max_iterations=config.max_iterations
    )
    
    for tool in config.tools or []:
        agent.tool_registry.register(tool)
        agent._tools.append(tool)
    
    return AgentClient(config, agent)
