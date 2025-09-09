"""Unified Agent architecture for miiflow-llm - Complete LlamaIndex replacement.

This module provides the core agent system that miiflow-web will use to replace
LlamaIndex functionality. It supports:
- Dependency injection for application-specific services
- Context management for conversation state
- Tool calling with proper typing
- Customizable agent behaviors for different use cases
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union, Protocol, AsyncIterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)

from .client import LLMClient
from .message import Message, MessageRole
from .tools import FunctionTool, ToolRegistry
from .exceptions import MiiflowLLMError, ErrorType

# Type variables for dependency injection and results
Deps = TypeVar('Deps')
Result = TypeVar('Result')


class AgentType(Enum):
    """Types of agents for different miiflow-web use cases."""
    CHAT = "chat"              # General conversation agent
    RAG = "rag"                # Retrieval-augmented generation
    SEARCH = "search"          # Search and discovery
    ANALYSIS = "analysis"      # Data analysis and insights
    WORKFLOW = "workflow"      # Multi-step task execution
    KNOWLEDGE = "knowledge"    # Knowledge base interaction


@dataclass
class RunResult(Generic[Result]):
    """Result from an agent run with metadata."""
    
    data: Result
    messages: List[Message]
    usage_cost: float = 0.0
    all_messages: List[Message] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.all_messages:
            self.all_messages = self.messages


# Protocols for miiflow-web to implement
class DatabaseService(Protocol):
    """Protocol for database services that miiflow-web provides."""
    async def query(self, sql: str) -> List[Dict[str, Any]]: ...
    async def get_user_context(self, user_id: str) -> Dict[str, Any]: ...


class VectorStoreService(Protocol):
    """Protocol for vector store services (replacing LlamaIndex VectorStore)."""
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]: ...
    async def add_documents(self, documents: List[Dict[str, Any]]) -> None: ...


class KnowledgeService(Protocol):
    """Protocol for knowledge base services."""
    async def get_relevant_docs(self, query: str) -> List[Dict[str, Any]]: ...
    async def get_thread_context(self, thread_id: str) -> Dict[str, Any]: ...
    async def save_thread_context(self, thread_id: str, context_data: Dict[str, Any]) -> None: ...


@dataclass 
class RunContext(Generic[Deps]):
    """Context passed to tools and agent functions with dependency injection.
    
    This replaces LlamaIndex's context management and provides miiflow-web
    with a clean dependency injection system.
    """
    
    deps: Deps
    messages: List[Message] = field(default_factory=list)
    retry: int = 0
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def last_user_message(self) -> Optional[Message]:
        """Get the last user message."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.USER:
                return msg
        return None
    
    def last_agent_message(self) -> Optional[Message]:
        """Get the last agent message."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.ASSISTANT:
                return msg
        return None
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for context."""
        if len(self.messages) <= 2:
            return "New conversation"
        
        user_messages = [msg.content for msg in self.messages if msg.role == MessageRole.USER]
        return f"Conversation with {len(user_messages)} user messages"
    
    def has_context(self, key: str) -> bool:
        """Check if context has a specific key."""
        return key in self.metadata


class Agent(Generic[Deps, Result]):
    """Unified Agent architecture - Complete LlamaIndex replacement for miiflow-web.
    
    This agent provides:
    - Dependency injection for miiflow-web services (database, vector store, etc.)
    - Context management for conversation state
    - Tool calling with proper typing
    - Customizable behaviors for different agent types
    """
    
    def __init__(
        self,
        client: LLMClient,
        *,
        agent_type: AgentType = AgentType.CHAT,
        deps_type: Optional[Type[Deps]] = None,
        result_type: Optional[Type[Result]] = None,
        system_prompt: Optional[Union[str, Callable[[RunContext[Deps]], str]]] = None,
        retries: int = 1,
        max_iterations: int = 10,
        temperature: float = 0.7
    ):
        self.client = client
        self.agent_type = agent_type
        self.deps_type = deps_type
        self.result_type = result_type or str
        self.system_prompt = system_prompt
        self.retries = retries
        self.max_iterations = max_iterations
        self.temperature = temperature
        
        # Share the tool registry with LLMClient for consistency
        self.tool_registry = self.client.tool_registry
        self._tools: List[FunctionTool] = []
        
        # Agent-specific configurations
        self._configure_for_type()
    
    def _configure_for_type(self):
        """Configure agent based on its type for miiflow-web use cases."""
        if self.agent_type == AgentType.RAG:
            self.max_iterations = 3
        elif self.agent_type == AgentType.WORKFLOW:
            self.max_iterations = 20
        elif self.agent_type == AgentType.ANALYSIS:
            self.temperature = 0.1
    
    def tool(
        self, 
        name_or_func=None, 
        description: Optional[str] = None, 
        *,
        name: Optional[str] = None
    ) -> Callable:
        """Decorator to register a tool with this agent.
        
        Supports multiple calling styles:
        
        @agent.tool  # Simple - uses function name
        async def search(ctx: RunContext, query: str) -> str:
            ...
        
        @agent.tool("custom_name")  # With custom name
        async def search_func(ctx: RunContext, query: str) -> str:
            ...
            
        @agent.tool("custom_name", "Custom description")  # With name and description
        async def search_func(ctx: RunContext, query: str) -> str:
            ...
            
        @agent.tool(name="custom_name", description="Custom description")  # With keywords
        async def search_func(ctx: RunContext, query: str) -> str:
            ...
        """
        
        def decorator(func: Callable) -> Callable:
            # Determine tool name - keyword takes precedence
            tool_name = name
            tool_desc = description
            
            if not tool_name:
                if isinstance(name_or_func, str):
                    tool_name = name_or_func
                elif name_or_func is not None and not callable(name_or_func):
                    tool_name = str(name_or_func)
            
            tool_instance = FunctionTool(func, tool_name, tool_desc)
            
            self.tool_registry.register(tool_instance)
            self._tools.append(tool_instance)
            
            logger.debug(f"Registered tool '{tool_instance.name}' with context pattern: {tool_instance.context_injection['pattern']}")
            
            return tool_instance
        
        # Handle direct decoration (@agent.tool)
        if callable(name_or_func):
            return decorator(name_or_func)
        else:
            return decorator
    
    async def run(
        self, 
        user_prompt: str, 
        *,
        deps: Optional[Deps] = None,
        message_history: Optional[List[Message]] = None,
        thread_id: Optional[str] = None
    ) -> RunResult[Result]:
        """Run the agent with dependency injection."""
        
        context = RunContext(
            deps=deps,
            messages=message_history or [],
            thread_id=thread_id
        )
        
        if thread_id and hasattr(deps, 'knowledge'):
            try:
                thread_context = await deps.knowledge.get_thread_context(thread_id)
                if thread_context and 'messages' in thread_context:
                    previous_messages = [
                        Message(role=MessageRole(msg['role']), content=msg['content'])
                        for msg in thread_context['messages']
                    ]
                    context.messages.extend(previous_messages)
                    context.metadata['restored_messages'] = len(previous_messages)
            except Exception as e:
                logger.warning(f"Could not load thread context: {e}")
        
        system_msg = None
        if self.system_prompt:
            if callable(self.system_prompt):
                system_content = self.system_prompt(context)
            else:
                system_content = self.system_prompt
            
            system_msg = Message(role=MessageRole.SYSTEM, content=system_content)
            context.messages.append(system_msg)
        
        user_msg = Message(role=MessageRole.USER, content=user_prompt)
        context.messages.append(user_msg)
        
        for attempt in range(self.retries):
            context.retry = attempt
            try:
                result = await self._execute_with_context(context)
                
                return RunResult(
                    data=result,
                    messages=context.messages,
                    all_messages=context.messages.copy()
                )
                
            except Exception as e:
                if attempt == self.retries - 1:
                    raise MiiflowLLMError(f"Agent failed after {self.retries} retries: {e}", ErrorType.MODEL_ERROR)
                continue
        
        raise MiiflowLLMError("Agent execution failed", ErrorType.MODEL_ERROR)
    
    async def _execute_with_context(self, context: RunContext[Deps]) -> Result:
        """Execute agent with multi-hop reasoning and persistent memory."""
        reasoning_steps = []
        
        for iteration in range(self.max_iterations):
            reasoning_step = {
                "step": iteration + 1,
                "timestamp": time.time(),
                "context_summary": context.get_conversation_summary(),
                "tools_available": len(self._tools),
                "reasoning": None,
                "action": None,
                "result": None
            }
            
            enhanced_messages = context.messages.copy()
            
            if iteration > 0:
                reasoning_summary = self._build_reasoning_summary(reasoning_steps)
                enhanced_messages.append(Message(
                    role=MessageRole.SYSTEM,
                    content=f"Previous reasoning steps: {reasoning_summary}"
                ))
            
            response = await self.client.achat(
                messages=enhanced_messages,
                tools=None,
                temperature=self.temperature
            )
            
            reasoning_step["reasoning"] = f"LLM response in step {iteration + 1}"
            context.messages.append(response.message)
            
            if response.message.tool_calls:
                reasoning_step["action"] = "tool_execution" 
                reasoning_step["tools_called"] = [
                    tc.function.name if hasattr(tc, 'function') else tc.get("function", {}).get("name") 
                    for tc in response.message.tool_calls
                ]
                
                await self._execute_tool_calls(response.message.tool_calls, context)
                reasoning_step["result"] = "tools_executed"
                reasoning_steps.append(reasoning_step)
                
                if iteration > 2:
                    recent_steps = reasoning_steps[-3:]
                    if all(step.get("action") == "tool_execution" for step in recent_steps):
                        logger.debug(f"Detected potential tool loop at iteration {iteration}")
                        context.messages.append(Message(
                            role=MessageRole.SYSTEM,
                            content="You have executed several tools successfully. Please provide your final answer based on the tool results. Do not call any more tools."
                        ))
                
                if iteration >= self.max_iterations - 2:
                    logger.debug(f"Near max iterations ({iteration}/{self.max_iterations}), forcing termination")
                    context.messages.append(Message(
                        role=MessageRole.SYSTEM,
                        content="This is your final opportunity to respond. Provide your best answer based on available information. Do not call any tools."
                    ))
                
                continue
            
            reasoning_step["action"] = "final_response"
            reasoning_step["result"] = response.message.content
            reasoning_steps.append(reasoning_step)
            
            if context.thread_id and hasattr(context.deps, 'knowledge'):
                await self._save_conversation_context(context, reasoning_steps)
            
            if self.result_type == str:
                return response.message.content
            else:
                return response.message.content
        
        raise MiiflowLLMError(f"Agent exceeded maximum reasoning steps ({self.max_iterations})", ErrorType.MODEL_ERROR)
    
    def _build_reasoning_summary(self, reasoning_steps: List[Dict]) -> str:
        """Build a summary of previous reasoning steps."""
        if not reasoning_steps:
            return "No previous steps"
        
        summary_parts = []
        for step in reasoning_steps[-3:]:
            step_summary = f"Step {step['step']}: {step['action']}"
            if step.get('tools_called'):
                step_summary += f" (used tools: {', '.join(step['tools_called'])})"
            summary_parts.append(step_summary)
        
        return " → ".join(summary_parts)
    
    async def _save_conversation_context(self, context: RunContext[Deps], reasoning_steps: List[Dict]):
        """Save conversation context for persistent memory."""
        try:
            conversation_data = {
                "messages": [
                    {"role": msg.role.value, "content": msg.content} 
                    for msg in context.messages
                ],
                "reasoning_chain": reasoning_steps,
                "metadata": context.metadata,
                "timestamp": time.time()
            }
            
            if hasattr(context.deps.knowledge, 'save_thread_context'):
                await context.deps.knowledge.save_thread_context(context.thread_id, conversation_data)
        
        except Exception as e:
            logger.warning(f"Could not save conversation context: {e}")
    
    async def _execute_tool_calls(
        self, 
        tool_calls: List[Dict[str, Any]], 
        context: RunContext[Deps]
    ) -> None:
        """Execute tool calls with dependency injection."""
        logger.debug(f"About to execute {len(tool_calls)} tool calls")
        
        for i, tool_call in enumerate(tool_calls):
            logger.debug(f"Executing tool call {i+1}/{len(tool_calls)}")
            
            if hasattr(tool_call, 'function'):
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                if isinstance(tool_args, str):
                    import json
                    tool_args = json.loads(tool_args)
            else:
                tool_name = tool_call.get("function", {}).get("name")
                tool_args = tool_call.get("function", {}).get("arguments", {})
                if isinstance(tool_args, str):
                    import json
                    tool_args = json.loads(tool_args)
            
            if tool_args is None:
                tool_args = {}
            elif not isinstance(tool_args, dict):
                logger.warning(f"Invalid tool_args type: {type(tool_args)}, converting to empty dict")
                tool_args = {}
            
            logger.debug(f"Tool '{tool_name}' with args: {tool_args}")
            
            tool = self.tool_registry.tools.get(tool_name)
            if tool and hasattr(tool, 'context_injection'):
                injection_pattern = tool.context_injection
                
                if injection_pattern['pattern'] == 'first_param':
                    logger.debug(f"Using Pydantic AI context injection for {tool_name}")
                    observation = await self.tool_registry.execute_safe_with_context(
                        tool_name, context, **tool_args
                    )
                else:
                    logger.debug(f"Plain function execution for {tool_name}")
                    observation = await self.tool_registry.execute_safe(tool_name, **tool_args)
            else:
                logger.debug(f"Plain function execution (no pattern detection) for {tool_name}")
                observation = await self.tool_registry.execute_safe(tool_name, **tool_args)
            
            logger.debug(f"Tool '{tool_name}' execution result: success={observation.success}, output='{observation.output}'")
            
            context.messages.append(Message(
                role=MessageRole.TOOL,
                content=str(observation.output) if observation.success else observation.error,
                tool_call_id=tool_call.id if hasattr(tool_call, 'id') else tool_call.get("id")
            ))
    
    def _tool_supports_context(self, tool_name: str) -> bool:
        """Check if a tool supports context injection."""
        if tool_name in self.tool_registry.tools:
            tool = self.tool_registry.tools[tool_name]
            if hasattr(tool, 'fn'):
                import inspect
                sig = inspect.signature(tool.fn)
                return 'context' in sig.parameters
        return False
    
    def _create_react_loop(
        self,
        max_steps: int = 10,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None,
        safety_profile: str = "balanced"
    ):
        """Create a configured ReAct loop."""
        from .react import ReActLoop, SafetyProfiles
        
        # Select safety profile
        if safety_profile == "conservative":
            safety_manager = SafetyProfiles.conservative()
        elif safety_profile == "balanced":
            safety_manager = SafetyProfiles.balanced()
        elif safety_profile == "permissive":
            safety_manager = SafetyProfiles.permissive()
        else:
            # Custom safety manager with provided parameters
            from .react.safety import SafetyManager
            safety_manager = SafetyManager(
                max_steps=max_steps,
                max_budget=max_budget,
                max_time_seconds=max_time_seconds
            )
        
        return ReActLoop(
            agent=self,
            safety_manager=safety_manager
        )
    
    async def run_react(
        self,
        query: str,
        context: RunContext,
        max_steps: int = 10,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None,
        safety_profile: str = "balanced"
    ):
        """Run agent in ReAct mode with structured reasoning.
        
        Args:
            query: User query to process
            context: Run context with dependencies
            max_steps: Maximum reasoning steps (default: 10)
            max_budget: Maximum cost in dollars (optional)
            max_time_seconds: Maximum execution time (optional)
            safety_profile: Safety profile ("conservative", "balanced", "permissive")
            
        Returns:
            ReActResult with complete reasoning chain
        """
        react_loop = self._create_react_loop(max_steps, max_budget, max_time_seconds, safety_profile)
        return await react_loop.execute(query, context, stream_events=False)
    
    async def stream_react(
        self,
        query: str,
        context: RunContext,
        max_steps: int = 10,
        max_budget: Optional[float] = None,
        max_time_seconds: Optional[float] = None,
        safety_profile: str = "balanced"
    ):
        """Run agent in ReAct mode with streaming events.
        
        Args:
            query: User query to process
            context: Run context with dependencies
            max_steps: Maximum reasoning steps (default: 10)
            max_budget: Maximum cost in dollars (optional)
            max_time_seconds: Maximum execution time (optional)
            safety_profile: Safety profile ("conservative", "balanced", "permissive")
            
        Yields:
            ReActEvent objects for each reasoning step
        """
        react_loop = self._create_react_loop(max_steps, max_budget, max_time_seconds, safety_profile)
        async for event in react_loop.stream_execute(query, context):
            yield event


class ExampleDeps:
    """Example dependency container - applications should create their own."""
    pass
