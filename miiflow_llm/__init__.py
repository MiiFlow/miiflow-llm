"""
MiiFlow LLM - A lightweight, unified interface for LLM providers.

This package provides a consistent API for calling multiple LLM providers
with support for streaming, tool calling, and structured output.
"""

from .core.client import LLMClient, ModelClient, ChatResponse, StreamChunk
from .core.message import Message, MessageRole, ContentBlock, TextBlock, ImageBlock
from .core.metrics import LLMMetrics, TokenCount, UsageData, MetricsCollector
from .core.exceptions import (
    MiiFlowLLMError,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ModelError,
    TimeoutError,
    ParsingError,
    ToolError,
)
# from .tools.registry import ToolRegistry, tool, async_tool
# from .tools.schema import JSONSchema, ValidationResult
# from .agents.simple import SimpleAgent
# from .agents.function_calling import FunctionCallingAgent
# from .agents.react import ReActAgent

__version__ = "0.1.0"
__author__ = "MiiFlow Team"

__all__ = [
    # Core Components
    "LLMClient",
    "ModelClient",
    "ChatResponse",
    "StreamChunk",
    "Message",
    "MessageRole", 
    "ContentBlock",
    "TextBlock",
    "ImageBlock",
    "LLMMetrics",
    "TokenCount",
    "UsageData",
    "MetricsCollector",
    
    # Exceptions
    "MiiFlowLLMError",
    "ProviderError", 
    "AuthenticationError",
    "RateLimitError",
    "ModelError",
    "TimeoutError",
    "ParsingError",
    "ToolError",
    
    # Tools (coming soon)
    # "ToolRegistry",
    # "tool", 
    # "async_tool",
    # "JSONSchema",
    # "ValidationResult",
    
    # Agents (coming soon)
    # "SimpleAgent",
    # "FunctionCallingAgent", 
    # "ReActAgent",
]