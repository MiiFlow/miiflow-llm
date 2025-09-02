"""Mistral client implementation."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    from mistralai.async_client import MistralAsyncClient
    from mistralai.models.chat_completion import ChatMessage, ChatCompletionResponse
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    # Define placeholder classes for type hints when import fails
    class ChatMessage:
        pass
    class ChatCompletionResponse:
        pass

from ..core.client import ModelClient
from ..core.message import Message, MessageRole, TextBlock, ImageBlock
from ..core.metrics import TokenCount
from ..core.exceptions import ProviderError, AuthenticationError, ModelError


class MistralClient(ModelClient):
    """Mistral client implementation."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs
    ):
        if not MISTRAL_AVAILABLE:
            raise ImportError(
                "mistralai is required for Mistral. Install with: pip install mistralai"
            )
        
        super().__init__(model, api_key, timeout, max_retries, **kwargs)
        
        if not api_key:
            raise AuthenticationError("Mistral API key is required", provider="mistral")
        
        # Initialize Mistral client
        self.client = MistralAsyncClient(api_key=api_key)
        
        self.provider_name = "mistral"
    
    def _convert_messages_to_mistral_format(self, messages: List[Message]) -> List[ChatMessage]:
        """Convert messages to Mistral format."""
        mistral_messages = []
        
        for message in messages:
            # Mistral uses different role names
            role = message.role.value
            if role == "assistant":
                role = "assistant"
            elif role == "user":
                role = "user"
            elif role == "system":
                role = "system"
            elif role == "tool":
                role = "tool"
            
            # Handle content
            if isinstance(message.content, str):
                content = message.content
            elif isinstance(message.content, list):
                # For multi-modal content, extract text (Mistral has limited vision support)
                content_parts = []
                for block in message.content:
                    if isinstance(block, TextBlock):
                        content_parts.append(block.text)
                    elif isinstance(block, ImageBlock):
                        content_parts.append(f"[Image: {block.image_url}]")
                content = " ".join(content_parts)
            else:
                content = str(message.content)
            
            mistral_message = ChatMessage(
                role=role,
                content=content,
                name=message.name,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id
            )
            
            mistral_messages.append(mistral_message)
        
        return mistral_messages
    
    async def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """Send chat completion request to Mistral."""
        try:
            mistral_messages = self._convert_messages_to_mistral_format(messages)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": mistral_messages,
                "temperature": temperature,
                **kwargs
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
            
            # Make API call
            response = await self.client.chat(**request_params)
            
            # Extract response content
            content = response.choices[0].message.content or ""
            
            # Extract token usage
            usage = TokenCount()
            if response.usage:
                usage = TokenCount(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            
            # Create response message
            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=response.choices[0].message.tool_calls
            )
            
            from ..core.client import ChatResponse
            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            raise ProviderError(f"Mistral API error: {e}", provider="mistral")
    
    async def stream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator:
        """Send streaming chat completion request to Mistral."""
        try:
            mistral_messages = self._convert_messages_to_mistral_format(messages)
            
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": mistral_messages,
                "temperature": temperature,
                "stream": True,
                **kwargs
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
            
            # Stream response
            response_stream = await self.client.chat_stream(**request_params)
            
            accumulated_content = ""
            
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta_content = chunk.choices[0].delta.content or ""
                    accumulated_content += delta_content
                    
                    # Extract usage if available (usually in last chunk)
                    usage = None
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage = TokenCount(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens
                        )
                    
                    from ..core.client import StreamChunk
                    yield StreamChunk(
                        content=accumulated_content,
                        delta=delta_content,
                        finish_reason=chunk.choices[0].finish_reason,
                        usage=usage,
                        tool_calls=chunk.choices[0].delta.tool_calls if hasattr(chunk.choices[0].delta, 'tool_calls') else None
                    )
            
        except Exception as e:
            raise ProviderError(f"Mistral streaming error: {e}", provider="mistral")


# Available Mistral models
MISTRAL_MODELS = {
    # Latest Mistral models
    "mistral-large-latest": "mistral-large-latest",
    "mistral-medium-latest": "mistral-medium-latest", 
    "mistral-small-latest": "mistral-small-latest",
    
    # Specific versions
    "mistral-large-2402": "mistral-large-2402",
    "mistral-medium-2312": "mistral-medium-2312",
    "mistral-small-2312": "mistral-small-2312",
    "mistral-tiny-2312": "mistral-tiny-2312",
    
    # Mixtral models
    "mixtral-8x7b-instruct-v0.1": "mixtral-8x7b-instruct-v0.1",
    "mixtral-8x22b-instruct-v0.1": "mixtral-8x22b-instruct-v0.1",
    
    # Open source models
    "mistral-7b-instruct-v0.1": "mistral-7b-instruct-v0.1",
    "mistral-7b-instruct-v0.2": "mistral-7b-instruct-v0.2",
    "mistral-7b-instruct-v0.3": "mistral-7b-instruct-v0.3",
    
    # Code models
    "codestral-latest": "codestral-latest",
    "codestral-2405": "codestral-2405",
    
    # Embeddings
    "mistral-embed": "mistral-embed",
}