"""TogetherAI client implementation."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..core.client import ModelClient
from ..core.message import Message, MessageRole
from ..core.metrics import TokenCount
from ..core.exceptions import ProviderError, AuthenticationError, ModelError


class TogetherClient(ModelClient):
    """TogetherAI client implementation using OpenAI-compatible API."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai is required for TogetherAI. Install with: pip install openai"
            )
        
        super().__init__(model, api_key, timeout, max_retries, **kwargs)
        
        if not api_key:
            raise AuthenticationError("TogetherAI API key is required", provider="together")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            timeout=timeout,
            max_retries=max_retries
        )
        
        self.provider_name = "together"
    
    def _convert_messages_to_openai_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format."""
        openai_messages = []
        
        for message in messages:
            openai_message = message.to_openai_format()
            openai_messages.append(openai_message)
        
        return openai_messages
    
    async def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """Send chat completion request to TogetherAI."""
        try:
            openai_messages = self._convert_messages_to_openai_format(messages)
            
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
                "stream": False,
                **kwargs
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
            
            response = await self.client.chat.completions.create(**request_params)
            
            content = response.choices[0].message.content or ""
            
            usage = TokenCount()
            if response.usage:
                usage = TokenCount(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            
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
            raise ProviderError(f"TogetherAI API error: {e}", provider="together")
    
    async def stream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator:
        """Send streaming chat completion request to TogetherAI."""
        try:
            openai_messages = self._convert_messages_to_openai_format(messages)
            
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
                "stream": True,
                **kwargs
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
            
            response_stream = await self.client.chat.completions.create(**request_params)
            
            accumulated_content = ""
            
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta_content = chunk.choices[0].delta.content or ""
                    accumulated_content += delta_content
                    
                    usage = None
                    if chunk.usage:
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
            raise ProviderError(f"TogetherAI streaming error: {e}", provider="together")


# Popular TogetherAI models
TOGETHER_MODELS = {
    # Meta Llama models
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    
    # Qwen models  
    "Qwen/Qwen2.5-7B-Instruct-Turbo": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    
    # Mistral models on Together
    "mistralai/Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    
    # Other popular models
    "microsoft/DialoGPT-medium": "microsoft/DialoGPT-medium",
    "togethercomputer/RedPajama-INCITE-7B-Chat": "togethercomputer/RedPajama-INCITE-7B-Chat",
}
