"""xAI Grok provider implementation."""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.client import ModelClient, ChatResponse, StreamChunk
from ..core.message import Message, MessageRole
from ..core.metrics import TokenCount, UsageData
from ..core.exceptions import (
    AuthenticationError,
    RateLimitError,
    ModelError,
    TimeoutError as MiiFlowTimeoutError,
    ProviderError,
)


class XAIClient(ModelClient):
    """xAI Grok provider client (OpenAI-compatible API)."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        self.provider_name = "xai"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatResponse:
        """Send chat completion request to xAI Grok."""
        try:
            openai_messages = [msg.to_openai_format() for msg in messages]
            
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
            
            response = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params),
                timeout=self.timeout
            )
            
            choice = response.choices[0]
            content = choice.message.content or ""
            
            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=choice.message.tool_calls
            )
            
            usage = TokenCount(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
            
            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=choice.finish_reason,
                metadata={"response_id": response.id}
            )
            
        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            retry_after = getattr(e.response.headers, 'retry-after', None)
            raise RateLimitError(str(e), self.provider_name, retry_after=retry_after, original_error=e)
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiFlowTimeoutError("Request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(f"xAI Grok API error: {str(e)}", self.provider_name, original_error=e)
    
    async def stream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming chat completion request to xAI Grok."""
        try:
            openai_messages = [msg.to_openai_format() for msg in messages]
            
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
                "stream": True,
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
            
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params),
                timeout=self.timeout
            )
            
            accumulated_content = ""
            final_usage = None
            
            async for chunk in stream:
                if not chunk.choices:
                    continue
                    
                choice = chunk.choices[0]
                delta = choice.delta
                
                content_delta = ""
                if delta.content:
                    content_delta = delta.content
                    accumulated_content += content_delta
                
                usage = None
                if chunk.usage:
                    usage = TokenCount(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens
                    )
                    final_usage = usage
                
                tool_calls = None
                if delta.tool_calls:
                    tool_calls = delta.tool_calls
                
                yield StreamChunk(
                    content=accumulated_content,
                    delta=content_delta,
                    finish_reason=choice.finish_reason,
                    usage=usage,
                    tool_calls=tool_calls
                )
            
        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            retry_after = getattr(e.response.headers, 'retry-after', None)
            raise RateLimitError(str(e), self.provider_name, retry_after=retry_after, original_error=e)
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiFlowTimeoutError("Streaming request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(f"xAI Grok streaming error: {str(e)}", self.provider_name, original_error=e)
