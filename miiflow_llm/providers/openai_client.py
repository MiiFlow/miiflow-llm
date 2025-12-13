"""OpenAI provider implementation."""

import asyncio
import copy
import re
from typing import Any, AsyncIterator, Dict, List, Optional

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.client import ChatResponse, ModelClient
from ..core.exceptions import AuthenticationError, ModelError, ProviderError, RateLimitError
from ..core.exceptions import TimeoutError as MiiflowTimeoutError
from ..core.message import DocumentBlock, ImageBlock, Message, MessageRole, TextBlock
from ..core.metrics import TokenCount, UsageData
from ..core.schema_normalizer import SchemaMode, normalize_json_schema
from ..core.stream_normalizer import OpenAIStreamNormalizer
from ..core.streaming import StreamChunk
from ..models.openai import get_token_param_name, supports_temperature


def _sanitize_tool_name(name: str) -> str:
    """Sanitize tool name to match OpenAI's pattern: ^[a-zA-Z0-9_-]+$"""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized[:64]  # OpenAI has a 64 char limit for function names


class OpenAIClient(ModelClient):
    """OpenAI provider client."""

    # Class-level mapping shared across instances for tool name resolution
    # Maps sanitized names back to original names
    _tool_name_mapping: Dict[str, str] = {}

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.provider_name = "openai"

        # Stream normalizer for unified streaming handling
        # Note: Pass class-level mapping for tool name restoration
        self._stream_normalizer = OpenAIStreamNormalizer(OpenAIClient._tool_name_mapping)

    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to OpenAI format with name sanitization."""
        original_name = schema["name"]
        sanitized_name = _sanitize_tool_name(original_name)

        # Track mapping for restoring original names from tool call responses
        if sanitized_name != original_name:
            OpenAIClient._tool_name_mapping[sanitized_name] = original_name

        sanitized_schema = {**schema, "name": sanitized_name}
        return {"type": "function", "function": sanitized_schema}

    @staticmethod
    def convert_schema_to_openai_format(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to OpenAI format with name sanitization.

        Note: For proper name mapping support, use convert_schema_to_provider_format instead.
        """
        original_name = schema["name"]
        sanitized_name = _sanitize_tool_name(original_name)

        # Track mapping for restoring original names from tool call responses
        if sanitized_name != original_name:
            OpenAIClient._tool_name_mapping[sanitized_name] = original_name

        sanitized_schema = {**schema, "name": sanitized_name}
        return {"type": "function", "function": sanitized_schema}

    def convert_message_to_provider_format(self, message: Message) -> Dict[str, Any]:
        return OpenAIClient.convert_message_to_openai_format(message)

    @staticmethod
    def convert_message_to_openai_format(message: Message) -> Dict[str, Any]:
        """Convert universal Message to OpenAI format (static for reuse by compatible providers)."""
        openai_message = {"role": message.role.value}

        if isinstance(message.content, str):
            openai_message["content"] = message.content
        else:
            content_list = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    content_list.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": block.image_url, "detail": block.detail},
                        }
                    )
                elif isinstance(block, DocumentBlock):
                    try:
                        from ..utils.pdf_extractor import extract_pdf_text_simple

                        pdf_text = extract_pdf_text_simple(block.document_url)

                        filename_info = f" [{block.filename}]" if block.filename else ""
                        pdf_content = f"[PDF Document{filename_info}]\n\n{pdf_text}"

                        content_list.append({"type": "text", "text": pdf_content})
                    except Exception as e:
                        filename_info = f" {block.filename}" if block.filename else ""
                        error_content = f"[Error processing PDF{filename_info}: {str(e)}]"
                        content_list.append({"type": "text", "text": error_content})

            openai_message["content"] = content_list

        if message.name:
            openai_message["name"] = message.name
        if message.tool_call_id:
            openai_message["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            # Sanitize tool names in tool_calls for OpenAI compatibility
            sanitized_tool_calls = []
            for tc in message.tool_calls:
                sanitized_tc = copy.deepcopy(tc) if isinstance(tc, dict) else tc
                if isinstance(sanitized_tc, dict) and "function" in sanitized_tc:
                    original_name = sanitized_tc["function"].get("name", "")
                    sanitized_tc["function"]["name"] = _sanitize_tool_name(original_name)
                sanitized_tool_calls.append(sanitized_tc)
            openai_message["tool_calls"] = sanitized_tool_calls

        return openai_message

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True
    )
    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send chat completion request to OpenAI."""
        try:
            openai_messages = [self.convert_message_to_provider_format(msg) for msg in messages]

            request_params = {
                "model": self.model,
                "messages": openai_messages,
            }

            if supports_temperature(self.model):
                request_params["temperature"] = temperature

            if max_tokens is not None:
                request_params[get_token_param_name(self.model)] = max_tokens
            else:
                # Provide sensible default when not specified
                request_params[get_token_param_name(self.model)] = 16384
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            if json_schema:
                normalized_schema = normalize_json_schema(json_schema, SchemaMode.STRICT)
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": normalized_schema,
                    },
                }

            response = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params), timeout=self.timeout
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            response_message = Message(
                role=MessageRole.ASSISTANT, content=content, tool_calls=choice.message.tool_calls
            )

            usage = TokenCount(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=choice.finish_reason,
                metadata={"response_id": response.id},
            )

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            retry_after = getattr(e.response.headers, "retry-after", None)
            raise RateLimitError(
                str(e), self.provider_name, retry_after=retry_after, original_error=e
            )
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", self.provider_name, original_error=e)

    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        try:
            openai_messages = [self.convert_message_to_provider_format(msg) for msg in messages]

            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "stream": True,
            }

            if supports_temperature(self.model):
                request_params["temperature"] = temperature

            if max_tokens is not None:
                request_params[get_token_param_name(self.model)] = max_tokens
            else:
                # Provide sensible default when not specified
                request_params[get_token_param_name(self.model)] = 16384
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            if json_schema:
                normalized_schema = normalize_json_schema(json_schema, SchemaMode.STRICT)
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": normalized_schema,
                    },
                }

            stream = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params), timeout=self.timeout
            )

            # Reset stream state for new streaming session
            self._stream_normalizer.reset_state()

            async for chunk in stream:
                if not chunk.choices:
                    continue

                normalized_chunk = self._stream_normalizer.normalize_chunk(chunk)

                # Only yield if there's content or metadata
                if (
                    normalized_chunk.delta
                    or normalized_chunk.tool_calls
                    or normalized_chunk.finish_reason
                ):
                    yield normalized_chunk

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            retry_after = getattr(e.response.headers, "retry-after", None)
            raise RateLimitError(
                str(e), self.provider_name, retry_after=retry_after, original_error=e
            )
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Streaming request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(
                f"OpenAI streaming error: {str(e)}", self.provider_name, original_error=e
            )
