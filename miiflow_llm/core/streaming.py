"""Streaming normalization and structured output parsing."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union, AsyncGenerator
from datetime import datetime

from .message import Message
from .metrics import TokenCount
from .exceptions import ParsingError


@dataclass
class StreamContent:
    """Normalized streaming content from any provider."""
    
    content: str
    is_delta: bool = True
    is_complete: bool = False
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    usage: Optional[TokenCount] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EnhancedStreamChunk:
    """Enhanced stream chunk with structured parsing support."""
    
    content: str
    delta: str = ""
    is_complete: bool = False
    partial_parse: Optional[Dict[str, Any]] = None
    structured_output: Optional[Any] = None
    finish_reason: Optional[str] = None
    usage: Optional[TokenCount] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.delta and self.content:
            self.delta = self.content


class ProviderStreamNormalizer:
    """Normalizes streaming chunks from different providers to unified format."""
    
    def normalize_chunk(self, chunk: Any, provider: str) -> StreamContent:
        """Convert provider-specific chunk to unified StreamContent."""
        
        normalization_map = {
            "openai": self._normalize_openai,
            "anthropic": self._normalize_anthropic,
            "groq": self._normalize_groq,
            "together": self._normalize_together,
            "xai": self._normalize_xai,
            "gemini": self._normalize_gemini,
            "openrouter": self._normalize_openrouter,
            "mistral": self._normalize_mistral,
            "ollama": self._normalize_ollama,
        }
        
        normalizer = normalization_map.get(provider.lower(), self._normalize_generic)
        
        try:
            return normalizer(chunk)
        except Exception as e:
            # Fallback to generic normalization
            return self._normalize_generic(chunk)
    
    def _normalize_openai(self, chunk) -> StreamContent:
        """Handle OpenAI's streaming format (GPT-4, GPT-5, GPT-4o)."""
        content = ""
        finish_reason = None
        function_call = None
        tool_calls = None
        usage = None
        
        try:
            # Handle both GPT-4 and GPT-5 formats
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                
                # GPT-5 style: chunk.choices[0].delta.content
                if hasattr(choice, 'delta') and choice.delta:
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        content = choice.delta.content
                    if hasattr(choice.delta, 'function_call'):
                        function_call = choice.delta.function_call
                    if hasattr(choice.delta, 'tool_calls'):
                        tool_calls = choice.delta.tool_calls
                
                # Finish reason
                if hasattr(choice, 'finish_reason'):
                    finish_reason = choice.finish_reason
            
            # Usage information (usually in last chunk)
            if hasattr(chunk, 'usage') and chunk.usage:
                usage = TokenCount(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens
                )
                
        except AttributeError:
            # Fallback for unexpected format changes
            content = str(chunk) if chunk else ""
        
        return StreamContent(
            content=content,
            is_delta=True,
            is_complete=finish_reason is not None,
            function_call=function_call,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            metadata={"provider": "openai", "raw_chunk_type": type(chunk).__name__}
        )
    
    def _normalize_anthropic(self, chunk) -> StreamContent:
        """Handle Anthropic's streaming format."""
        content = ""
        finish_reason = None
        usage = None
        
        try:
            # Anthropic format variations
            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                content = chunk.delta.text
            elif hasattr(chunk, 'content_block_delta') and hasattr(chunk.content_block_delta, 'delta'):
                if hasattr(chunk.content_block_delta.delta, 'text'):
                    content = chunk.content_block_delta.delta.text
            elif hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                content = chunk.message.content[0].text if chunk.message.content else ""
            
            # Check for completion
            if hasattr(chunk, 'type'):
                if chunk.type == 'message_stop':
                    finish_reason = 'stop'
                elif chunk.type == 'content_block_stop':
                    finish_reason = 'stop'
            
            # Usage (typically in final message)
            if hasattr(chunk, 'usage'):
                usage = TokenCount(
                    prompt_tokens=getattr(chunk.usage, 'input_tokens', 0),
                    completion_tokens=getattr(chunk.usage, 'output_tokens', 0),
                    total_tokens=getattr(chunk.usage, 'input_tokens', 0) + getattr(chunk.usage, 'output_tokens', 0)
                )
                
        except AttributeError:
            content = str(chunk) if chunk else ""
        
        return StreamContent(
            content=content,
            is_delta=True,
            is_complete=finish_reason is not None,
            finish_reason=finish_reason,
            usage=usage,
            metadata={"provider": "anthropic", "raw_chunk_type": type(chunk).__name__}
        )
    
    def _normalize_groq(self, chunk) -> StreamContent:
        """Handle Groq's streaming format."""
        content = ""
        finish_reason = None
        usage = None
        
        try:
            # Groq follows OpenAI-compatible format
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    content = choice.delta.content or ""
                if hasattr(choice, 'finish_reason'):
                    finish_reason = choice.finish_reason
            
            if hasattr(chunk, 'usage') and chunk.usage:
                usage = TokenCount(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens
                )
                
        except AttributeError:
            # Fallback - Groq sometimes returns plain strings
            content = str(chunk) if chunk else ""
        
        return StreamContent(
            content=content,
            is_delta=True,
            is_complete=finish_reason is not None,
            finish_reason=finish_reason,
            usage=usage,
            metadata={"provider": "groq", "raw_chunk_type": type(chunk).__name__}
        )
    
    def _normalize_together(self, chunk) -> StreamContent:
        """Handle TogetherAI's streaming format."""
        return self._normalize_openai(chunk)  # TogetherAI uses OpenAI-compatible format
    
    def _normalize_openrouter(self, chunk) -> StreamContent:
        """Handle OpenRouter's streaming format."""
        return self._normalize_openai(chunk)  # OpenRouter uses OpenAI-compatible format
    
    def _normalize_mistral(self, chunk) -> StreamContent:
        """Handle Mistral's streaming format."""
        content = ""
        finish_reason = None
        usage = None
        
        try:
            # Mistral streaming format (similar to OpenAI)
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                
                if hasattr(choice, 'delta') and choice.delta:
                    if hasattr(choice.delta, 'content'):
                        content = choice.delta.content or ""
                
                if hasattr(choice, 'finish_reason'):
                    finish_reason = choice.finish_reason
            
            # Usage information
            if hasattr(chunk, 'usage') and chunk.usage:
                usage = TokenCount(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens
                )
                
        except AttributeError:
            content = str(chunk) if chunk else ""
        
        return StreamContent(
            content=content,
            is_delta=True,
            is_complete=finish_reason is not None,
            finish_reason=finish_reason,
            usage=usage,
            metadata={"provider": "mistral", "raw_chunk_type": type(chunk).__name__}
        )
    
    def _normalize_ollama(self, chunk) -> StreamContent:
        """Handle Ollama's streaming format."""
        content = ""
        finish_reason = None
        
        try:
            # Ollama streaming format (JSON with message field)
            if isinstance(chunk, dict):
                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                if chunk.get("done", False):
                    finish_reason = "stop"
            elif hasattr(chunk, 'message'):
                content = chunk.message.get("content", "")
                if hasattr(chunk, 'done') and chunk.done:
                    finish_reason = "stop"
            else:
                content = str(chunk) if chunk else ""
                
        except (AttributeError, TypeError):
            content = str(chunk) if chunk else ""
        
        return StreamContent(
            content=content,
            is_delta=True,
            is_complete=finish_reason is not None,
            finish_reason=finish_reason,
            usage=None,  # Ollama doesn't provide detailed usage in streams
            metadata={"provider": "ollama", "raw_chunk_type": type(chunk).__name__}
        )
    
    def _normalize_xai(self, chunk) -> StreamContent:
        """Handle XAI's streaming format."""
        return self._normalize_openai(chunk)  # XAI uses OpenAI-compatible format
    
    def _normalize_gemini(self, chunk) -> StreamContent:
        """Handle Google Gemini's streaming format."""
        content = ""
        finish_reason = None
        usage = None
        
        try:
            # Gemini streaming format
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                
                # Extract content from parts
                if hasattr(candidate, 'content') and candidate.content.parts:
                    content = candidate.content.parts[0].text
                
                # Check finish reason
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    finish_reason = candidate.finish_reason.name
            
            # Usage metadata (Gemini provides detailed token counts)
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                usage = TokenCount(
                    prompt_tokens=getattr(chunk.usage_metadata, 'prompt_token_count', 0) or 0,
                    completion_tokens=getattr(chunk.usage_metadata, 'candidates_token_count', 0) or 0,
                    total_tokens=getattr(chunk.usage_metadata, 'total_token_count', 0) or 0
                )
                
        except AttributeError:
            # Fallback for unexpected format
            if hasattr(chunk, 'text'):
                content = chunk.text
            else:
                content = str(chunk) if chunk else ""
        
        return StreamContent(
            content=content,
            is_delta=True,
            is_complete=finish_reason is not None,
            finish_reason=finish_reason,
            usage=usage,
            metadata={"provider": "gemini", "raw_chunk_type": type(chunk).__name__}
        )
    
    def _normalize_generic(self, chunk) -> StreamContent:
        """Generic fallback normalization."""
        # Try to extract content using common patterns
        content = ""
        finish_reason = None
        
        if hasattr(chunk, 'content'):
            content = str(chunk.content)
        elif hasattr(chunk, 'text'):
            content = str(chunk.text)
        elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
            content = str(chunk.delta.content or "")
        else:
            content = str(chunk) if chunk else ""
        
        # Check for completion
        if hasattr(chunk, 'finish_reason') and chunk.finish_reason:
            finish_reason = chunk.finish_reason
        
        return StreamContent(
            content=content,
            is_delta=True,
            is_complete=finish_reason is not None,
            finish_reason=finish_reason,
            metadata={"provider": "generic", "raw_chunk_type": type(chunk).__name__}
        )


class IncrementalParser:
    """Parses structured output incrementally from streaming content."""
    
    def __init__(self, schema: Optional[Type] = None):
        self.schema = schema
        self.buffer = ""
        self.partial_objects: List[Dict[str, Any]] = []
    
    def try_parse_partial(self, new_content: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse partial structured output from accumulated content."""
        self.buffer += new_content
        
        # Try to extract complete JSON objects
        complete_objects = self._extract_complete_json_objects(self.buffer)
        
        if complete_objects:
            # Return the most recent complete object
            latest_object = complete_objects[-1]
            self.partial_objects.extend(complete_objects)
            return latest_object
        
        # Try to parse incomplete JSON for preview
        return self._attempt_partial_json_parse(self.buffer)
    
    def finalize_parse(self, complete_text: str) -> Optional[Any]:
        """Final parsing attempt with fallback strategies."""
        if complete_text.strip():
            self.buffer = complete_text
        
        # Strategy 1: Try direct JSON parsing
        try:
            return json.loads(self.buffer.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from text
        json_match = self._extract_json_from_text(self.buffer)
        if json_match:
            try:
                return json.loads(json_match)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Use regex patterns for common structures
        return self._fallback_regex_parse(self.buffer)
    
    def _extract_complete_json_objects(self, text: str) -> List[Dict[str, Any]]:
        """Extract complete JSON objects from text buffer."""
        objects = []
        
        # Find potential JSON objects using braces
        brace_stack = []
        start_pos = None
        
        for i, char in enumerate(text):
            if char == '{':
                if not brace_stack:
                    start_pos = i
                brace_stack.append(char)
            elif char == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start_pos is not None:
                        # Found complete object
                        json_str = text[start_pos:i+1]
                        try:
                            obj = json.loads(json_str)
                            objects.append(obj)
                        except json.JSONDecodeError:
                            continue
        
        return objects
    
    def _attempt_partial_json_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse incomplete JSON for preview."""
        # Look for partial objects that might be parseable
        text = text.strip()
        
        # Simple heuristic: if we have opening brace and some content
        if text.startswith('{') and len(text) > 2:
            # Try to close the JSON and parse
            attempts = [
                text + '}',
                text + '"}',
                text + '"}}'
            ]
            
            for attempt in attempts:
                try:
                    return json.loads(attempt)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON content from mixed text."""
        # Pattern to find JSON objects in text
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if matches:
            # Return the longest match (most likely to be complete)
            return max(matches, key=len)
        
        return None
    
    def _fallback_regex_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Fallback parsing using regex patterns."""
        result = {}
        
        # Common patterns for key-value extraction
        patterns = [
            r'"(\w+)":\s*"([^"]+)"',  # "key": "value"
            r'"(\w+)":\s*(\d+(?:\.\d+)?)',  # "key": number
            r'"(\w+)":\s*(true|false|null)',  # "key": boolean/null
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for key, value in matches:
                # Convert value types
                if value.lower() == 'true':
                    result[key] = True
                elif value.lower() == 'false':
                    result[key] = False
                elif value.lower() == 'null':
                    result[key] = None
                elif value.replace('.', '').isdigit():
                    result[key] = float(value) if '.' in value else int(value)
                else:
                    result[key] = value
        
        return result if result else None


class UnifiedStreamingClient:
    """Unified streaming client with structured output support."""
    
    def __init__(self, client, normalizer: Optional[ProviderStreamNormalizer] = None):
        self.client = client
        self.normalizer = normalizer or ProviderStreamNormalizer()
    
    async def stream_with_schema(
        self,
        messages: List[Message],
        schema: Optional[Type] = None,
        **kwargs
    ) -> AsyncGenerator[EnhancedStreamChunk, None]:
        """Stream with incremental structured output parsing."""
        
        parser = IncrementalParser(schema) if schema else None
        buffer = ""
        
        try:
            async for raw_chunk in self.client.stream_chat(messages, **kwargs):
                # Step 1: Normalize the chunk
                normalized = self.normalizer.normalize_chunk(
                    raw_chunk, 
                    self.client.provider_name
                )
                
                # Step 2: Accumulate content
                buffer += normalized.content
                
                # Step 3: Try incremental parsing if schema provided
                partial_parse = None
                if parser and normalized.content:
                    partial_parse = parser.try_parse_partial(normalized.content)
                
                # Step 4: Yield enhanced chunk
                yield EnhancedStreamChunk(
                    content=buffer,
                    delta=normalized.content,
                    is_complete=normalized.is_complete,
                    partial_parse=partial_parse,
                    finish_reason=normalized.finish_reason,
                    usage=normalized.usage,
                    tool_calls=normalized.tool_calls,
                    metadata={
                        **normalized.metadata,
                        "buffer_length": len(buffer),
                        "has_partial_parse": partial_parse is not None
                    }
                )
                
                # Break if stream is complete
                if normalized.is_complete:
                    break
            
            # Final parsing attempt
            if parser and buffer:
                final_result = parser.finalize_parse(buffer)
                yield EnhancedStreamChunk(
                    content=buffer,
                    delta="",
                    is_complete=True,
                    structured_output=final_result,
                    metadata={
                        "final_parse": True,
                        "parse_success": final_result is not None
                    }
                )
                
        except Exception as e:
            # Yield error information
            yield EnhancedStreamChunk(
                content="",
                delta="",
                is_complete=True,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise