"""Message handling and format conversion for different providers."""

import mimetypes
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union


class MessageRole(Enum):
    """Standard message roles across all providers."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentType(Enum):
    """Content block types for multi-modal messages."""
    
    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_FILE = "image_file"
    DOCUMENT = "document"


@dataclass
class TextBlock:
    """Text content block."""
    
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImageBlock:
    """Image content block with URL or base64 data."""
    
    type: Literal["image_url"] = "image_url"
    image_url: str = ""
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


@dataclass
class DocumentBlock:
    """Document content block for PDFs and other documents."""
    
    type: Literal["document"] = "document"
    document_url: str = ""
    document_type: str = "pdf"
    filename: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_url(cls, document_url: str, filename: Optional[str] = None, **kwargs) -> "DocumentBlock":
        """Create DocumentBlock with auto-detected document type from URL."""
        document_type = cls._detect_type_from_url(document_url)
        return cls(
            document_url=document_url,
            document_type=document_type,
            filename=filename,
            **kwargs
        )
    
    @staticmethod
    def _detect_type_from_url(url: str) -> str:
        """Detect document type from URL."""
        if url.startswith('data:'):
            return 'pdf' if 'application/pdf' in url else 'pdf'
        
        extension = Path(urllib.parse.urlparse(url).path).suffix.lower()
        types = {'.pdf': 'pdf', '.doc': 'doc', '.docx': 'docx', '.txt': 'txt', '.csv': 'csv'}
        return types.get(extension, 'pdf')


# Union type for content blocks
ContentBlock = Union[TextBlock, ImageBlock, DocumentBlock]


@dataclass
class Message:
    """Unified message format across all LLM providers."""
    
    role: MessageRole
    content: Union[str, List[ContentBlock]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def user(cls, content: Union[str, List[ContentBlock]], **kwargs) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, **kwargs)
    
    @classmethod
    def assistant(cls, content: str, **kwargs) -> "Message":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, **kwargs)
    
    @classmethod
    def system(cls, content: str, **kwargs) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)
    
    @classmethod
    def tool(cls, content: str, tool_call_id: str, **kwargs) -> "Message":
        """Create a tool response message."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            **kwargs
        )
    
    @classmethod
    def from_pdf(cls, text: str, pdf_url: str, filename: Optional[str] = None, **kwargs) -> "Message":
        """Create a user message with PDF attachment."""
        content = [
            TextBlock(text=text),
            DocumentBlock(
                document_url=pdf_url,
                document_type="pdf",
                filename=filename
            )
        ]
        return cls(role=MessageRole.USER, content=content, **kwargs)
    
    @classmethod
    def from_image(cls, text: str, image_url: str, detail: Optional[Literal["auto", "low", "high"]] = "auto", **kwargs) -> "Message":
        """Create a user message with image attachment."""
        content = [
            TextBlock(text=text),
            ImageBlock(
                image_url=image_url,
                detail=detail
            )
        ]
        return cls(role=MessageRole.USER, content=content, **kwargs)
    
    @classmethod
    def from_attatchments(cls, text: str, attachments: List[Union[str, Dict[str, Any]]], **kwargs) -> "Message":
        """Create a user message with multiple attachments."""
        content = [TextBlock(text=text)]
        
        for attachment in attachments:
            if isinstance(attachment, str):
                content.append(ImageBlock(image_url=attachment))
            elif isinstance(attachment, dict):
                attachment_type = attachment.get("type", "image")
                if attachment_type == "pdf":
                    content.append(DocumentBlock.from_url(
                        document_url=attachment["url"],
                        filename=attachment.get("filename")
                    ))
                elif attachment_type == "image":
                    content.append(ImageBlock(
                        image_url=attachment["url"],
                        detail=attachment.get("detail", "auto")
                    ))
        
        return cls(role=MessageRole.USER, content=content, **kwargs)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI message format."""
        message = {"role": self.role.value}
        
        if isinstance(self.content, str):
            message["content"] = self.content
        else:
            content_list = []
            for block in self.content:
                if isinstance(block, TextBlock):
                    content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": block.image_url,
                            "detail": block.detail
                        }
                    })
                elif isinstance(block, DocumentBlock):
                    try:
                        from miiflow_llm.utils.pdf_extractor import extract_pdf_text_simple
                        pdf_text = extract_pdf_text_simple(block.document_url)
                        
                        filename_info = f" [{block.filename}]" if block.filename else ""
                        pdf_content = f"[PDF Document{filename_info}]\n\n{pdf_text}"
                        
                        content_list.append({"type": "text", "text": pdf_content})
                    except Exception as e:
                        filename_info = f" {block.filename}" if block.filename else ""
                        error_content = f"[Error processing PDF{filename_info}: {str(e)}]"
                        content_list.append({"type": "text", "text": error_content})
            
            message["content"] = content_list
        
        if self.name:
            message["name"] = self.name
        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
            
        return message
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic message format."""
        message = {"role": self.role.value}
        
        if isinstance(self.content, str):
            message["content"] = self.content
        else:
            content_list = []
            for block in self.content:
                if isinstance(block, TextBlock):
                    content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    if block.image_url.startswith("data:"):
                        # Extract base64 content and media type using universal helper
                        base64_content, media_type = self._extract_base64_from_data_uri(block.image_url)
                        content_list.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_content
                            }
                        })
                    else:
                        content_list.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "media_type": "image/jpeg",  # Default for URLs
                                "data": block.image_url
                            }
                        })
                elif isinstance(block, DocumentBlock):
                    # Anthropic supports documents natively (PDFs, etc.)
                    if block.document_url.startswith("data:"):
                        # Extract base64 content and media type using universal helper
                        base64_content, media_type = self._extract_base64_from_data_uri(block.document_url)
                        content_list.append({
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_content
                            }
                        })
                    else:
                        content_list.append({
                            "type": "document", 
                            "source": {
                                "type": "url",
                                "media_type": f"application/{block.document_type}",
                                "data": block.document_url
                            }
                        })
            message["content"] = content_list
            
        return message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "role": self.role.value,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "tool_calls": self.tool_calls,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @staticmethod
    def _extract_base64_from_data_uri(data_uri: str) -> tuple[str, str]:
        """
        Universal base64 extractor for all multimedia content types.
        """
        if not data_uri.startswith("data:"):
            return data_uri, "application/octet-stream"
        
        try:
            if "," not in data_uri:
                return data_uri, "application/octet-stream"
                
            header, base64_content = data_uri.split(",", 1)
            
            # Extract media type from header: data:media_type;base64
            media_type = "application/octet-stream"  # default fallback
            if ":" in header and ";" in header:
                media_type = header.split(":")[1].split(";")[0]
            
            return base64_content, media_type
            
        except Exception:
            return data_uri, "application/octet-stream"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=data.get("tool_calls"),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            metadata=data.get("metadata", {}),
        )
