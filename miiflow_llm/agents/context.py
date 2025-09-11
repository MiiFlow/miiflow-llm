"""Minimal context system for agent execution."""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class ContextType(Enum):
    """Types of context for agent execution."""
    USER = "user"              # User conversation context
    EMAIL = "email"            # Email conversation context  
    DOCUMENT = "document"      # Document analysis context
    WORKFLOW = "workflow"      # Workflow execution context
    RAG = "rag"               # Retrieval-augmented generation context


@dataclass
class AgentContext:
    """Context container for agent execution."""
    
    context_type: ContextType
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default=None):
        """Get metadata value."""
        return self.metadata.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set metadata value."""
        self.metadata[key] = value
