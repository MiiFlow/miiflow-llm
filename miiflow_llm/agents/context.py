"""Agent context system for miiflow-web integration."""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass


class ContextType(Enum):
    """Types of context that miiflow-web agents can use."""
    USER = "user"              # User-specific context
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
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def get(self, key: str, default=None):
        """Get metadata value."""
        return self.metadata.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set metadata value."""
        self.metadata[key] = value