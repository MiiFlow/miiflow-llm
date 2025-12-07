"""Event dataclasses for ReAct and Plan & Execute systems."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict

from .enums import ReActEventType, PlanExecuteEventType


@dataclass
class ReActEvent:
    """Event emitted during ReAct execution for streaming."""

    event_type: ReActEventType
    step_number: int
    data: Dict[str, Any]

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for streaming."""
        return {
            "event_type": self.event_type.value,
            "step_number": self.step_number,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        import json

        return json.dumps(self.to_dict())


@dataclass
class PlanExecuteEvent:
    """Event emitted during Plan and Execute."""

    event_type: PlanExecuteEventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }
