"""ReAct (Reasoning + Acting) and Plan & Execute - Architecture

Usage - ReAct:
    from miiflow_llm.core.react import ReActOrchestrator, ReActFactory

    orchestrator = ReActFactory.create_orchestrator(agent, max_steps=10)
    result = await orchestrator.execute("Find today's top news", context)

Usage - Plan & Execute:
    from miiflow_llm.core.react import PlanAndExecuteOrchestrator

    orchestrator = PlanAndExecuteOrchestrator(tool_executor, event_bus, safety_manager)
    result = await orchestrator.execute("Create Q4 sales report", context)
"""

# New clean architecture - no legacy imports
from .orchestrator import ReActOrchestrator
from .plan_execute_orchestrator import PlanAndExecuteOrchestrator
from .factory import ReActFactory
from .events import EventBus, EventFactory

# Core data structures (still needed)
from .data import (
    ReActStep,
    ReActResult,
    ReActEvent,
    ReActEventType,
    # Plan & Execute structures
    SubTask,
    Plan,
    PlanExecuteResult,
    PlanExecuteEvent,
    PlanExecuteEventType,
)
from .parser import ReActParser, ReActParsingError
from .safety import StopCondition, StopReason, SafetyManager

__all__ = [
    # Main interfaces
    "ReActOrchestrator",
    "PlanAndExecuteOrchestrator",
    "ReActFactory",
    # Clean event system
    "EventBus",
    "EventFactory",
    # ReAct data structures
    "ReActStep",
    "ReActResult",
    "ReActEvent",
    "ReActEventType",
    # Plan & Execute data structures
    "SubTask",
    "Plan",
    "PlanExecuteResult",
    "PlanExecuteEvent",
    "PlanExecuteEventType",
    # Parser and safety
    "ReActParser",
    "ReActParsingError",
    "StopCondition",
    "StopReason",
    "SafetyManager",
]

__version__ = "0.3.0"  # Added Plan and Execute orchestrator
