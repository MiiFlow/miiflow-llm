"""Enumeration types for ReAct and Plan & Execute systems."""

from enum import Enum


class ReActEventType(Enum):
    """Types of events emitted during ReAct execution."""

    STEP_START = "step_start"
    THOUGHT = "thought"
    THINKING_CHUNK = "thinking_chunk"  # Streaming chunks during thinking
    ACTION_PLANNED = "action_planned"
    ACTION_EXECUTING = "action_executing"
    OBSERVATION = "observation"
    STEP_COMPLETE = "step_complete"
    FINAL_ANSWER = "final_answer"
    FINAL_ANSWER_CHUNK = "final_answer_chunk"  # Streaming chunks for final answer
    ERROR = "error"
    STOP_CONDITION = "stop_condition"


class StopReason(Enum):
    """Reasons why ReAct loop terminated."""

    ANSWER_COMPLETE = "answer_complete"
    MAX_STEPS = "max_steps"
    MAX_BUDGET = "max_budget"
    MAX_TIME = "max_time"
    REPEATED_ACTIONS = "repeated_actions"
    ERROR_THRESHOLD = "error_threshold"
    FORCED_STOP = "forced_stop"


class PlanExecuteEventType(Enum):
    """Types of events emitted during Plan and Execute."""

    PLANNING_START = "planning_start"
    PLANNING_THINKING_CHUNK = "planning_thinking_chunk"  # LLM reasoning during planning
    PLANNING_COMPLETE = "planning_complete"
    REPLANNING_START = "replanning_start"
    REPLANNING_THINKING_CHUNK = "replanning_thinking_chunk"  # Streaming during replanning
    REPLANNING_COMPLETE = "replanning_complete"

    SUBTASK_START = "subtask_start"
    SUBTASK_THINKING_CHUNK = "subtask_thinking_chunk"  # ReAct reasoning during subtask execution
    SUBTASK_PROGRESS = "subtask_progress"
    SUBTASK_COMPLETE = "subtask_complete"
    SUBTASK_FAILED = "subtask_failed"

    PLAN_PROGRESS = "plan_progress"  # Overall plan progress update
    SYNTHESIS_START = "synthesis_start"  # Starting final answer synthesis
    FINAL_ANSWER = "final_answer"
    FINAL_ANSWER_CHUNK = "final_answer_chunk"  # Streaming chunks for final answer
    ERROR = "error"
