"""Plan and Execute orchestrator for complex multi-step tasks."""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..agent import RunContext
from ..message import Message, MessageRole
from .data import (
    PLAN_AND_EXECUTE_PLANNING_PROMPT,
    PLAN_AND_EXECUTE_REPLAN_PROMPT,
    Plan,
    PlanExecuteEvent,
    PlanExecuteEventType,
    PlanExecuteResult,
    StopReason,
    SubTask,
)
from .events import EventBus
from .orchestrator import ReActOrchestrator
from .safety import SafetyManager
from .tool_executor import AgentToolExecutor

logger = logging.getLogger(__name__)


class PlanAndExecuteOrchestrator:
    """Plan and Execute orchestrator with composable ReAct execution.

    This orchestrator breaks down complex tasks into structured plans with subtasks,
    then executes each subtask (optionally using ReAct for complex subtasks).

    Workflow:
    1. Planning Phase: Generate structured plan with subtasks
    2. Execution Phase: Execute subtasks in dependency order
    3. Re-planning Phase: Adapt plan if subtasks fail
    4. Synthesis Phase: Combine results into final answer
    """

    def __init__(
        self,
        tool_executor: AgentToolExecutor,
        event_bus: EventBus,
        safety_manager: SafetyManager,
        subtask_orchestrator: Optional[ReActOrchestrator] = None,
        max_replans: int = 2,
        use_react_for_subtasks: bool = True,
    ):
        """Initialize Plan and Execute orchestrator.

        Args:
            tool_executor: Tool execution adapter
            event_bus: Event bus for streaming events
            safety_manager: Safety condition checker
            subtask_orchestrator: Optional ReAct orchestrator for complex subtasks
            max_replans: Maximum number of re-planning attempts
            use_react_for_subtasks: Whether to use ReAct for subtask execution
        """
        self.tool_executor = tool_executor
        self.event_bus = event_bus
        self.safety_manager = safety_manager
        self.subtask_orchestrator = subtask_orchestrator
        self.max_replans = max_replans
        self.use_react_for_subtasks = use_react_for_subtasks

    async def execute(self, query: str, context: RunContext) -> PlanExecuteResult:
        """Execute Plan and Execute workflow.

        Args:
            query: User's goal/query
            context: Run context with messages and state

        Returns:
            PlanExecuteResult with plan, results, and final answer
        """
        start_time = time.time()
        replans = 0

        try:
            # Phase 1: Initial Planning
            plan = await self._generate_plan(query, context)

            # Phase 2: Execute plan with re-planning on failures
            while replans <= self.max_replans:
                execution_success = await self._execute_plan(plan, context)

                if execution_success:
                    break

                # Re-planning needed
                if replans < self.max_replans:
                    replans += 1
                    logger.info(f"Re-planning (attempt {replans}/{self.max_replans})")
                    plan = await self._replan(plan, context)
                else:
                    logger.warning("Max replans reached, stopping execution")
                    break

            # Phase 3: Synthesize final answer
            final_answer = await self._synthesize_results(plan, query, context)

            # Calculate totals
            total_time = time.time() - start_time
            total_cost = sum(st.cost for st in plan.subtasks)
            total_tokens = sum(st.tokens_used for st in plan.subtasks)

            # Determine stop reason
            if plan.failed_subtasks == 0:
                stop_reason = StopReason.ANSWER_COMPLETE
            elif replans >= self.max_replans:
                stop_reason = StopReason.MAX_STEPS  # Reusing enum value
            else:
                stop_reason = StopReason.FORCED_STOP

            result = PlanExecuteResult(
                plan=plan,
                final_answer=final_answer,
                stop_reason=stop_reason,
                replans=replans,
                total_cost=total_cost,
                total_execution_time=total_time,
                total_tokens=total_tokens,
            )

            # Emit final answer event
            await self._publish_event(
                PlanExecuteEventType.FINAL_ANSWER,
                {"answer": final_answer, "result": result.to_dict()},
            )

            return result

        except Exception as e:
            logger.error(f"Plan and Execute execution failed: {e}", exc_info=True)
            # Return error result
            empty_plan = Plan(subtasks=[], goal=query, reasoning="Execution failed")
            return PlanExecuteResult(
                plan=empty_plan,
                final_answer=f"Error occurred during execution: {str(e)}",
                stop_reason=StopReason.FORCED_STOP,
            )

    async def _generate_plan(self, query: str, context: RunContext) -> Plan:
        """Generate initial plan for the query.

        Args:
            query: User's goal
            context: Run context

        Returns:
            Plan with subtasks
        """
        await self._publish_event(PlanExecuteEventType.PLANNING_START, {"goal": query})

        # Build planning prompt
        tools_info = self.tool_executor.build_tools_description()
        planning_prompt = PLAN_AND_EXECUTE_PLANNING_PROMPT.format(tools=tools_info)

        # Create planning messages
        messages = [
            Message(role=MessageRole.SYSTEM, content=planning_prompt),
            Message(role=MessageRole.USER, content=f"Task to plan: {query}"),
        ]

        # Call LLM to generate plan
        response = await self.tool_executor._client.achat(messages=messages, temperature=0.3)

        # Parse JSON plan
        plan = self._parse_plan_json(response.message.content, query)

        await self._publish_event(
            PlanExecuteEventType.PLANNING_COMPLETE,
            {"plan": plan.to_dict(), "subtask_count": len(plan.subtasks)},
        )

        logger.info(f"Generated plan with {len(plan.subtasks)} subtasks: {plan.reasoning}")

        return plan

    async def _replan(self, current_plan: Plan, context: RunContext) -> Plan:
        """Re-generate plan after failure.

        Args:
            current_plan: Current plan with failures
            context: Run context

        Returns:
            New revised plan
        """
        await self._publish_event(
            PlanExecuteEventType.REPLANNING_START, {"current_plan": current_plan.to_dict()}
        )

        # Find failed subtask
        failed_subtask = next((st for st in current_plan.subtasks if st.status == "failed"), None)

        # Build plan status summary
        plan_status = self._format_plan_status(current_plan)

        # Build replanning prompt
        replan_prompt = PLAN_AND_EXECUTE_REPLAN_PROMPT.format(
            goal=current_plan.goal,
            plan_status=plan_status,
            failed_subtask=failed_subtask.description if failed_subtask else "Unknown",
            error=failed_subtask.error if failed_subtask else "Unknown error",
        )

        # Create replanning messages
        messages = [Message(role=MessageRole.USER, content=replan_prompt)]

        # Call LLM to replan
        response = await self.tool_executor._client.achat(messages=messages, temperature=0.3)

        # Parse new plan
        new_plan = self._parse_plan_json(response.message.content, current_plan.goal)

        await self._publish_event(
            PlanExecuteEventType.REPLANNING_COMPLETE,
            {"new_plan": new_plan.to_dict(), "subtask_count": len(new_plan.subtasks)},
        )

        logger.info(f"Re-planned with {len(new_plan.subtasks)} subtasks: {new_plan.reasoning}")

        return new_plan

    async def _execute_plan(self, plan: Plan, context: RunContext) -> bool:
        """Execute all subtasks in the plan.

        Args:
            plan: Plan to execute
            context: Run context

        Returns:
            True if all subtasks succeeded, False if any failed
        """
        logger.info(f"Executing plan with {len(plan.subtasks)} subtasks")

        # Track completed subtasks for dependency checking
        completed_subtask_ids = set()

        for subtask in plan.subtasks:
            # Check dependencies
            if not self._dependencies_met(subtask, completed_subtask_ids):
                logger.warning(
                    f"Subtask {subtask.id} dependencies not met, skipping: {subtask.dependencies}"
                )
                subtask.status = "failed"
                subtask.error = "Dependencies not satisfied"
                continue

            # Execute subtask
            success = await self._execute_subtask(subtask, context)

            if success:
                completed_subtask_ids.add(subtask.id)
            else:
                # Subtask failed - stop execution and trigger re-planning
                logger.warning(f"Subtask {subtask.id} failed: {subtask.error}")
                return False

            # Publish progress
            await self._publish_event(
                PlanExecuteEventType.PLAN_PROGRESS,
                {
                    "completed": len(completed_subtask_ids),
                    "total": len(plan.subtasks),
                    "progress_percentage": (len(completed_subtask_ids) / len(plan.subtasks)) * 100,
                },
            )

        return True

    async def _execute_subtask(self, subtask: SubTask, context: RunContext) -> bool:
        """Execute a single subtask.

        Args:
            subtask: Subtask to execute
            context: Run context

        Returns:
            True if successful, False otherwise
        """
        subtask.status = "running"
        start_time = time.time()

        await self._publish_event(
            PlanExecuteEventType.SUBTASK_START,
            {"subtask": subtask.to_dict(), "description": subtask.description},
        )

        try:
            if self.use_react_for_subtasks and self.subtask_orchestrator:
                # Use ReAct orchestrator for complex subtasks
                logger.info(f"Executing subtask {subtask.id} with ReAct: {subtask.description}")
                result = await self.subtask_orchestrator.execute(subtask.description, context)

                subtask.result = result.final_answer
                subtask.cost = result.total_cost
                subtask.tokens_used = result.total_tokens
                subtask.status = "completed"

            else:
                # Direct tool execution (simpler, faster)
                logger.info(f"Executing subtask {subtask.id} directly: {subtask.description}")

                # For now, we'll use a simple approach: execute the first required tool
                # In a real implementation, you might parse the description to determine the tool and inputs
                if subtask.required_tools:
                    tool_name = subtask.required_tools[0]
                    # This is simplified - in practice, you'd need to extract tool inputs from the description
                    # or use an LLM to determine inputs
                    tool_result = await self.tool_executor.execute_tool(
                        tool_name, {}, context=context
                    )

                    if tool_result.success:
                        subtask.result = str(tool_result.output)
                        subtask.status = "completed"
                    else:
                        subtask.error = tool_result.error
                        subtask.status = "failed"
                else:
                    # No tools required - might be a reasoning or synthesis task
                    # Use LLM to complete the subtask
                    messages = [
                        Message(
                            role=MessageRole.USER,
                            content=f"Complete this task: {subtask.description}",
                        )
                    ]
                    response = await self.tool_executor._client.achat(messages=messages)
                    subtask.result = response.message.content
                    subtask.status = "completed"

            subtask.execution_time = time.time() - start_time

            await self._publish_event(
                PlanExecuteEventType.SUBTASK_COMPLETE,
                {
                    "subtask": subtask.to_dict(),
                    "result": subtask.result,
                    "execution_time": subtask.execution_time,
                },
            )

            logger.info(
                f"Subtask {subtask.id} completed in {subtask.execution_time:.2f}s: {subtask.result[:100]}..."
            )

            return True

        except Exception as e:
            subtask.status = "failed"
            subtask.error = str(e)
            subtask.execution_time = time.time() - start_time

            await self._publish_event(
                PlanExecuteEventType.SUBTASK_FAILED, {"subtask": subtask.to_dict(), "error": str(e)}
            )

            logger.error(f"Subtask {subtask.id} failed: {e}", exc_info=True)

            return False

    async def _synthesize_results(self, plan: Plan, query: str, context: RunContext) -> str:
        """Synthesize subtask results into final answer.

        Args:
            plan: Executed plan with results
            query: Original user query
            context: Run context

        Returns:
            Final answer string
        """
        # Collect successful subtask results
        results = []
        for subtask in plan.subtasks:
            if subtask.status == "completed" and subtask.result:
                results.append(f"- {subtask.description}: {subtask.result}")

        if not results:
            return "No subtasks completed successfully. Unable to provide an answer."

        # Use LLM to synthesize final answer
        synthesis_prompt = f"""Based on the following subtask results, provide a comprehensive answer to the user's question.

Original Question: {query}

Subtask Results:
{chr(10).join(results)}

Provide a clear, well-formatted final answer that directly addresses the user's question:"""

        messages = [Message(role=MessageRole.USER, content=synthesis_prompt)]
        response = await self.tool_executor._client.achat(messages=messages, temperature=0.5)

        return response.message.content

    def _parse_plan_json(self, json_str: str, goal: str) -> Plan:
        """Parse JSON plan from LLM response.

        Args:
            json_str: JSON string from LLM
            goal: User's goal

        Returns:
            Parsed Plan object
        """
        try:
            # Extract JSON from response (might have markdown code blocks)
            json_str = json_str.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            plan_data = json.loads(json_str)

            # Parse subtasks
            subtasks = []
            for st_data in plan_data.get("subtasks", []):
                subtask = SubTask(
                    id=st_data["id"],
                    description=st_data["description"],
                    required_tools=st_data.get("required_tools", []),
                    dependencies=st_data.get("dependencies", []),
                    success_criteria=st_data.get("success_criteria"),
                )
                subtasks.append(subtask)

            plan = Plan(
                subtasks=subtasks,
                goal=goal,
                reasoning=plan_data.get("reasoning", "No reasoning provided"),
            )

            return plan

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            logger.debug(f"JSON string was: {json_str}")
            # Return empty plan on parse failure
            return Plan(subtasks=[], goal=goal, reasoning=f"Failed to parse plan: {str(e)}")

    def _dependencies_met(self, subtask: SubTask, completed_ids: set) -> bool:
        """Check if subtask dependencies are satisfied.

        Args:
            subtask: Subtask to check
            completed_ids: Set of completed subtask IDs

        Returns:
            True if all dependencies are met
        """
        return all(dep_id in completed_ids for dep_id in subtask.dependencies)

    def _format_plan_status(self, plan: Plan) -> str:
        """Format plan status for re-planning prompt.

        Args:
            plan: Current plan

        Returns:
            Formatted status string
        """
        lines = []
        for st in plan.subtasks:
            status_emoji = {"completed": "✓", "failed": "✗", "pending": "○", "running": "⟳"}.get(
                st.status, "?"
            )
            lines.append(f"{status_emoji} Subtask {st.id}: {st.description} [{st.status}]")
            if st.result:
                lines.append(f"  Result: {st.result[:100]}...")
            if st.error:
                lines.append(f"  Error: {st.error}")

        return "\n".join(lines)

    async def _publish_event(self, event_type: PlanExecuteEventType, data: Dict[str, Any]):
        """Publish event to event bus.

        Args:
            event_type: Type of event
            data: Event data
        """
        event = PlanExecuteEvent(event_type=event_type, data=data)
        await self.event_bus.publish(event)

    def get_current_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {"agent_type": "plan_and_execute_orchestrator"}
