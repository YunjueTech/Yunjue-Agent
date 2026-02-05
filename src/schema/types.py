# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langchain_core.tools import BaseTool
from enum import Enum
class LLMType(Enum):
    BASIC = "basic"
    VISION = "vision"
    SUMMARIZE = "summarize"
    CLUSTER = "cluster"
    TOOL_ANALYZE = "tool_analyze"

class ToolRequest(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]

class StepToolAnalysis(BaseModel):
    """Structured output for step tool analysis."""

    required_tool_names: List[str] = Field(default_factory=list, description="Names of tools to reuse")
    tool_usage_guidance: str = Field("", description="High-level outline describing how to execute the task")
    tool_requests: List[ToolRequest] = Field(
        default_factory=list, description="Definitions for new tools"
    )


class ResponseAnalysis(BaseModel):
    """Structured output for worker response analysis."""

    status: str = Field("RETRY", description="Overall status, e.g., FINISH or RETRY")
    reason: str = Field("", description="Detailed reasoning for the status")


class ToolExecutionRecord(BaseModel):
    tool_name: str
    caller_message_id: str
    tool_message_id: str = ""
    tool_call_id: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None

class TaskExecutionContext(BaseModel):
    bound_tools: List[BaseTool]
    tool_executions: List[ToolExecutionRecord] = []
    context_summary: str = ""

class State(MessagesState):

    user_query: str = "" 
    final_answer: str = ""

    pending_tool_requests: List[ToolRequest] = []
    task_execution_context: Optional[TaskExecutionContext] = None


    # Worker Team Analysis related
    task_failure_report: Optional[str] = None  # Maps step_id to a single failure analysis report
    tool_usage_guidance: Optional[str] = None
    pending_step_response: str = ""  # current step' s pending response before setting true execution_res
    task_execution_count: int = 0


    input_model_error_tools: List[ToolExecutionRecord] = []
    required_tool_names: List[str] = []
    execution_res: str = ""
    worker_exist_messages: List[BaseMessage] = []
    recur_limit_exceeded: bool = False

    # for loss
    cumulative_tool_call_cnt: int = 0