# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, List, Literal, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.schema.types import ToolExecutionRecord, LLMType
from src.utils.context_trimmer import ContextTrimmer
from src.utils.utils import (
    filter_tools_by_names,
    get_preset_tools,
    extract_tool_calls_from_messages,
    tool_enhancement,
)
from src.services.llms.llm import create_llm, get_max_tokens
from src.prompts.loader import prompt_loader
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # Counts how many times we've transitioned into the "tools" node.
    # This is the concrete implementation of `max_steps` for this agent.
    tool_steps: int
    # Counts how many times we've retried (rollback).
    retry_count: int

    tool_call_cnt: int


success_tool_names = set()


class ReActAgent:
    def __init__(
        self,
        llm,
        tools,
        max_steps: Optional[int] = None,
        max_retries: Optional[int] = 10,
        tool_enhance_interval: Optional[int] = None,
        dynamic_tools_dir: str = None,
        dynamic_tools_public_dir: str = None,
        user_query: str = None,
        failure_report: str = None,
        context_summary: str = None,
    ):
        """
        ReAct-style agent built with LangGraph.

        Args:
            llm: Base chat model instance.
            tools: Tool list passed to the underlying ToolNode + model tool binding.
            max_steps: Optional maximum number of graph recursion steps.
            max_retries: Optional maximum number of retry attempts (rollback). Defaults to 3.

        """
        # `max_steps` limits how many times the graph is allowed to enter the "tools" node.
        # (I.e., number of tool-execution iterations.)
        self.max_steps = max_steps
        self.max_retries = max_retries

        self.tools = tools
        self._llm_base = llm
        self._llm_with_tools = llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        self.tool_enhance_interval = tool_enhance_interval
        self.dynamic_tools_dir = dynamic_tools_dir
        self.dynamic_tools_public_dir = dynamic_tools_public_dir
        self.user_query = user_query
        llm_token_limit = get_max_tokens(LLMType.BASIC)
        self.context_trimmer = ContextTrimmer(llm_token_limit, user_query=user_query)
        self.failure_report = failure_report
        self.context_summary = context_summary
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.call_tools)
        workflow.add_node("enhance_tools", self.enhance_tools)
        workflow.add_node("rollback", self.rollback)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.should_continue)
        workflow.add_node("context_summary", self.context_summary_internal)
        workflow.add_conditional_edges("tools", self.need_enhance, ["enhance_tools", "context_summary"])
        workflow.add_edge("enhance_tools", "context_summary")
        workflow.add_edge("context_summary", "agent")
        workflow.add_edge("rollback", "agent")

        self.graph = workflow.compile()

    def need_enhance(self, state):
        tool_steps = state.get("tool_steps", 0)
        if tool_steps % self.tool_enhance_interval == 0:
            return "enhance_tools"
        return "context_summary"

    @staticmethod
    def is_response_empty(response) -> bool:
        """Check if response content is empty."""
        # If response has tool_calls, it's not empty
        if hasattr(response, "tool_calls") and response.tool_calls:
            return False
        
        # Extract content from response
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content_text = "".join(part if isinstance(part, str) else str(part) for part in content)
        else:
            content_text = str(content) if content is not None else ""
        
        return not content_text.strip()

    def context_summary_internal(self, state: AgentState):
        state = context_summary(state, self.context_trimmer)
        return state

    def call_model(self, state: AgentState):
        tool_steps = state.get("tool_steps", 0)
        retry_count = state.get("retry_count", 0)
        # Create a copy of messages to avoid mutating the state
        messages = list(state["messages"])
        if self.max_steps is not None and tool_steps >= self.max_steps:
            return {"messages": ["Recur limit exceeded"], "tool_steps": tool_steps, "retry_count": retry_count}
        else:
            llm_to_use = self._llm_with_tools
        system_prompt = prompt_loader.get_prompt(
            "worker.md",
            **{
                "user_query": self.user_query,
                "failure_report": self.failure_report,
                "context_summary": self.context_summary,
            }
        )
        # Insert SystemMessage at the beginning of the copy, not the original state
        messages.insert(0, SystemMessage(content=system_prompt))
        # Call LLM with retry logic (handled by ChatModel)
        # ChatModel now handles:
        # - Exception handling and retry
        # - Multiple LLM fallback (if configured)
        # Empty responses are handled by the rollback node
        response = llm_to_use.invoke(messages)
        return {"messages": [response], "tool_steps": tool_steps, "retry_count": retry_count}

    def call_tools(self, state: AgentState):
        # Increment tool_steps on each visit to the tools node.
        last_message = state["messages"][-1]
        current = state.get("tool_steps", 0)
        tool_call_cnt = state.get("tool_call_cnt", 0)
        next_steps = current + 1
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call_cnt += len(last_message.tool_calls)

        result = self.tool_node.invoke(state)
        # ToolNode returns a partial state update (typically containing "messages").
        if isinstance(result, dict):
            result["tool_steps"] = next_steps
            result["tool_call_cnt"] = tool_call_cnt
        return result

    def enhance_tools(self, state: AgentState):
        tool_steps = state.get("tool_steps", 0)
        retry_count = state.get("retry_count", 0)
        messages = state["messages"]
        # try to enhance the tool
        enhanced_success_tools, new_messages = asyncio.run(
            enhance_tools(
                messages,
                self.dynamic_tools_dir,
                self.dynamic_tools_public_dir,
                self.user_query,
                self.tools,
            )
        )
        messages = [RemoveMessage(id=message.id) for message in messages] + new_messages
        enhanced_tools_name = set([tool.name for tool in enhanced_success_tools])
        bound_tools = enhanced_success_tools + [
            tool for tool in self.tools if tool.name not in enhanced_tools_name
        ]
        self._llm_with_tools = self._llm_with_tools.bind_tools(bound_tools)

        return {"messages": messages, "tool_steps": tool_steps, "retry_count": retry_count}

    def rollback(self, state: AgentState):
        """
        Rollback node that removes the second-to-last AIMessage and all subsequent messages.
        This is triggered when the last AIMessage is empty, so we need to remove the previous
        AIMessage (which likely had tool_calls) and everything after it.
        """
        messages = state["messages"]
        tool_steps = state.get("tool_steps", 0)
        retry_count = state.get("retry_count", 0) + 1
        
        # Find all AIMessage indices
        ai_message_indices = []
        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage):
                ai_message_indices.append(i)
        
        if len(ai_message_indices) < 2:
            # If there's less than 2 AIMessages, just remove the last message
            logger.warning(f"Less than 2 AIMessages found ({len(ai_message_indices)}), removing last message only")
            if messages:
                return {"messages": [RemoveMessage(id=messages[-1].id)], "tool_steps": tool_steps, "retry_count": retry_count}
            return {"retry_count": retry_count}
        
        # Get the index of the second-to-last AIMessage
        second_to_last_ai_idx = ai_message_indices[-2]
        
        # Collect all messages from the second-to-last AIMessage onwards
        messages_to_remove = messages[second_to_last_ai_idx:]
        
        # Check if any ToolMessages are being removed, if so, decrement tool_steps
        has_tool_messages = any(isinstance(msg, ToolMessage) for msg in messages_to_remove)
        if has_tool_messages and tool_steps > 0:
            tool_steps -= 1
        
        # Create RemoveMessage objects
        remove_messages = [RemoveMessage(id=msg.id) for msg in messages_to_remove]
        
        logger.info(f"Rollback: removing {len(remove_messages)} messages starting from second-to-last AIMessage, tool_steps: {tool_steps}, retry_count: {retry_count}")
        
        return {"messages": remove_messages, "tool_steps": tool_steps, "retry_count": retry_count}

    def should_continue(self, state: AgentState) -> Literal["tools", "rollback", END]:
        last_message = state["messages"][-1]
        tool_steps = state.get("tool_steps", 0)
        retry_count = state.get("retry_count", 0)
        
        # Check if retry limit is exceeded
        if self.max_retries is not None and retry_count >= self.max_retries:
            logger.warning(f"Retry limit exceeded ({retry_count} >= {self.max_retries}), ending execution")
            return END
        
        # Check if the response is empty
        if self.is_response_empty(last_message):
            logger.warning("Empty response detected, routing to rollback node")
            return "rollback"
        
        # Safely check for tool_calls attribute (HumanMessage doesn't have it)
        tool_calls = getattr(last_message, "tool_calls", None)
        if tool_calls:
            # Allow entering tools only if we haven't exhausted the tool-step budget.
            if self.max_steps is not None and tool_steps >= self.max_steps:
                return END
            return "tools"
        
        asyncio.run(enhance_tools(
            state["messages"],
            self.dynamic_tools_dir,
            self.dynamic_tools_public_dir,
            self.user_query,
            self.tools,
        ))
        return END

    def invoke(self, inputs, config=None):
        return self.graph.invoke(inputs, config)

    def stream(self, inputs, config=None):
        return self.graph.stream(inputs, config)

    async def astream(
        self,
        inputs: Any,
        stream_mode: Any = "values",
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Async streaming wrapper around the underlying LangGraph compiled graph."""
        async for item in self.graph.astream(
            inputs,
            stream_mode=stream_mode,
            **kwargs,
        ):
            yield item


async def enhance_tools(
    origin_messages: List[BaseMessage],
    dynamic_tools_dir: str,
    dynamic_tools_public_dir: str,
    user_query: str,
    tools: List,
):
    global success_tool_names
    success_tool_names.update(set([path.stem for path in Path(dynamic_tools_public_dir).glob("*.py")]))

    tool_executions = extract_tool_calls_from_messages(origin_messages)
    tool_executions_to_analyze = [
        tool_execution
        for tool_execution in tool_executions
        if tool_execution.tool_name not in success_tool_names
    ]
    tool_analyze_llm = create_llm(LLMType.TOOL_ANALYZE)
    tool_descriptions = {tool.name: tool.description for tool in tools}
    tool_input_schema = {tool.name: tool.args_schema for tool in tools}
    human_prompts = [
        (
            """Analyze the provided Python code execution result. Classify the result into one of three categories:

1.  **Input Error:** If the result or output contains clear validation failure messages (e.g., 'validation error for InputModel', 'Value error', 'invalid argument', 'missing required parameter').
2.  **Execution Failure:** If the execution encountered a core Python exception (e.g., 'Exception', 'SyntaxError', 'ValueError') that is not an Input Error. Also classify as **Execution Failure** if the tool returns empty. **Note:** HTTP error codes (e.g., 404, 500, 403) or failure due to anti-robot policy should NOT be classified as Execution Failure - they are valid responses indicating resource status.
3.  **Success:** If the execution completed without any exceptions, validation errors, general failure indicators, and produced non-empty, meaningful output.

Respond with only the word 'Input Error', 'Execution Failure', or 'Success'."
"""
            "Tool name: {tool_name}.\n"
            "Tool description: {tool_description}\n"
            "Tool input schema: {tool_input_schema}\n"
            "Tool called input arguments: {tool_arguments}\n"
            "Tool execution result: {tool_result}"
        ).format(
            tool_name=tool_execution.tool_name,
            tool_arguments=tool_execution.arguments,
            tool_description=tool_descriptions[tool_execution.tool_name],
            tool_input_schema=tool_input_schema[tool_execution.tool_name],
            tool_result=tool_execution.result
            if not tool_execution.error
            else f"**Error:** {tool_execution.error}",
        )
        for tool_execution in tool_executions_to_analyze if tool_descriptions.get(tool_execution.tool_name, None) and tool_input_schema.get(tool_execution.tool_name, None)
    ]

    messages = [HumanMessage(human_prompt) for human_prompt in human_prompts]
    tool_analyze_tasks = [tool_analyze_llm.ainvoke([message]) for message in messages]
    tool_analyze_results = await asyncio.gather(*tool_analyze_tasks)
    need_enhancement_tools = []
    input_model_error_tools = []
    success_tools = []
    logger.info(f"tool_analyze_results: {tool_analyze_results}")
    for te, result in zip(tool_executions_to_analyze, tool_analyze_results):
        content = getattr(result, "content", result)
        if isinstance(content, list):
            content_text = "".join(
                part if isinstance(part, str) else json.dumps(part, ensure_ascii=False) for part in content
            )
        else:
            content_text = str(content)
        content_text = content_text.strip()
        if content_text == "Input Error":
            input_model_error_tools.append(te)
        elif content_text == "Execution Failure":
            need_enhancement_tools.append(te)
        elif content_text == "Success":
            success_tools.append(te)
    logger.info(
        f"success_tools: {success_tool_names}, need_enhancement_tools: {need_enhancement_tools}, input_model_error_tools: {input_model_error_tools}"
    )
    success_tool_names.update(set([tool.tool_name for tool in success_tools]))

    input_model_error_tools = [
        tool_execution
        for tool_execution in input_model_error_tools
        if tool_execution.tool_name not in success_tool_names
    ]
    need_enhancement_tools = [
        tool_execution
        for tool_execution in need_enhancement_tools
        if tool_execution.tool_name not in success_tool_names
    ]
    logger.info(
        f"need_enhancement_tools: {need_enhancement_tools}, input_model_error_tools: {input_model_error_tools}, success_tools: {success_tools}"
    )
    enhanced_tools = []
    caller_message_id = set()
    enhanced_tool_call_id = set()
    enhanced_tool_messages_id = set()
    if need_enhancement_tools:
        historical_tool_executions = {}
        for tool_execution in need_enhancement_tools:
            tool_name = tool_execution.tool_name
            tool_arguments = tool_execution.arguments
            tool_result = tool_execution.result
            tool_error = tool_execution.error
            if tool_name not in historical_tool_executions:
                historical_tool_executions[tool_name] = []
            historical_tool_executions[tool_name].append(
                ToolExecutionRecord(
                    tool_name=tool_name,
                    tool_call_id=tool_execution.tool_call_id,
                    caller_message_id=tool_execution.caller_message_id,
                    tool_message_id=tool_execution.tool_message_id,
                    arguments=tool_arguments,
                    result=tool_result,
                    error=tool_error,
                )
            )

        codex_tasks = []
        preset_tools = set([tool.name for tool in get_preset_tools()])
        for tool_name, tool_executions in historical_tool_executions.items():
            if tool_name in preset_tools:
                continue

            # Check both private and public directories for the tool
            tool_filename = os.path.join(dynamic_tools_dir, f"{tool_name}.py")
            if not os.path.exists(tool_filename):
                # Try public directory
                tool_filename = os.path.join(dynamic_tools_public_dir, f"{tool_name}.py")
                if not os.path.exists(tool_filename):
                    logger.info(
                        f"Tool {tool_name} not found in private or public directories, skip enhancement"
                    )
                    continue

            # Always enhance to private directory
            codex_tasks.append(
                tool_enhancement(
                    tool_filename,
                    tool_executions,
                    dynamic_tools_dir, 
                )
            )
        if codex_tasks:
            codex_results = await asyncio.gather(*codex_tasks)
            enhanced_tool_names = [
                os.path.basename(codex_result).split(".")[0] for codex_result in codex_results if codex_result
            ]
            logger.info(f"Enhanced tool names: {enhanced_tool_names}")
            enhanced_tools = await filter_tools_by_names(
                enhanced_tool_names, dynamic_tools_dir, dynamic_tools_public_dir, user_query
            )

        enhanced_tool_messages_id = set([tool_info.tool_message_id for tool_info in need_enhancement_tools])
        enhanced_tool_call_id = set([tool_info.tool_call_id for tool_info in need_enhancement_tools])
        caller_message_id = set([tool_info.caller_message_id for tool_info in need_enhancement_tools])
    new_messages = []
    for message in origin_messages:
        if isinstance(message, ToolMessage):
            if message.id in enhanced_tool_messages_id:
                continue
        elif isinstance(message, AIMessage) and hasattr(message, "tool_calls") and message.tool_calls:
            if message.id in caller_message_id:
                new_ai_message = deepcopy(message)
                new_ai_message.tool_calls = [
                    tool_call
                    for tool_call in message.tool_calls
                    if tool_call.get("id") not in enhanced_tool_call_id
                ]
                if new_ai_message.tool_calls:
                    new_messages.append(new_ai_message)
                continue
        new_messages.append(deepcopy(message))

    return enhanced_tools, new_messages


def context_summary(
    origin_state,
    context_trimmer,
):
    messages = origin_state["messages"]
    if context_trimmer.is_exceeded(messages):
        return context_trimmer.trim(origin_state)
    return origin_state
