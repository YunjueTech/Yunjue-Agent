# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import os
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agents import ReActAgent
from src.schema.types import State, TaskExecutionContext, LLMType
from src.config.config import Configuration
from src.services.llms.llm import create_llm, get_max_tokens
from src.prompts.loader import prompt_loader
from src.utils.context_trimmer import ContextTrimmer
from src.utils.utils import (
    analyze_task_tools,
    call_codex_exec,
    filter_tools_by_names,
    analyze_response,
    extract_tool_calls_from_messages,
    parse_markdown_sections,
    summarize_context,
)

logger = logging.getLogger(__name__)

class GiveAnswerResponse(BaseModel):
    """Structured output aligned with give_answer JSON schema."""

    final_answer: str = Field("", description="Direct answer in required format")
    reasoning_summary: str = Field(
        "",
        description="Brief justification (1-2 sentences) based on findings",
    )


async def integrator_node(state: State, config: RunnableConfig):
    """Integrator node that extract final answer."""
    logger.info("Using give_answer mode for QA task")
    template_name = "give_answer"

    user_query = state.get("user_query", "")

    system_prompt = prompt_loader.get_prompt("give_answer.md", **{
        "user_query": user_query,
    })
    invoke_messages = [SystemMessage(content=system_prompt)]

    # todo: give all findings?
    execution_res = state.get("execution_res", "")
    observation_messages = [
        HumanMessage(
            content=execution_res,
            name="Findings",
        )
    ]

    llm_token_limit = get_max_tokens(LLMType.BASIC)
    compressed_state = ContextTrimmer(llm_token_limit).trim(
        {"messages": observation_messages}
    )
    invoke_messages += compressed_state.get("messages", [])
    logger.debug(f"Current invoke messages: {invoke_messages}")
    integrator_llm = create_llm(LLMType.BASIC).with_structured_output(
        GiveAnswerResponse,
        method="json_mode",
    )

    response_content = ""
    structured_payload = None
    try:
        structured_payload = await integrator_llm.ainvoke(invoke_messages)
    except Exception as e:
        logger.error(f"Integrator generation failed: {e}")

    if structured_payload:
        response_content = json.dumps(
            {
                "final_answer": structured_payload.final_answer,
                "reasoning_summary": structured_payload.reasoning_summary,
            },
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"{template_name} response: {response_content}")

    return {"final_answer": response_content, "cumulative_tool_call_cnt": state.get("cumulative_tool_call_cnt", 0)}


async def manager_node(
    state: State, config: RunnableConfig
) -> Command[Literal["integrator", "manager", "tool_developer"]]:
    logger.info("Manager is analyzing the task.")
    configurable = Configuration.resolve(config)
    max_task_execution_cnt = configurable.max_task_execution_cnt
    user_query = state.get("user_query", "")

    task_execution_context = state.get("task_execution_context", None)
    if not task_execution_context:
        task_execution_context = TaskExecutionContext(
            bound_tools=[],
        )
        if state.get("pending_step_response"):
            error_msg = f"This is a new step, but the pending_step_response is not a empty str. It is {state.get('pending_step_response')}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    current_task_execution_cnt = state.get("task_execution_count", 0)
    pending_responses = state.get("pending_step_response", "")
    context_summary = task_execution_context.context_summary
    analyze_tasks = []

    user_query = state.get("user_query", "")
    failure_report = ""
    update_payload = {}
    tool_usage_guidance = state.get("tool_usage_guidance", None)
    recur_limit_exceeded = state.get("recur_limit_exceeded", False)
    if current_task_execution_cnt >= 1:  
        logger.info(
            f"current_step_execution_cnt: {current_task_execution_cnt}"
        )

        if pending_responses and not recur_limit_exceeded:
            analyze_tasks.append(analyze_response(pending_responses))

        results = await asyncio.gather(*analyze_tasks)
        check_pass = not recur_limit_exceeded
        suggestions = ""
        for result in results:
            diagnosis, suggestion = result
            check_pass = check_pass and diagnosis
            if not diagnosis:
                suggestions = f"{suggestions}\n{suggestion}"

        if check_pass:
            return Command(
                update={
                    "task_execution_context": task_execution_context,
                    "pending_step_response": "",
                    "execution_res": pending_responses,
                },
                goto="integrator",  
            )
        else:
            if current_task_execution_cnt >= max_task_execution_cnt:
                execution_res = ""
                if pending_responses:
                    execution_res = pending_responses   
                elif context_summary:
                    execution_res = context_summary
                else:
                    execution_res = "No response from worker."
                return Command(
                    update={
                        "task_execution_context": task_execution_context,
                        "pending_step_response": "",
                        "execution_res": execution_res,
                    },
                    goto="integrator",
                )

            failure_report = suggestions
            update_payload = {
                "pending_step_response": "",
                "task_failure_report": failure_report,
            }

    dynamic_tools_private_dir = configurable.dynamic_tools_dir
    dynamic_tools_public_dir = configurable.dynamic_tools_public_dir
    if recur_limit_exceeded:
        additional_tool_requests = parse_markdown_sections(pending_responses).get('additional_tool_requirement', [])
    else:
        additional_tool_requests = []
    required_tool_names, tool_usage_guidance, tool_requests = await analyze_task_tools(
        user_query,
        dynamic_tools_private_dir,
        dynamic_tools_public_dir,
        failure_report,
        additional_tool_requests=additional_tool_requests,
    )
    update_payload = {
        **update_payload,
        "task_execution_context": task_execution_context,
        "required_tool_names": required_tool_names,
        "pending_tool_requests": tool_requests,
        "tool_usage_guidance": tool_usage_guidance,
    }

    return Command(update=update_payload, goto="tool_developer")

async def _build_single_tool(tool_req, tool_index: int, total: int, dynamic_tools_dir: str) -> dict:
    """
    Build a single tool with retry logic.

    Returns:
        dict with keys: tool_req, success (bool), tool_filename (str|None),
        evaluation_payload (dict|None), error_message (str|None)
    """
    tool_name = getattr(tool_req, "name", "generated_tool")
    logger.info(f"Building tool {tool_index + 1}/{total}: {tool_name}")

    # TODO: move to config
    max_retries = 3
    os.makedirs(dynamic_tools_dir, exist_ok=True)

    safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", tool_name).strip("_") or "generated_tool"
    tool_filename = os.path.join(dynamic_tools_dir, f"{safe_name}.py")
    suffix = 1
    while os.path.exists(tool_filename):
        tool_filename = os.path.join(dynamic_tools_dir, f"{safe_name}_{suffix:02d}.py")
        suffix += 1

    tool_request_json = json.dumps(tool_req.model_dump(), ensure_ascii=False)

    for attempt in range(1, max_retries + 1):
        logger.info(f"Tool {tool_name}: Generating code (attempt {attempt}/{max_retries})")


        proxy_url = os.environ.get("PROXY_URL", None)

        builder_prompt = prompt_loader.get_prompt(
            "toolsmiths_agent.md",
            **{
                "tool_request_json": tool_request_json,
                "proxy_url": proxy_url,
            }
        )

        if os.path.exists(tool_filename):
            try:
                os.remove(tool_filename)
            except Exception as cleanup_exc:  # pragma: no cover - best effort cleanup
                logger.warning(f"Failed to remove existing tool file {tool_filename}: {cleanup_exc}")

        code, build_success = await call_codex_exec(builder_prompt, tool_filename)
        if not build_success:
            last_error_message = f"Codex exec failed to generate tool code: {code}"
            logger.warning(f"Tool {tool_name}: Attempt {attempt}/{max_retries} failed: {last_error_message}")
            continue

        if not os.path.exists(tool_filename):
            last_error_message = f"Codex exec did not create tool file at {tool_filename}"
            logger.warning(f"Tool {tool_name}: Attempt {attempt}/{max_retries} failed: {last_error_message}")
            continue

        return {
            "tool_req": tool_req,
            "success": True,
            "tool_filename": tool_filename,
        }

    # All retries failed
    logger.error(
        "Failed to build tool %s after %s attempts. Last error: %s",
        tool_name,
        max_retries,
        last_error_message,
    )

    return {
        "tool_req": tool_req,
        "success": False,
        "tool_filename": None,
    }


async def tool_developer_node(
    state: State, config: RunnableConfig
) -> Command[Literal["executor", "integrator"]]:
    """Build tools using Codex exec and evaluate them with concurrent processing and retry loops."""

    pending_tool_requests = state.get("pending_tool_requests", [])
    configurable = Configuration.resolve(config)

    user_query = state.get("user_query", "")
    task_execution_context = state.get("task_execution_context", None)
    dynamic_tools_dir = configurable.dynamic_tools_dir
    dynamic_tools_public_dir = configurable.dynamic_tools_public_dir
    dynamic_tools_private_dir = configurable.dynamic_tools_dir
    required_tool_names = state.get("required_tool_names", [])
    if not pending_tool_requests:
        logger.info("No pending tool requests; bound tools to worker and run")
        bound_tools = await filter_tools_by_names(
            required_tool_names, dynamic_tools_private_dir, dynamic_tools_public_dir, user_query
        )
        task_execution_context.bound_tools = bound_tools
        update_payload = {
            "task_execution_context": task_execution_context,
        }
        return Command(
            update=update_payload,
            goto="executor",
        )

    total_tools = len(pending_tool_requests)
    logger.info(f"Starting concurrent build of {total_tools} tool(s)")

    configurable = Configuration.resolve(config)
    dynamic_tools_dir = configurable.dynamic_tools_dir
    # Create tasks for all tool requests
    build_tasks = [
        _build_single_tool(tool_req, idx, total_tools, dynamic_tools_dir)
        for idx, tool_req in enumerate(pending_tool_requests)
    ]

    # Execute all builds concurrently
    results = await asyncio.gather(*build_tasks, return_exceptions=True)

    # Process results
    successful_tools = []
    failed_tools = []

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Unexpected exception during tool build: {result}")
            failed_tools.append(
                {
                    "tool_req": None,
                    "error_message": str(result),
                }
            )
            continue

        if result["success"]:
            successful_tools.append(result)
            logger.info(f"Tool {result['tool_req'].name} built successfully: {result['tool_filename']}")
        else:
            failed_tools.append(result)
            logger.warning(f"Tool {result['tool_req'].name} failed")

    # Clear all pending requests (all have been processed)

    # Log summary
    logger.info(
        f"Tool building completed: {len(successful_tools)} successful, {len(failed_tools)} failed out of {total_tools} total"
    )

    if failed_tools:
        for failed in failed_tools:
            if failed.get("tool_req"):
                logger.warning(f"Failed tool: {failed['tool_req'].name}")
                return Command(goto="integrator")
    required_tool_names = required_tool_names + [
        os.path.basename(tool["tool_filename"]).split(".")[0] for tool in successful_tools
    ]
    bound_tools = await filter_tools_by_names(
        required_tool_names, dynamic_tools_private_dir, dynamic_tools_public_dir, user_query
    )
    task_execution_context.bound_tools = bound_tools
    update = {
        "pending_tool_requests": [],
        "task_execution_context": task_execution_context
    }
    # Always return to manager after processing all tools
    return Command(update=update, goto="executor")


async def executor_node(
    state: State, config: RunnableConfig
) -> Command[Literal["manager", "__end__", "executor"]]:
    """Helper function to execute a step using the specified agent."""
    configurable = Configuration.resolve(config)
    user_query = state.get("user_query", "")
    worker_exist_messages = state.get("worker_exist_messages", [])
    # Get user query from workflow state (stored globally during initialization)
    user_query = state.get("user_query", "")

    agent_input = {
        "messages": [],
        "user_query": user_query,  # Pass user query to agent state
    }

    # Invoke the agent
    default_recursion_limit = int(os.environ.get("MAX_WORKER_RECURSION_LIMIT", 10))
    tool_enhance_interval = int(os.environ.get("WORKER_TOOL_ENHANCE_INTERVAL", 2))
    # Check if there's a failure report for this step (retry scenario)
    failure_report = state.get("task_failure_report", None)
    task_execution_context = state.get("task_execution_context", None)
    tool_usage_guidance = state.get("tool_usage_guidance", None)
    context_summary = task_execution_context.context_summary

    bound_tools = task_execution_context.bound_tools or []

    tools = list(bound_tools)  # Start with bound tools

    tool_names = []
    for tool in tools:
        if hasattr(tool, "name") and tool.name:
            tool_names.append(tool.name)
        elif hasattr(tool, "__name__"):
            tool_names.append(tool.__name__)
        else:
            tool_names.append(str(type(tool).__name__))
    logger.info(f"Worker tools: {tool_names}")

    # Build worker LLM here (as requested) and render system prompt from `src/prompts/worker.md`
    llm = create_llm(LLMType.BASIC)

    agent = ReActAgent(
        llm,
        tools,
        max_steps=default_recursion_limit,
        tool_enhance_interval=tool_enhance_interval,
        dynamic_tools_dir=configurable.dynamic_tools_dir,
        dynamic_tools_public_dir=configurable.dynamic_tools_public_dir,
        user_query=user_query,
        failure_report=failure_report,
        context_summary=context_summary,
    )

    task_info = f"""# Task
{user_query}
{"## Previous Failure Report" if failure_report else ""}
{failure_report if failure_report else ""}
{"## Tool Usage Guidance\n" + tool_usage_guidance + "\n" if tool_usage_guidance else ""}
"""
    agent_input["messages"].append(HumanMessage(content=task_info))
    agent_input["messages"].extend(worker_exist_messages)
    # Use stream to handle recursion limit and collect all messages
    all_messages = []
    recur_limit_exceeded = False
    final_state = None
    try:
        async for stream_state in agent.astream(
            {"messages": agent_input["messages"]},
            stream_mode="values",
            config={"recursion_limit": 1000}
        ):
            final_state = stream_state
            if isinstance(stream_state, dict):
                # Check if tool_steps has reached max_steps to detect recur limit exceeded
                tool_steps = stream_state.get("tool_steps", 0)
                if agent.max_steps is not None and tool_steps >= agent.max_steps:
                    recur_limit_exceeded = True
                    logger.info(f"Recur limit exceeded: tool_steps={tool_steps}, max_steps={agent.max_steps}")
                
                if "messages" in stream_state:
                    all_messages = list(stream_state["messages"])
    except Exception as e:
        logger.error(f"Error during agent stream execution: {e}")
        # Fallback to exception-based detection if stream_state doesn't contain tool_steps
        recur_limit_exceeded = True
    
    tool_call_cnt = final_state.get("tool_call_cnt", 0)
    last_message = all_messages[-1]

    context_summary = ""
    new_messages = []
    for message in all_messages:
        if isinstance(message, HumanMessage):
            if message.content.startswith("## Context Summary"):
                new_messages.append(message)
                context_summary = message.content
        elif isinstance(message, AIMessage) or isinstance(message, ToolMessage):
            new_messages.append(message)
    new_tool_executions = extract_tool_calls_from_messages(new_messages)
    if recur_limit_exceeded:
        context_summary = await summarize_context(user_query, new_tool_executions, context_summary, True)

    if context_summary:
        task_execution_context.context_summary = context_summary
    
    final_response = parse_markdown_sections(context_summary)
    if recur_limit_exceeded:
        worker_exist_messages = [ 
            HumanMessage(
                content=f"## Context Summary\n### Key Findings\n{final_response.get('task_relevant_key_findings', '')}\n"
            )
        ]
    else:
        worker_exist_messages = new_messages


    if not recur_limit_exceeded:
        pending_responses = last_message.content
    else:  
        pending_responses = context_summary

    logger.info(
        f"Step '{user_query}' response stored for evaluation (length: {len(pending_responses)} chars)"
    )

    update_load = {
        "task_execution_context": task_execution_context,
        "pending_step_response": pending_responses,
        "task_execution_count": state.get("task_execution_count", 0) + 1,
        "worker_exist_messages": worker_exist_messages,
        "recur_limit_exceeded": recur_limit_exceeded,
        "cumulative_tool_call_cnt": state.get("cumulative_tool_call_cnt", 0) + tool_call_cnt,
    }

    return Command(
        update=update_load,
        goto="manager",
    )
