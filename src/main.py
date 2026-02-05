# Copyright (c) 2026 Yunjue Tech
# SPDX-License-Identifier: Apache-2.0
import logging
from contextvars import ContextVar
from pathlib import Path
from typing import Optional
import os
from src.core import build_graph
from src.agents.react import success_tool_names

# Context variable to store task_id for each coroutine
task_id_context: ContextVar[Optional[str]] = ContextVar("task_id", default=None)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level is INFO
    format="- %(name)s - %(levelname)s - %(message)s",
)

class TaskIdFilter(logging.Filter):
    """Filter to only allow logs from the current task_id context."""

    def __init__(self, task_id: str):
        super().__init__()
        self.task_id = task_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Only allow logs from the matching task_id."""
        current_task_id = task_id_context.get()
        return current_task_id == self.task_id

def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    logging.getLogger("src").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)

# Create the graph
builder = build_graph()

async def run_task(
    user_input: str,
    run_dir: Path,
    debug: bool = False,
    task_id: str = "default",
):
    if not user_input:
        raise ValueError("Input could not be empty")

    graph = builder.compile()
    # Set task_id in context for this coroutine
    token = task_id_context.set(task_id)

    # Setup task-specific file logging with filter
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"task_{task_id}.log"

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("- %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    # Add filter to only log messages from this task_id
    file_handler.addFilter(TaskIdFilter(task_id))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    try:
        if debug:
            enable_debug_logging()

        initial_state = {
            "user_query": user_input,  # Store original user query for global access
        }
        config = {
            "configurable": {
                "thread_id": task_id,
                "dynamic_tools_dir": f"{run_dir}/private_dynamic_tools/dynamic_tools_{task_id}",
                "dynamic_tools_public_dir": f"{run_dir}/dynamic_tools_public",
            },
            "recursion_limit": 1000
        }

        # Ensure dynamic tools directories exist
        Path(config["configurable"]["dynamic_tools_dir"]).mkdir(parents=True, exist_ok=True)
        Path(config["configurable"]["dynamic_tools_public_dir"]).mkdir(parents=True, exist_ok=True)

        final_state = await graph.ainvoke(input=initial_state, config=config)

        logger.info("The task has completed successfully")
        private_dynamic_tools_dir = Path(config["configurable"]["dynamic_tools_dir"])
        private_dynamic_tools_files = list(private_dynamic_tools_dir.glob("*.py"))
        for file in private_dynamic_tools_files:
            if os.path.basename(file).split(".")[0] not in success_tool_names:
                os.remove(file)
        return final_state["final_answer"], final_state["cumulative_tool_call_cnt"]
    except Exception as e:
        logger.error(f"Error in the task: {e}", exc_info=True)
        return "['Error in the task']", 0
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()
        task_id_context.reset(token)

