import argparse
import asyncio
import json
import logging
import multiprocessing
import os
import re
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from pydantic import BaseModel, Field

from dataloader import load_dataset
from src.services.llms.llm import create_llm
from src.prompts.loader import prompt_loader
from src.tools.utils import extract_tool_info
from src.utils.utils import call_codex_exec
from src.main import run_task
from src.schema.types import LLMType

def count_unique_tools(tools_dir: str) -> int:
    """Count unique tools by normalizing names (removing version suffixes like _01_01_01)."""
    tools_path = Path(tools_dir)
    unique_tools = set()
    for file in tools_path.glob("*.py"):
        name = file.stem  # remove .py
        # Remove trailing _digits sequences
        name = re.sub(r"(_(\d+))+$", "", name)
        unique_tools.add(name)
    return len(unique_tools)


logger = logging.getLogger(__name__)

TRAIN_STATE_FILENAME = "train_state.json"


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    return Path(run_dir) / "checkpoints" / f"checkpoint_{step}"


def save_train_state(
    run_dir: Path, step: int, total_tool_call_cnt: int, total_needed_tool_cnt: int
) -> None:
    """
    Save counters into {run_dir}/checkpoints/checkpoint_{step}/train_state.json.
    Uses atomic replace to avoid partial writes.
    """
    ckpt_dir = checkpoint_dir(run_dir, step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state_path = ckpt_dir / TRAIN_STATE_FILENAME
    tmp_path = ckpt_dir / f".{TRAIN_STATE_FILENAME}.tmp"
    state = {
        "total_tool_call_cnt": int(total_tool_call_cnt),
        "total_needed_tool_cnt": int(total_needed_tool_cnt),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, state_path)


def load_train_state(run_dir: Path, step: int) -> tuple[int, int] | None:
    """
    Load counters from {run_dir}/checkpoints/checkpoint_{step}/train_state.json.
    Returns (total_tool_call_cnt, total_needed_tool_cnt), or None if not available.
    """
    ckpt_dir = checkpoint_dir(run_dir, step)
    if not ckpt_dir.exists():
        return None
    state_path = ckpt_dir / TRAIN_STATE_FILENAME
    if not state_path.exists():
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return int(state.get("total_tool_call_cnt", 0)), int(state.get("total_needed_tool_cnt", 0))
    except Exception as e:
        logger.warning(f"Failed to load train state from {state_path}: {e}")
        return None


class ToolCluster(BaseModel):
    suggested_master_tool_name: str = Field(..., description="Consolidated cluster name")
    tool_names: List[str] = Field(..., description="Names of tools in the cluster")


class ToolClusterResponse(BaseModel):
    consolidated_tool_clusters: List[ToolCluster] = Field(..., description="Clustered tool list")

def cluster_tools(tool_meta_list: List[dict]):
    tool_cluster_llm = create_llm(LLMType.CLUSTER).with_structured_output(
        ToolClusterResponse,
        method="json_mode",
    )
    tool_cluster_prompt = prompt_loader.get_prompt("tool_cluster.md",
        **{"available_tools": tool_meta_list,}
    )
    messages = [HumanMessage(content=tool_cluster_prompt)]
    try:
        tool_cluster_response = tool_cluster_llm.invoke(messages)
        return [cluster.model_dump() for cluster in tool_cluster_response.consolidated_tool_clusters]
    except Exception as e:
        logger.warning("Tool clustering failed: %s", e)
    
    return naive_cluster_tools(tool_meta_list)


def naive_cluster_tools(tool_meta_list: List[dict]):
    tool_clusters = {}
    for tool_meta in tool_meta_list:
        tool_name = tool_meta.get("name")
        base_name = os.path.basename(tool_name)
        base_name = re.sub(r"(_(\d+))+$", "", base_name)
        if base_name not in tool_clusters:
            tool_clusters[base_name] = []
        tool_clusters[base_name].append(tool_name)
    return [
        {
            "suggested_master_tool_name": base_name,
            "tool_names": tool_names,
        }
        for base_name, tool_names in tool_clusters.items()
    ]


def tool_name_tokens(tool_name: str) -> tuple[str, ...]:
    base_name = os.path.basename(tool_name)
    base_name = re.sub(r"(_\d+)$", "", base_name)
    tokens = [token for token in base_name.split("_") if token]
    return tuple(sorted(tokens))


def is_permutation_cluster(tool_cluster: dict) -> bool:
    tokens_signature: tuple[str, ...] | None = None
    for tool_name in tool_cluster.get("tool_names", []):
        tokens = tool_name_tokens(tool_name)
        if not tokens:
            return False
        if tokens_signature is None:
            tokens_signature = tokens
            continue
        if tokens_signature != tokens:
            return False
    return bool(tokens_signature)


async def merge_tools(tool_cluster: dict, source_dir: str):
    need_merge_tools = tool_cluster["tool_names"]
    tool_merge_input = [
        {"idx": idx, "name": tool_name, "code": open(Path(f"{tool_name}.py"), "r").read()}
        for idx, tool_name in enumerate(need_merge_tools)
    ]
    # Get proxy URL from environment variable
    proxy_url = os.environ.get("PROXY_URL", None)
    tool_merge_prompt = prompt_loader.get_prompt("tool_merge.md",
        **{
            "tools_code": tool_merge_input,
            "suggest_name": tool_cluster["suggested_master_tool_name"],
            "proxy_url": proxy_url,
        }
    )
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        tool_merge_response, success = await call_codex_exec(
            tool_merge_prompt, f"{source_dir}/{tool_cluster['suggested_master_tool_name']}.py"
        )
        if not success:
            logger.error(f"Failed to merge tools: {tool_merge_response}")
            continue
        if not os.path.exists(f"{source_dir}/{tool_cluster['suggested_master_tool_name']}.py"):
            logger.error(f"Failed to merge tools: {tool_merge_response}")
            continue
        return True, tool_cluster
    return False, tool_cluster


async def naive_merge_tools(tool_cluster: dict, source_dir: str):
    try:
        target_file_name = f"{source_dir}/{tool_cluster['suggested_master_tool_name']}.py"

        source_file_name = ""
        min_size = float("inf")
        for tool_name in tool_cluster["tool_names"]:
            file_size = os.path.getsize(f"{tool_name}.py")
            if file_size < min_size:
                min_size = file_size
                source_file_name = f"{tool_name}.py"
        shutil.copy(source_file_name, target_file_name)
    except Exception as e:
        logger.error(f"Failed to merge tools: {e}")
        return False, tool_cluster
    return True, tool_cluster


async def optimize_tools(task_ids: List[str], step: int, run_dir: Path, merge_policy: str = "naive"):
    dynamic_tools_dirs = [f"{run_dir}/private_dynamic_tools/dynamic_tools_{task_id}" for task_id in task_ids]
    public_tool_dir = f"{run_dir}/dynamic_tools_public"
    target_dir = f"{run_dir}/checkpoints/checkpoint_{step}"
    os.makedirs(target_dir, exist_ok=True)

    python_files = Path(public_tool_dir).glob("*.py")
    for file in python_files:
        file_name = os.path.basename(file)
        source_item = os.path.join(public_tool_dir, file_name)
        target_item = os.path.join(target_dir, file_name)
        shutil.move(source_item, target_item)
    dynamic_tools_dirs.append(target_dir)
    dynamic_tools_files = [
        file for dynamic_tools_dir in dynamic_tools_dirs for file in Path(dynamic_tools_dir).glob("*.py")
    ]
    tool_meta_list = []
    for file in dynamic_tools_files:
        extraction_success, tool_info, extraction_error = extract_tool_info(file)
        if extraction_success:
            tool_meta_list.append(
                {
                    "name": os.path.splitext(file)[0],
                    "description": tool_info["tool_description"],
                    "input_schema_code": tool_info["input_schema_code"],
                }
            )
        else:
            logger.error(f"Failed to extract tool info for {file}: {extraction_error}")
    logger.info(
        f"Before optimize({len(tool_meta_list)}): {[tool_meta['name'] for tool_meta in tool_meta_list]}"
    )
    tool_clusters = cluster_tools(tool_meta_list)
    logger.info(f"Tool clusters: {tool_clusters}")


    tool_clusters_to_merge = [
        tool_cluster for tool_cluster in tool_clusters if len(tool_cluster["tool_names"]) > 1
    ]
    tool_merge_results: list[tuple[bool, dict]] = []
    if merge_policy == "naive":
        naive_merge_candidates: list[dict] = []
        for tool_cluster in tool_clusters_to_merge:
            tool_names = [
                tool_name for tool_name in tool_cluster["tool_names"] if public_tool_dir not in tool_name
            ]
            tool_cluster["tool_names"] = tool_names
            naive_merge_candidates.append(tool_cluster)
        if naive_merge_candidates:
            tool_merge_results.extend(
                await asyncio.gather(
                    *[
                        naive_merge_tools(tool_cluster, public_tool_dir)
                        for tool_cluster in naive_merge_candidates
                    ]
                )
            )
    else:
        naive_merge_candidates: list[dict] = []
        llm_merge_candidates: list[dict] = []
        for tool_cluster in tool_clusters_to_merge:
            # naive_merge_candidates.append(tool_cluster)
            if is_permutation_cluster(tool_cluster):
                # Remove public tools from the cluster
                tool_names = [
                    tool_name for tool_name in tool_cluster["tool_names"] if public_tool_dir not in tool_name
                ]
                tool_cluster["tool_names"] = tool_names
                naive_merge_candidates.append(tool_cluster)
            else:
                llm_merge_candidates.append(tool_cluster)

        if naive_merge_candidates:
            tool_merge_results.extend(
                await asyncio.gather(
                    *[
                        naive_merge_tools(tool_cluster, public_tool_dir)
                        for tool_cluster in naive_merge_candidates
                    ]
                )
            )
        if llm_merge_candidates:
            tool_merge_results.extend(
                await asyncio.gather(
                    *[merge_tools(tool_cluster, public_tool_dir) for tool_cluster in llm_merge_candidates]
                )
            )
    for success, tool_cluster in tool_merge_results:
        if not success:
            tool_names = tool_cluster["tool_names"]
            logger.info(f"Failed to merge tools: {tool_cluster['suggested_master_tool_name']}")
            for tool_name in tool_names:
                source_path = f"{tool_name}.py"
                if not os.path.exists(source_path):
                    logger.warning(f"Skip copying missing tool file: {source_path}")
                    continue
                base_name = os.path.basename(tool_name)
                base_name = re.sub(r"(_\d+)$", "", base_name)
                target_name = f"{public_tool_dir}/{base_name}"
                suff = 1
                while os.path.exists(f"{target_name}.py"):
                    target_name = f"{public_tool_dir}/{base_name}_{suff:02d}"
                    suff += 1
                shutil.copy(source_path, f"{target_name}.py")

    unique_tool_names = [
        tool_cluster["tool_names"][0]
        for tool_cluster in tool_clusters
        if len(tool_cluster["tool_names"]) == 1
    ]
    for tool_name in unique_tool_names:
        base_name = os.path.basename(tool_name)
        if not os.path.exists(f"{public_tool_dir}/{base_name}.py"):
            shutil.copy(f"{tool_name}.py", f"{public_tool_dir}/{base_name}.py")

    optimized_tools = list(Path(public_tool_dir).glob("*.py"))
    logger.info(f"After optimize({len(optimized_tools)}): {optimized_tools}")

    return True

def compute_loss(new_tool_call_cnt, total_tool_call_cnt, new_needed_tool_cnt, total_needed_tool_cnt):
    total_tool_call_cnt += new_tool_call_cnt
    total_needed_tool_cnt += new_needed_tool_cnt
    return total_needed_tool_cnt / total_tool_call_cnt


def run_task_process(
    query: str,
    run_dir: Path,
    task_id: str,
    timeout: int = 5400,
):
    """Wrapper to run the async task in a separate process."""

    async def run_with_timeout():
        try:
            return await asyncio.wait_for(
                run_task(
                    query,
                    run_dir,
                    task_id=task_id,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.error(f"Task {task_id} timed out after {timeout} seconds")
            return ({"error": f"Timeout after {timeout} seconds", "task_id": task_id}, {})

    result = asyncio.run(run_with_timeout())
    return result


async def train(
    data_iter: Iterable[dict],
    train_steps: int,
    start: int,
    run_dir: Path,
    prediction_file: str,
    timeout: int,
    merge_policy: str = "naive",
):
    manager = multiprocessing.Manager()
    step = 0
    total_queries = 0
    total_tool_call_cnt = 0
    total_needed_tool_cnt = 0
    if start > 0:
        resume_step = start - 1
        restored = load_train_state(run_dir, resume_step)
        if restored is not None:
            total_tool_call_cnt, total_needed_tool_cnt = restored
            logger.info(
                f"Restored counters from checkpoint_{resume_step}: "
                f"total_tool_call_cnt={total_tool_call_cnt}, total_needed_tool_cnt={total_needed_tool_cnt}"
            )
    try:
        for data in data_iter:
            data_items = data["data_items"]
            if step < start:
                step += 1
                total_queries += len(data_items)
                continue
            task_ids = [data_item["task_id"] for data_item in data_items]

            # Prepare arguments for multiprocessing
            process_args = [
                (
                    data_item["query"],
                    run_dir,
                    data_item["task_id"],
                    timeout,
                )
                for idx, data_item in enumerate(data_items)
            ]

            # Use multiprocessing to run tasks in parallel processes.
            # Per-task apply_async + polling lets us keep completed results even if some workers hang.
            with multiprocessing.Pool(processes=len(data_items)) as pool:
                async_jobs = [pool.apply_async(run_task_process, args) for args in process_args]
                deadline = time.time() + timeout + 60  # small buffer for teardown
                results = [None] * len(async_jobs)
                pending: set[int] = set(range(len(async_jobs)))

                while pending and time.time() < deadline:
                    done_now = []
                    for idx in list(pending):
                        if async_jobs[idx].ready():
                            try:
                                results[idx] = async_jobs[idx].get()
                            except Exception as e:
                                results[idx] = {"error": str(e), "task_id": task_ids[idx]}, 0
                            done_now.append(idx)
                    for idx in done_now:
                        pending.discard(idx)
                    if pending:
                        time.sleep(1)  # brief backoff

                # Mark any leftover tasks as timed out and drop their private tools.
                timed_out_task_ids: set[str] = set()

                for idx in pending:
                    results[idx] = {"error": f"Timeout after {timeout} seconds", "task_id": task_ids[idx]}, 0
                    timed_out_task_ids.add(task_ids[idx])

                if pending:
                    logger.error(
                        f"Batch {step} exceeded timeout ({timeout}s) in {len(pending)} worker(s); terminating pool"
                    )
                    pool.terminate()
                else:
                    pool.close()
                pool.join()
            # Also drop tools for tasks that timed out inside the worker.
            for idx, result in enumerate(results):
                result_payload = result[0]
                if isinstance(result_payload, dict):
                    err = result_payload.get("error", "")
                    if isinstance(err, str) and "Timeout" in err:
                        timed_out_task_ids.add(task_ids[idx])
            for task_id in timed_out_task_ids:
                private_tool_dir = Path(run_dir) / "private_dynamic_tools" / f"dynamic_tools_{task_id}"
                shutil.rmtree(private_tool_dir, ignore_errors=True)
            state_should_save = False
            try:

                with open(prediction_file, "a", encoding="utf-8") as f:
                    for idx, result in enumerate(results):
                        result = {
                            "question_index": data_items[idx]["task_id"],
                            "question": data_items[idx]["query"],
                            "prediction": result[0],
                        }
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                active_task_ids = [task_id for task_id in task_ids if task_id not in timed_out_task_ids]
                new_tool_call_cnt = sum([result[1] for result in results])
                dynamic_tools_dirs = [f"{run_dir}/private_dynamic_tools/dynamic_tools_{task_id}" for task_id in active_task_ids]
                dynamic_tools_files = [
                    file for dynamic_tools_dir in dynamic_tools_dirs for file in Path(dynamic_tools_dir).glob("*.py")
                ]
                new_needed_tool_cnt = len(dynamic_tools_files)
                egl = compute_loss(new_tool_call_cnt, total_tool_call_cnt, new_needed_tool_cnt, total_needed_tool_cnt)
                total_tool_call_cnt += new_tool_call_cnt
                total_needed_tool_cnt += new_needed_tool_cnt
                state_should_save = True
                logger.info(f"Step: {step}, EGL: {egl}, Total tool call cnt: {total_tool_call_cnt}, Total needed tool cnt: {total_needed_tool_cnt}")

                await optimize_tools(active_task_ids, step, run_dir, merge_policy)
            except Exception as e:
                logger.error(f"Error processing data: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
            finally:
                if state_should_save:
                    try:
                        save_train_state(run_dir, step, total_tool_call_cnt, total_needed_tool_cnt)
                    except Exception as e:
                        logger.warning(f"Failed to save train state for step {step}: {e}")

            total_queries += len(data_items)

            wait_for_all_tracers()

            step += 1
            if step >= train_steps + start:
                break
    finally:
        manager.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument(
        "--dataset", type=str, default="GAIA-valid", help="Dataset name (e.g., 'GAIA-valid', 'GAIA-test')"
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=100000,
        help="Number of training steps (default 100000 to run all data)",
    )
    parser.add_argument("--start", type=int, default=0, help="Start from the n-th step")
    parser.add_argument("--run_name", type=str, default=None, help="Run name")
    parser.add_argument("--merge_policy", type=str, default="naive", help="Merge policy")

    parser.add_argument(
        "--timeout",
        type=int,
        default=5400,
        help="Timeout in seconds for each query execution (default: None, no timeout)",
    )

    args = parser.parse_args()

    data_file = Path(args.dataset).stem
    run_name = (
        args.run_name
        or f"{data_file}_steps{args.train_steps}_bs{args.batch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    run_dir = Path("output") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = run_dir / "predictions.jsonl"

    data_iter = load_dataset(args.dataset, batch_size=args.batch_size)

    asyncio.run(
        train(
            data_iter,
            args.train_steps,
            args.start,
            run_dir,
            predictions_file,
            args.timeout,
            args.merge_policy,
        )
    )
