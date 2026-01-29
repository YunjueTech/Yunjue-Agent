import base64
import json
import re
from pathlib import Path
from typing import Iterator
import os


def load_hle_dataset(batch_size: int) -> Iterator[dict]:
    dataset_path = Path(__file__).parent / "dataset" / "HLE" / "hle_500.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data["data"]
    items = []
    for item in data:
        if (question := item.get("question", None)) is None:
            continue
        if image := item.get("image", None):
            images_dir = Path(__file__).parent / "dataset" / "HLE" / "images"
            os.makedirs(images_dir, exist_ok=True)
            matching_files = list(images_dir.glob(f"{item['id']}.*"))
            if matching_files:
                image_path = matching_files[0] 
                question = f"{question} [Image path: {image_path}]"
                items.append((item["id"], question))
                continue
            match = re.match(r"data:(.*);base64,(.*)", image, re.DOTALL)
            if not match:
                print(f"Parse error: {item['id']}")
            mime_type = match.group(1).strip()
            base64_string = match.group(2).strip()
            binary_data = base64.b64decode(base64_string)
            image_path = f"dataset/HLE/images/{item['id']}.{mime_type.split('/')[-1]}"
            image_path = Path(__file__).parent / image_path
            with open(image_path, "wb") as f:
                f.write(binary_data)
            question = f"{question} [Image path: {image_path}]"
        items.append((item["id"], question))

    for i in range(0, len(items), batch_size):
        batch_items = items[i : i + batch_size]
        data_items = [{"task_id": task_id, "query": question} for task_id, question in batch_items]
        yield {"data_items": data_items}


def load_xbench_dataset(batch_size: int, type: str = "deepsearch") -> Iterator[dict]:
    dataset_paths = []
    if type == "deepsearch":
        dataset_paths.append(Path(__file__).parent / "dataset" / "XBENCH" / "DeepSearch-2510.json")
    elif type == "scienceqa":
        dataset_paths.append(Path(__file__).parent / "dataset" / "XBENCH" / "ScienceQA.json")
    elif type == "all":
        dataset_paths.append(Path(__file__).parent / "dataset" / "XBENCH" / "DeepSearch-2510.json")
        dataset_paths.append(Path(__file__).parent / "dataset" / "XBENCH" / "ScienceQA.json")
    else:
        raise ValueError(f"Unknown XBENCH dataset type: {type}")
    data = []
    for dataset_path in dataset_paths:
        with open(dataset_path, "r", encoding="utf-8") as f:
            temp_data = json.load(f)
        data.extend(temp_data)
    for i in range(0, len(data), batch_size):
        batch_items = data[i : i + batch_size]
        data_items = [{"task_id": item["task_id"], "query": item["task_question"]} for item in batch_items]
        yield {"data_items": data_items}


def load_deepsearchqa_dataset(batch_size: int) -> Iterator[dict]:
    dataset_path = Path(__file__).parent / "dataset" / "DEEPSEARCHQA" / "DSQA-full.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i in range(0, len(data), batch_size):
        batch_items = data[i : i + batch_size]
        data_items = [{"task_id": item["example_id"], "query": item["problem"]} for item in batch_items]
        yield {"data_items": data_items}


def load_finsearchcomp_dataset(batch_size: int) -> Iterator[dict]:
    dataset_path = Path(__file__).parent / "dataset" / "FinSearchComp" / "t2_t3_questions.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i in range(0, len(data), batch_size):
        batch_items = data[i : i + batch_size]
        data_items = [{"task_id": item["prompt_id"], "query": item["prompt"]} for item in batch_items]
        yield {"data_items": data_items}


def load_dataset(dataset: str, batch_size: int) -> Iterator[dict]:
    """
    Load dataset based on the dataset name.

    Args:
        dataset: Dataset name ('GAIA-valid', 'GAIA-test', 'HLE', etc.)
        batch_size: Number of queries per batch

    Returns:
        Iterator that yields dictionaries with 'data_items' key, where each data_item
        contains 'task_id' and 'query' fields
    """
    if dataset == "HLE":
        return load_hle_dataset(batch_size)
    elif dataset == "XBENCH-deepsearch":
        return load_xbench_dataset(batch_size, type="deepsearch")
    elif dataset == "XBENCH-scienceqa":
        return load_xbench_dataset(batch_size, type="scienceqa")
    elif dataset == "XBENCH-all":
        return load_xbench_dataset(batch_size, type="all")
    elif dataset == "DEEPSEARCHQA":
        return load_deepsearchqa_dataset(batch_size)
    elif dataset == "FINSEARCHCOMP":
        return load_finsearchcomp_dataset(batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
