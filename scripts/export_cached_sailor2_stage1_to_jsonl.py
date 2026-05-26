import json
import os
from pathlib import Path
from datasets import load_dataset

DATASET_NAME = "sailor2/sailor2-sft-stage1"
OUTPUT_PATH = Path("../dataset/sailor2_sft_stage1_jsonl/train.jsonl")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("HF_HOME =", os.environ.get("HF_HOME"))
print("HF_DATASETS_CACHE =", os.environ.get("HF_DATASETS_CACHE"))
print("Loading dataset:", DATASET_NAME)

ds = load_dataset(DATASET_NAME)
split = "train" if "train" in ds else list(ds.keys())[0]
data = ds[split]

print("Using split:", split)
print("Columns:", data.column_names)
print("Num rows:", len(data))

def normalize_role(role):
    role = str(role).lower().strip()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return role

def parse_messages(example):
    messages = example.get("messages")

    if isinstance(messages, str):
        messages = json.loads(messages)

    if not isinstance(messages, list):
        return None

    return messages

def convert_example(example):
    messages = parse_messages(example)
    if messages is None:
        return []

    system_parts = []
    rows = []
    current_user = None

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = normalize_role(msg.get("role", ""))
        content = msg.get("content", "")

        if content is None:
            content = ""

        content = str(content).strip()

        if not content:
            continue

        if role == "system":
            system_parts.append(content)

        elif role == "user":
            current_user = content

        elif role == "assistant":
            if current_user is not None:
                rows.append({
                    "system": "\n".join(system_parts).strip(),
                    "user": current_user,
                    "assistant": content,
                })
                current_user = None

    return rows

n_out = 0
n_bad = 0

with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
    for i, example in enumerate(data):
        rows = convert_example(example)

        if not rows:
            n_bad += 1
            continue

        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

        if i > 0 and i % 10000 == 0:
            print(f"Processed {i} source rows, wrote {n_out} SFT rows")

print("Done")
print("Source rows:", len(data))
print("Written SFT rows:", n_out)
print("Skipped/bad rows:", n_bad)
print("Output:", OUTPUT_PATH)