from datasets import load_dataset
import os

os.makedirs(
    "data/raw/grpo/codecontests-plus-1x",
    exist_ok=True
)

dataset = load_dataset(
    "ByteDance-Seed/Code-Contests-Plus",
    "1x",
    split="train",
    streaming=True
)

dataset = dataset.select_columns([
    "source",
    "id",
    "title",
    "description",
    "time_limit",
    "memory_limit",
    "validator",
    "checker",
    "test_cases"
])

dataset.to_parquet(
    "data/raw/grpo/codecontests-plus-1x/data.parquet"
)