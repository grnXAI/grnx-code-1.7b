from datasets import load_dataset
from tqdm import tqdm
import os

path = "data/raw/grpo/codecontests-plus-1x"
os.makedirs(path, exist_ok=True)

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

for i in tqdm(range(dataset.num_shards)):
    shard = dataset.shard(dataset.num_shards, i)
    shard.to_parquet(
        f"{path}/shard-{i:05d}.parquet"
    )