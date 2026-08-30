import os
import pandas as pd
import yaml
from tqdm import tqdm

with open("configs/grpo.yaml", "r") as f:
    config = yaml.safe_load(f)

raw_path = config["data"]["raw_dir"]
save_path = config["data"]["processed_dir"]

os.makedirs(save_path, exist_ok=True)

files = sorted([
    file
    for file in os.listdir(raw_path)
    if file.endswith(".parquet")
])

for idx, file in tqdm(
    enumerate(files),
    total=len(files)
):
    shard = pd.read_parquet(
        os.path.join(raw_path, file)
    )

    shard["prompt"] = (
        shard["title"].fillna("")
        + "\n\n"
        + shard["description"].fillna("")
    )

    shard = shard[
        [
            "id",
            "source",
            "prompt",
            "test_cases",
            "checker",
            "time_limit",
            "memory_limit"
        ]
    ]

    shard.to_parquet(
        os.path.join(
            save_path,
            f"shard-{idx:05d}.parquet"
        ),
        index=False
    )

    del shard