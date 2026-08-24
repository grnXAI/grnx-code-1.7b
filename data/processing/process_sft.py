from transformers import AutoTokenizer
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
import pandas as pd
import torch
import yaml
import os

with open("sft.yaml", "r") as f:
    config = yaml.safe_load(f)

data = {}
path = "data/raw/sft/opencode-reasoning-50k"
for idx, name in enumerate(os.listdir(path)):
    full = os.path.join(path, name)

    if name.endswith(".arrow"):
        data[f"shard_{idx}"] = Dataset.from_file(full).to_pandas()

tokenizer = AutoTokenizer.from_pretrained("tokenizer/qwen")
for shard in data:
    tokenized = data[shard]["r1_generation"].apply(tokenizer.encode)
    data[shard]["tokenized_data"] = tokenized
    data[shard] = data[shard][["tokenized_data"]]

for shard in data:
    tokens = []

    for sequence in data[shard]["tokenized_data"]:
        tokens.extend(sequence)

    data[shard] = torch.tensor(tokens, dtype=torch.long)

class TokenDataset(TorchDataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len

        input_ids = self.tokens[start:end]
        labels = self.tokens[start + 1:end + 1]

        return {
            "input_ids": input_ids,
            "labels": labels
        }

seq_len = config["data"]["sequence_length"]
save_path = config["data"]["processed_dir"]

os.makedirs(save_path, exist_ok=True)

for shard in data:
    data[shard] = TokenDataset(
        data[shard],
        seq_len=seq_len
    )

    torch.save(
        data[shard],
        os.path.join(save_path, f"{shard}.pt")
    )