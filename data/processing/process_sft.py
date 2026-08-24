from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import Dataset
import pandas as pd
import torch
import os

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
    data[shard] = data[shard].flatten().long()

class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len=2048):
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

train_dataset = TokenDataset(
    data["shard_1"],
    seq_len=2048
)