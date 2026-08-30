from transformers import AutoTokenizer
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset, load_dataset
import torch
import yaml
import os

with open("configs/sft.yaml", "r") as f:
    config = yaml.safe_load(f)

hf_datasets = {
    "taco": load_dataset("BAAI/TACO"),
    "apps": load_dataset("codeparrot/apps"),
    "code_contests": load_dataset("deepmind/code_contests"),
    "open-r1/codeforces": load_dataset("open-r1/codeforces")
}

question_cache = {}

def get_question(ds_name, split, index):
    key = (ds_name, split, int(index))

    if key in question_cache:
        return question_cache[key]

    benchmark = hf_datasets[ds_name][split][int(index)]

    if ds_name == "code_contests":
        question = benchmark["description"]

    elif ds_name in ["taco", "apps"]:
        question = benchmark["question"]

    elif ds_name == "open-r1/codeforces":
        question = benchmark["description"]

        if benchmark["input_format"]:
            question += "\n\nInput\n\n" + benchmark["input_format"]

        if benchmark["output_format"]:
            question += "\n\nOutput\n\n" + benchmark["output_format"]

        if benchmark["examples"]:
            question += "\n\nExamples"

            for example in benchmark["examples"]:
                if "input" in example:
                    question += "\n\nInput\n\n" + example["input"]

                if "output" in example:
                    question += "\n\nOutput\n\n" + example["output"]

        if benchmark["note"]:
            question += "\n\nNote\n\n" + benchmark["note"]

    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    question_cache[key] = question

    return question

data = {}
path = config["data"]["raw_dir"]

for idx, name in enumerate(os.listdir(path)):
    full = os.path.join(path, name)

    if name.endswith(".arrow"):
        data[f"shard_{idx}"] = Dataset.from_file(full).to_pandas()

tokenizer = AutoTokenizer.from_pretrained(
    config["model"]["tokenizer"]
)

def tokenize_row(row):
    question = get_question(
        row["dataset"],
        row["split"],
        row["index"]
    )

    messages = [
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",
            "content": row["r1_generation"]
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False
    )

for shard in data:
    data[shard]["tokenized_data"] = data[shard].apply(
        tokenize_row,
        axis=1
    )

    data[shard] = data[shard][["tokenized_data"]]

for shard in data:
    tokens = []

    for sequence in data[shard]["tokenized_data"]:
        tokens.extend(sequence)

    data[shard] = torch.tensor(
        tokens,
        dtype=torch.long
    )

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