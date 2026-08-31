from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pyarrow.parquet as pq
import torch
import yaml
import os

with open("configs/sft.yaml", "r") as f:
    config = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained(
    config["model"]["tokenizer"]
)

seq_len = config["data"]["sequence_length"]
raw_dir = config["data"]["raw_dir"]
processed_dir = config["data"]["processed_dir"]
stage_1_name = config["data"]["stage_1"]["name"]
stage_2_name = config["data"]["stage_2"]["name"]

benchmark_datasets = None


def load_benchmarks():
    global benchmark_datasets

    if benchmark_datasets is None:
        benchmark_datasets = {
            "taco": load_dataset("BAAI/TACO"),
            "apps": load_dataset("codeparrot/apps"),
            "code_contests": load_dataset("deepmind/code_contests"),
            "open-r1/codeforces": load_dataset("open-r1/codeforces")
        }

    return benchmark_datasets


def get_question(ds_name, split, index):
    datasets = load_benchmarks()
    row = datasets[ds_name][split][int(index)]

    if ds_name == "code_contests":
        return row["description"]

    if ds_name in ["taco", "apps"]:
        return row["question"]

    if ds_name == "open-r1/codeforces":
        question = row["description"]

        if row.get("input_format"):
            question += "\n\nInput\n\n" + row["input_format"]

        if row.get("output_format"):
            question += "\n\nOutput\n\n" + row["output_format"]

        examples = row.get("examples") or []

        if examples:
            question += "\n\nExamples"

            for example in examples:
                if example.get("input"):
                    question += "\n\nInput\n\n" + example["input"]

                if example.get("output"):
                    question += "\n\nOutput\n\n" + example["output"]

        if row.get("note"):
            question += "\n\nNote\n\n" + row["note"]

        return question

    raise ValueError(f"Unknown dataset: {ds_name}")


def encode_example(prompt, response):
    prompt_messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    full_messages = [
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": response
        }
    ]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True
    )

    input_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=True
    )

    labels = [-100] * len(prompt_ids)
    labels.extend(input_ids[len(prompt_ids):])

    return input_ids, labels


def save_chunks(input_buffer, label_buffer, output_dir, shard_idx):
    samples = []

    while len(input_buffer) >= seq_len:
        samples.append({
            "input_ids": torch.tensor(
                input_buffer[:seq_len],
                dtype=torch.long
            ),
            "labels": torch.tensor(
                label_buffer[:seq_len],
                dtype=torch.long
            )
        })

        del input_buffer[:seq_len]
        del label_buffer[:seq_len]

    if samples:
        torch.save(
            samples,
            os.path.join(output_dir, f"shard_{shard_idx:05d}.pt")
        )
        shard_idx += 1

    return shard_idx


def process_stage(stage_name, reasoning=False):
    input_dir = os.path.join(raw_dir, stage_name)
    output_dir = os.path.join(processed_dir, stage_name)

    os.makedirs(output_dir, exist_ok=True)

    files = sorted([
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.endswith(".parquet")
    ])

    input_buffer = []
    label_buffer = []
    output_shard = 0

    for file in tqdm(files, desc=f"Processing {stage_name}"):
        parquet = pq.ParquetFile(file)

        for batch in parquet.iter_batches(batch_size=256):
            rows = batch.to_pylist()

            for row in rows:
                if reasoning:
                    prompt = get_question(
                        row["dataset"],
                        row["split"],
                        row["index"]
                    )
                else:
                    prompt = row["prompt"]

                response = row["response"]

                if not prompt or not response:
                    continue

                input_ids, labels = encode_example(
                    prompt,
                    response
                )

                if len(input_ids) > seq_len:
                    input_ids = input_ids[:seq_len]
                    labels = labels[:seq_len]

                input_buffer.extend(input_ids)
                label_buffer.extend(labels)

            output_shard = save_chunks(
                input_buffer,
                label_buffer,
                output_dir,
                output_shard
            )

    if input_buffer:
        padding = seq_len - len(input_buffer)
        pad_id = tokenizer.pad_token_id

        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        input_buffer.extend([pad_id] * padding)
        label_buffer.extend([-100] * padding)

        torch.save(
            [{
                "input_ids": torch.tensor(input_buffer, dtype=torch.long),
                "labels": torch.tensor(label_buffer, dtype=torch.long)
            }],
            os.path.join(output_dir, f"shard_{output_shard:05d}.pt")
        )


process_stage(stage_1_name)
process_stage(stage_2_name, reasoning=True)
