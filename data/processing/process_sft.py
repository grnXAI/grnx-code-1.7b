import os
import yaml
import torch

from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


with open("configs/sft.yaml", "r") as f:
    config = yaml.safe_load(f)


tokenizer = AutoTokenizer.from_pretrained(
    config["model"]["tokenizer"]
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


seq_len = config["data"]["sequence_length"]
raw_dir = config["data"]["raw_dir"]
processed_dir = config["data"]["processed_dir"]
write_batch_size = config["data"]["write_batch_size"]

stage_1_name = config["data"]["stage_1"]["name"]
stage_2_name = config["data"]["stage_2"]["name"]

benchmark_datasets = {}


def get_benchmark(name):
    if name not in benchmark_datasets:
        if name == "taco":
            benchmark_datasets[name] = load_dataset(
                "BAAI/TACO"
            )

        elif name == "apps":
            benchmark_datasets[name] = load_dataset(
                "codeparrot/apps"
            )

        elif name == "code_contests":
            benchmark_datasets[name] = load_dataset(
                "deepmind/code_contests"
            )

        elif name == "open-r1/codeforces":
            benchmark_datasets[name] = load_dataset(
                "open-r1/codeforces"
            )

        else:
            raise ValueError(
                f"Unknown reasoning dataset: {name}"
            )

    return benchmark_datasets[name]


def get_question(dataset_name, split, index):
    dataset = get_benchmark(dataset_name)

    row = dataset[split][int(index)]

    if dataset_name == "code_contests":
        return row["description"]

    if dataset_name in ["taco", "apps"]:
        return row["question"]

    if dataset_name == "open-r1/codeforces":
        question = row["description"]

        if row.get("input_format"):
            question += (
                "\n\nInput\n\n"
                + row["input_format"]
            )

        if row.get("output_format"):
            question += (
                "\n\nOutput\n\n"
                + row["output_format"]
            )

        examples = row.get("examples") or []

        if isinstance(examples, dict):
            inputs = examples.get("input", [])
            outputs = examples.get("output", [])

            question += "\n\nExamples"

            for example_input, example_output in zip(
                inputs,
                outputs
            ):
                question += (
                    "\n\nInput\n\n"
                    + str(example_input)
                )

                question += (
                    "\n\nOutput\n\n"
                    + str(example_output)
                )

        elif isinstance(examples, list) and examples:
            question += "\n\nExamples"

            for example in examples:
                if example.get("input"):
                    question += (
                        "\n\nInput\n\n"
                        + example["input"]
                    )

                if example.get("output"):
                    question += (
                        "\n\nOutput\n\n"
                        + example["output"]
                    )

        if row.get("note"):
            question += (
                "\n\nNote\n\n"
                + row["note"]
            )

        return question

    raise ValueError(
        f"Unknown dataset: {dataset_name}"
    )


def encode_example(prompt, response):
    prompt_messages = [
        {
            "role": "user",
            "content": str(prompt)
        }
    ]

    full_messages = [
        {
            "role": "user",
            "content": str(prompt)
        },
        {
            "role": "assistant",
            "content": str(response)
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

    labels = (
        [-100] * len(prompt_ids)
        + input_ids[len(prompt_ids):]
    )

    return input_ids, labels


def save_shard(samples, output_dir, shard_idx):
    if not samples:
        return shard_idx

    torch.save(
        samples,
        os.path.join(
            output_dir,
            f"shard_{shard_idx:05d}.pt"
        )
    )

    return shard_idx + 1


def process_stage(stage_name, reasoning=False):
    input_dir = os.path.join(
        raw_dir,
        stage_name
    )

    output_dir = os.path.join(
        processed_dir,
        stage_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    dataset_dirs = sorted([
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if os.path.isdir(
            os.path.join(input_dir, name)
        )
    ])

    if not dataset_dirs:
        raise ValueError(
            f"No datasets found in {input_dir}"
        )

    input_buffer = []
    label_buffer = []

    samples = []
    shard_idx = 0

    total_rows = 0

    for dataset_dir in dataset_dirs:
        dataset = load_from_disk(
            dataset_dir
        )

        total_rows += len(dataset)

    print(
        f"\nStage: {stage_name} "
        f"({total_rows:,} examples)"
    )

    progress = tqdm(
        total=total_rows,
        desc="Processing"
    )

    for dataset_dir in dataset_dirs:
        dataset = load_from_disk(
            dataset_dir
        )

        for row in dataset:
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
                progress.update(1)
                continue

            input_ids, labels = encode_example(
                prompt,
                response
            )

            if len(input_ids) > seq_len:
                input_ids = input_ids[:seq_len]
                labels = labels[:seq_len]

            input_buffer.extend(
                input_ids
            )

            label_buffer.extend(
                labels
            )

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

                if len(samples) >= write_batch_size:
                    shard_idx = save_shard(
                        samples,
                        output_dir,
                        shard_idx
                    )

                    samples = []

            progress.update(1)

    progress.close()

    if input_buffer:
        padding = (
            seq_len
            - len(input_buffer)
        )

        input_buffer.extend(
            [tokenizer.pad_token_id] * padding
        )

        label_buffer.extend(
            [-100] * padding
        )

        samples.append({
            "input_ids": torch.tensor(
                input_buffer,
                dtype=torch.long
            ),
            "labels": torch.tensor(
                label_buffer,
                dtype=torch.long
            )
        })

    save_shard(
        samples,
        output_dir,
        shard_idx
    )

    print(
        f"Finished: {stage_name}"
    )


process_stage(
    stage_1_name
)

process_stage(
    stage_2_name,
    reasoning=True
)

print("\nSFT processing complete.")