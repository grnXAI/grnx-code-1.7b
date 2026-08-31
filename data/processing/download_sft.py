import os
import yaml

from datasets import load_dataset, Dataset
from tqdm import tqdm


with open("configs/sft.yaml", "r") as f:
    config = yaml.safe_load(f)


raw_dir = config["data"]["raw_dir"]

stage_1_dir = os.path.join(
    raw_dir,
    config["data"]["stage_1"]["name"]
)

stage_2_dir = os.path.join(
    raw_dir,
    config["data"]["stage_2"]["name"]
)

os.makedirs(stage_1_dir, exist_ok=True)
os.makedirs(stage_2_dir, exist_ok=True)


def limited_generator(dataset, total, transform, filter_fn=None):
    count = 0
    bar = tqdm(total=total)

    for row in dataset:
        if filter_fn is not None and not filter_fn(row):
            continue

        sample = transform(row)

        if sample is None:
            continue

        yield sample

        count += 1
        bar.update(1)

        if count >= total:
            break

    bar.close()


def download_opencode_instruct():
    total = config["data"]["stage_1"]["opencode_instruct_samples"]
    min_score = config["data"]["stage_1"]["min_opencode_test_score"]

    dataset = load_dataset(
        "nvidia/OpenCodeInstruct",
        split="train",
        streaming=True
    )

    def transform(row):
        return {
            "prompt": row["input"],
            "response": row["output"],
            "source": "opencode-instruct"
        }

    def filter_fn(row):
        score = row.get("average_test_score")

        if score is None:
            return False

        return float(score) >= min_score

    dataset = Dataset.from_generator(
        lambda: limited_generator(
            dataset,
            total,
            transform,
            filter_fn
        )
    )

    dataset.save_to_disk(
        os.path.join(
            stage_1_dir,
            "opencode-instruct"
        )
    )


def download_maple():
    total = config["data"]["stage_1"]["maple_samples"]

    dataset = load_dataset(
        "tudor-iustin22/maple",
        split="train",
        streaming=True
    )

    def transform(row):
        return {
            "prompt": row["input"],
            "response": row["output"],
            "source": "maple"
        }

    dataset = Dataset.from_generator(
        lambda: limited_generator(
            dataset,
            total,
            transform
        )
    )

    dataset.save_to_disk(
        os.path.join(
            stage_1_dir,
            "maple"
        )
    )


def download_vulcan():
    total = config["data"]["stage_1"]["vulcan_samples"]

    dataset = load_dataset(
        "xlelords/vulcan",
        split="train",
        streaming=True
    )

    def transform(row):
        messages = row["messages"]

        prompt = None
        response = None

        for message in messages:
            if message["role"] == "user":
                prompt = message["content"]

            elif message["role"] == "assistant":
                response = message["content"]

        if prompt is None or response is None:
            return None

        return {
            "prompt": prompt,
            "response": response,
            "source": "vulcan"
        }

    dataset = Dataset.from_generator(
        lambda: limited_generator(
            dataset,
            total,
            transform
        )
    )

    dataset.save_to_disk(
        os.path.join(
            stage_1_dir,
            "vulcan"
        )
    )


def download_nextjs():
    total = config["data"]["stage_1"]["nextjs_samples"]

    dataset = load_dataset(
        "Slava32/next.js-15.4-with-reasoning",
        split="train",
        streaming=True
    )

    def transform(row):
        response = row["response"]
        reasoning = row.get("reasoning")

        if reasoning:
            response = (
                "<think>\n"
                + reasoning
                + "\n</think>\n\n"
                + response
            )

        return {
            "prompt": row["question"],
            "response": response,
            "source": "nextjs"
        }

    dataset = Dataset.from_generator(
        lambda: limited_generator(
            dataset,
            total,
            transform
        )
    )

    dataset.save_to_disk(
        os.path.join(
            stage_1_dir,
            "nextjs"
        )
    )


def download_opencode_reasoning():
    total = config["data"]["stage_2"]["opencode_reasoning_samples"]
    min_pass_rate = config["data"]["stage_2"]["min_pass_rate"]

    dataset = load_dataset(
        "nvidia/OpenCodeReasoning-2",
        "train",
        split="python",
        streaming=True
    )

    def transform(row):
        return {
            "dataset": row["dataset"],
            "split": row["split"],
            "index": row["index"],
            "response": row["r1_generation"],
            "source": "opencode-reasoning"
        }

    def filter_fn(row):
        return float(row["pass_rate"]) >= min_pass_rate

    dataset = Dataset.from_generator(
        lambda: limited_generator(
            dataset,
            total,
            transform,
            filter_fn
        )
    )

    dataset.save_to_disk(
        os.path.join(
            stage_2_dir,
            "opencode-reasoning"
        )
    )


print("\nStage 1 — General + Web")
download_opencode_instruct()
download_maple()
download_vulcan()
download_nextjs()

print("\nStage 2 — Code Reasoning")
download_opencode_reasoning()

print("\nSFT download complete.")