from datasets import load_dataset, Dataset

dataset = load_dataset(
    "nvidia/OpenCodeReasoning-2",
    "train",
    split="python",
    streaming=True
)

dataset = dataset.shuffle(seed=42, buffer_size=10000)
samples = []

for row in dataset:
    if float(row["pass_rate"]) == 1.0:
        samples.append(row)

    if len(samples) == 50000:
        break

Dataset.from_list(samples).save_to_disk(
    "data/raw/sft/opencode-reasoning-50k"
)