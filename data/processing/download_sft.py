from datasets import load_dataset, Dataset
from tqdm import tqdm

dataset = load_dataset(
    "nvidia/OpenCodeReasoning-2",
    "train",
    split="python",
    streaming=True
)

samples = []
bar = tqdm(total=50000)
for row in dataset:
    if float(row["pass_rate"]) == 1.0:
        samples.append(row)
        bar.update(1)

    if len(samples) >= 50000:
        break

bar.close()
Dataset.from_list(samples).save_to_disk(
    "data/raw/sft/opencode-reasoning-50k"
)