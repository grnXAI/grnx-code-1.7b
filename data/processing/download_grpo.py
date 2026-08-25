from datasets import load_dataset, Dataset
from tqdm import tqdm

dataset = load_dataset(
    "ByteDance-Seed/Code-Contests-Plus",
    "1x",
    split="train",
    streaming=True
)

dataset = dataset.remove_columns([
    "correct_submissions",
    "incorrect_submissions",
    "generator",
    "generator_cmd"
])

rows = []

for row in tqdm(dataset, total=11690):
    rows.append(row)

Dataset.from_list(rows).save_to_disk(
    "data/raw/grpo/codecontests-plus-1x"
)