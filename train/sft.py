import os
import glob
import yaml
import torch
import argparse

from torch.utils.data import ConcatDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator
)

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
args = parser.parse_args()

with open("configs/sft.yaml", "r") as f:
    config = yaml.safe_load(f)

if args.stage == 1:
    stage_name = "general-web"
    model_path = config["model"]["name"]
    training = config["training"]["stage_1"]

else:
    stage_name = "code-reasoning"
    model_path = config["training"]["stage_1"]["output_dir"]
    training = config["training"]["stage_2"]

output_path = training["output_dir"]

dataset_path = os.path.join(
    config["data"]["processed_dir"],
    stage_name
)

files = sorted(
    glob.glob(
        os.path.join(dataset_path, "*.pt")
    )
)

if not files:
    raise ValueError(f"No processed shards found in {dataset_path}")

dataset = ConcatDataset([
    torch.load(file, weights_only=False)
    for file in files
])

print(f"\nGRNX Code SFT — Stage {args.stage}: {stage_name}")
print(f"Sequences: {len(dataset):,}")
print("Starting training...\n")

dtype = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32
}[config["model"]["dtype"]]

tokenizer = AutoTokenizer.from_pretrained(
    config["model"]["tokenizer"]
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype
)

if config["model"]["gradient_checkpointing"]:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

training_args = TrainingArguments(
    output_dir=output_path,
    num_train_epochs=training["epochs"],
    learning_rate=training["learning_rate"],
    per_device_train_batch_size=training["batch_size"],
    gradient_accumulation_steps=training["gradient_accumulation_steps"],
    warmup_ratio=training["warmup_ratio"],
    weight_decay=training["weight_decay"],
    lr_scheduler_type=training["lr_scheduler_type"],
    logging_steps=training["logging_steps"],
    save_steps=training["save_steps"],
    save_total_limit=training["save_total_limit"],
    bf16=dtype == torch.bfloat16,
    fp16=dtype == torch.float16,
    gradient_checkpointing=config["model"]["gradient_checkpointing"],
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=training["dataloader_num_workers"],
    optim=training["optimizer"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=default_data_collator
)

trainer.train()

trainer.save_model(output_path)
tokenizer.save_pretrained(output_path)

print(f"\nStage {args.stage} finished.")
print(f"Saved to: {output_path}")