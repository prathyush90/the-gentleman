"""QLoRA fine-tune a model on the RitchieBot conversations.

Each experiment gets its own folder: outputs/<name>/run (checkpoints) and outputs/<name>/lora (best adapter).

Usage:
    python train/train.py --name llama8b --model unsloth/Llama-3.1-8B-Instruct --check   # sanity checks only
    python train/train.py --name llama8b --model unsloth/Llama-3.1-8B-Instruct           # full training run
    python train/train.py --name ritchie3b-v2                                            # default model (common.MODEL)
"""
import argparse
import json
import sys
from pathlib import Path

# Unsloth must be imported BEFORE transformers/trl so it can patch them for speed.
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from common import MODEL, to_text

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MAX_LEN = 1024
SEED = 42

# ---- Command-line options: which model, and what to call this experiment
parser = argparse.ArgumentParser(description="QLoRA fine-tune for RitchieBot")
parser.add_argument("--model", default=MODEL, help="Hugging Face model to fine-tune")
parser.add_argument("--name", required=True, help="experiment name; outputs go to outputs/<name>/")
parser.add_argument("--check", action="store_true", help="build + sanity checks only, no training")
cli = parser.parse_args()   # 'cli', not 'args': 'args' is used below for the SFTConfig

EXP_DIR = ROOT / "outputs" / cli.name
RUN_DIR = EXP_DIR / "run"       # checkpoints during training
ADAPTER_DIR = EXP_DIR / "lora"  # final adapter we keep

# Safety guard: never overwrite a finished experiment
if ADAPTER_DIR.exists():
    sys.exit(f"outputs/{cli.name}/lora already exists. Choose a different --name.")
print(f"Experiment '{cli.name}': model={cli.model} -> {EXP_DIR}")

# ---------------------------------------------------------------- 1. Load the base model in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cli.model,
    max_seq_length=MAX_LEN,
    load_in_4bit=True,          # the "Q" in QLoRA
)

# ---------------------------------------------------------------- 2. Attach LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",  # recompute some values instead of storing them: saves VRAM
    random_state=SEED,
)
model.print_trainable_parameters()

# ---------------------------------------------------------------- 3. Load data and turn chats into text
ds = load_dataset("json", data_files={
    "train": str(DATA / "train.jsonl"),
    "eval": str(DATA / "eval.jsonl"),
})
ds = ds.map(
    lambda ex: {"text": to_text(tokenizer, ex["messages"])},
    remove_columns=["messages", "category"],   # the trainer only needs "text"
)

# ---------------------------------------------------------------- 4. Training settings
args = SFTConfig(
    output_dir=str(RUN_DIR),
    dataset_text_field="text",
    max_length=MAX_LEN,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    bf16=True,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=SEED,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
)

# ---------------------------------------------------------------- 5. Only learn from the assistant's replies
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# ---------------------------------------------------------------- 6. Sanity checks (always look before you train!)
example = trainer.train_dataset[0]
ids, labels = example["input_ids"], example["labels"]
bos = tokenizer.bos_token_id
print("\n=== SANITY CHECKS ===")
print("BOS tokens at start:", sum(1 for i in ids[:3] if i == bos), "(should be 1)")
print("Text starts with:", repr(tokenizer.decode(ids[:12])), "(should go straight to the user turn)")
print("Tokens in example:", len(ids), "| tokens the model learns from:",
      sum(1 for l in labels if l != -100))
print("\n--- What the model is TRAINED on (user parts removed) ---")
print(tokenizer.decode([i for i, l in zip(ids, labels) if l != -100]))
print("=====================\n")

if cli.check:
    print("Check mode: stopping before training.")
    sys.exit(0)

# ---------------------------------------------------------------- 7. Train
stats = trainer.train()
print(stats)
print(f"Peak VRAM: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

# ---------------------------------------------------------------- 8. Save the best adapter + the loss history
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
with open(ADAPTER_DIR / "log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)
print(f"Saved adapter to {ADAPTER_DIR}")
