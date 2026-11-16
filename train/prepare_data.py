"""Load all conversations, split into train/eval, and check token lengths."""
import json
import random
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer
from common import MODEL, to_text


# ---- Settings ----
ROOT = Path(__file__).resolve().parent.parent      # the RitchieBot folder
SRC = ROOT / "data" / "conversations"             # where the .jsonl files live
OUT = ROOT / "data" / "processed"                 # where train/eval files will go
# MODEL = "unsloth/Llama-3.2-3B-Instruct"           # base model (only the tokenizer is downloaded here)
EVAL_FRACTION = 0.05                              # 5% held out
SEED = 42                                         # fixed seed = same split every run


def load_conversations():
    """Read every line of every .jsonl file into one list."""
    convs = []
    for path in sorted(SRC.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    convs.append(json.loads(line))
    return convs


def stratified_split(convs):
    """Take 5% from EACH category for eval, so every category is represented."""
    by_cat = defaultdict(list)
    for c in convs:
        by_cat[c["category"]].append(c)

    rng = random.Random(SEED)
    train, evals = [], []
    for cat, items in sorted(by_cat.items()):
        rng.shuffle(items)
        n_eval = max(1, round(len(items) * EVAL_FRACTION))
        evals += items[:n_eval]
        train += items[n_eval:]
        print(f"{cat:35} total={len(items):4}  train={len(items) - n_eval:4}  eval={n_eval}")

    rng.shuffle(train)  # mix categories so training doesn't see 100 banters in a row
    return train, evals


def save(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def token_stats(convs):
    """Show how long conversations are in tokens, and what the model actually sees."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    lengths = []
    for c in convs:
        text = to_text(tok, c["messages"])
        lengths.append(len(tok(text, add_special_tokens=False)["input_ids"]))

    lengths.sort()
    print(f"\nTokens per conversation: min={lengths[0]}  median={lengths[len(lengths) // 2]}  "
          f"p95={lengths[int(len(lengths) * 0.95)]}  max={lengths[-1]}")

    print("\n--- One conversation, exactly as the model will see it ---")
    print(to_text(tok, convs[0]["messages"]))



if __name__ == "__main__":
    convs = load_conversations()
    print(f"Loaded {len(convs)} conversations\n")

    train, evals = stratified_split(convs)
    save(train, OUT / "train.jsonl")
    save(evals, OUT / "eval.jsonl")
    print(f"\nSaved {len(train)} train / {len(evals)} eval to {OUT}")

    token_stats(convs)
