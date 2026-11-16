"""ask the same fresh questions to the BASE model and the FINE-TUNED model, side by side.

Usage:
    python train/eval_probes.py llama8b                              # published adapter (auto-downloaded if missing)
    python train/eval_probes.py ritchie-3b                           # published adapter (auto-downloaded if missing)
    python train/eval_probes.py outputs/llama8b/run/checkpoint-60    # any local adapter folder
Reports are written to outputs/reports/<experiment>_<checkpoint>.md
"""
import re
import sys
from pathlib import Path

from unsloth import FastLanguageModel
import torch

from common import ROOT, resolve_adapter, to_text

ADAPTER = resolve_adapter(sys.argv[1] if len(sys.argv) > 1 else "llama8b")

# Name the report after the experiment AND the checkpoint, so different models never overwrite each other:
#   outputs/llama8b/run/checkpoint-120        -> outputs/reports/llama8b_checkpoint-120.md
#   outputs/ritchie-3b-run/checkpoint-120     -> outputs/reports/ritchie-3b_checkpoint-120.md
#   outputs/llama8b/lora                      -> outputs/reports/llama8b_lora.md
EXP_NAME = ADAPTER.parent.parent.name if ADAPTER.parent.name == "run" else ADAPTER.parent.name
EXP_NAME = EXP_NAME.removesuffix("-run")
REPORT = ROOT / "outputs" / "reports" / f"{EXP_NAME}_{ADAPTER.name}.md"
REPORT.parent.mkdir(parents=True, exist_ok=True)

# Fresh prompts NOT in the dataset. Each probe is a list of messages; the last one must be from the user.
# Mix: everyday chat, real help (is it still correct?), identity, emotion, provocation, group chat.
PROBES = [
    [{"role": "user", "content": "my boss just moved my deadline up by a week. cool cool cool"}],
    [{"role": "user", "content": "how do I boil an egg so the yolk is still runny?"}],
    [{"role": "user", "content": "are you a bot?"}],
    [{"role": "user", "content": "who are you exactly"}],
    [{"role": "user", "content": "I think my flatmate is stealing my oat milk"}],
    [{"role": "user", "content": "what's the capital of Australia"}],
    [{"role": "user", "content": "can you explain what an API is"}],
    [{"role": "user", "content": "you're honestly not that funny"}],
    [{"role": "user", "content": "I failed my driving test for the third time"}],
    [{"role": "user", "content": "give me a good excuse for missing my mate's birthday"}],
    [{"role": "user", "content": "should I learn python or javascript first"}],
    [{"role": "user", "content": "tell me about the worst job you ever had"}],
    [{"role": "user", "content": "my cat knocked my coffee onto my laptop"}],
    [{"role": "user", "content": "Sam: pizza tonight?\nJo: we had pizza yesterday\nSam: and?\nJo: AND it's my turn to pick"}],
    [{"role": "user", "content": "I've been feeling really low lately and I don't know who to talk to"}],
    [{"role": "user", "content": "rate my plan: quit my job and become a professional poker player"}],
    [
        {"role": "user", "content": "I'm thinking of getting a dog"},
        {"role": "assistant", "content": "A dog. Big decision. What breed, and how big is your flat?"},
        {"role": "user", "content": "a husky. and it's a studio"},
    ],
    [{"role": "user", "content": "can you just talk normally for once"}],
    [{"role": "user", "content": "what's 17 times 23"}],
    [{"role": "user", "content": "hi"}],
]

FLAG_WORDS = re.compile(r"ritchie|snatch|lock,? stock|rocknrolla|guy rich", re.I)


def generate(model, tokenizer, messages, seed):
    text = to_text(tokenizer, messages, add_generation_prompt=True)   # ends with the empty assistant header
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    torch.manual_seed(seed)                                            # same seed -> reproducible answers
    out = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]                 # drop the prompt, keep only the reply
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    # Loading the adapter folder: Unsloth reads adapter_config.json, loads the base model, then attaches the adapter.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER), max_seq_length=1024, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)   # switches Unsloth into its faster generation mode

    lines = [f"# Probe report — adapter: `{ADAPTER}`\n"]
    for i, messages in enumerate(PROBES, 1):
        with model.disable_adapter():                    # adapter OFF = original Llama
            before = generate(model, tokenizer, messages, seed=i)
        after = generate(model, tokenizer, messages, seed=i)   # adapter ON = RitchieBot

        convo = "\n".join(f"> **{m['role']}:** {m['content']}" for m in messages).replace("\n", "\n> ")
        flag = " ⚠️ **FLAG: mentions source material**" if FLAG_WORDS.search(after) else ""
        lines += [f"## {i}.\n", convo, "\n### Before (base)\n", before,
                  f"\n### After (fine-tuned){flag}\n", after, "\n---\n"]
        REPORT.write_text("\n".join(lines), encoding="utf-8")   # save after every probe: watch it fill in live
        print(f"[{i}/{len(PROBES)}] done")

    print(f"\nReport written to {REPORT}")


if __name__ == "__main__":
    main()
