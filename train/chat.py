"""Chat with the bot in your terminal.

Usage:
    python train/chat.py                    # Llama 3.1 8B adapter (auto-downloaded if missing)
    python train/chat.py ritchie-3b         # Llama 3.2 3B adapter
    python train/chat.py outputs/my-run/lora

Commands inside the chat:
    /reset   forget the conversation so far
    /base    toggle the adapter off/on (compare against the original Llama)
    /quit    exit (Ctrl+D or Ctrl+C also work)
"""
import sys

from unsloth import FastLanguageModel
from transformers import TextStreamer
from transformers.utils import logging as hf_logging

from common import resolve_adapter, to_text

MAX_HISTORY_TOKENS = 768   # keep prompt + reply inside the 1024-token window we trained with


def main():
    adapter = resolve_adapter(sys.argv[1] if len(sys.argv) > 1 else "llama8b")
    print(f"Loading {adapter} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter), max_seq_length=1024, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    model.generation_config.max_length = None   # we use max_new_tokens; this stops a warning on every reply
    hf_logging.set_verbosity_error()             # hide library chatter so the chat stays readable
    # Prints tokens as they are generated, so replies appear word by word instead of all at once.
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    history, use_base = [], False
    print("\nReady. Type a message (/reset, /base, /quit).\n")
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history = []
            print("(conversation cleared)\n")
            continue
        if user == "/base":
            use_base = not use_base
            print(f"(now talking to the {'BASE model' if use_base else 'fine-tuned model'})\n")
            continue

        history.append({"role": "user", "content": user})
        # Drop the oldest exchanges if the conversation outgrows the context window.
        while len(history) > 1 and len(tokenizer(to_text(tokenizer, history))["input_ids"]) > MAX_HISTORY_TOKENS:
            history = history[2:]

        text = to_text(tokenizer, history, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        print("bot > ", end="", flush=True)
        kwargs = dict(**inputs, streamer=streamer, max_new_tokens=300, do_sample=True,
                      temperature=0.8, top_p=0.9, repetition_penalty=1.1)
        if use_base:
            with model.disable_adapter():
                out = model.generate(**kwargs)
        else:
            out = model.generate(**kwargs)
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
