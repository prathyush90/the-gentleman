"""Shared settings and helpers used by every script in train/."""
import re
import urllib.request
import zipfile
from pathlib import Path

MODEL = "unsloth/Llama-3.2-3B-Instruct"
ROOT = Path(__file__).resolve().parent.parent

# Published adapters: if the local folder is missing (e.g. a fresh clone), the zip is pulled from the
# GitHub release and unpacked into that folder. If the folder is already there, it is used as-is.
RELEASE_URL = "https://github.com/prathyush90/the-gentleman/releases/download/v1.0"
ADAPTERS = {
    "ritchie-3b": ROOT / "outputs" / "ritchie-3b-run" / "checkpoint-120",  # Llama 3.2 3B, epoch 2
    "llama8b": ROOT / "outputs" / "llama8b" / "run" / "checkpoint-120",    # Llama 3.1 8B, epoch 2
}


def resolve_adapter(name_or_path):
    """Return a local adapter folder. Accepts a published name (see ADAPTERS) or any folder path."""
    if name_or_path not in ADAPTERS:
        return Path(name_or_path).resolve()

    local = ADAPTERS[name_or_path]
    if (local / "adapter_model.safetensors").exists():
        return local

    url = f"{RELEASE_URL}/{name_or_path}.zip"
    zip_path = local.parent / f"{name_or_path}.zip"
    local.mkdir(parents=True, exist_ok=True)
    print(f"Adapter '{name_or_path}' not found locally, downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(local)
    zip_path.unlink()
    print(f"Adapter ready at {local}")
    return local

# Llama 3.x's template silently adds a system block ("Cutting Knowledge Date... Today Date...").
# We remove it so training text matches inference, where we send no system prompt.
_SYSTEM_BLOCK = re.compile(
    r"<\|start_header_id\|>system<\|end_header_id\|>.*?<\|eot_id\|>",
    flags=re.DOTALL,   # DOTALL lets '.' match newlines, since the block spans several lines
)


def to_text(tokenizer, messages, add_generation_prompt=False):
    """Turn a list of {role, content} messages into the exact string the model sees.

    add_generation_prompt=True appends an empty '<|start_header_id|>assistant<|end_header_id|>'
    header at the end. We use that at chat time to tell the model "your turn now".
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return _SYSTEM_BLOCK.sub("", text, count=1)
