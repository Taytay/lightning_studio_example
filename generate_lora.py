
from pathlib import Path
import sys

sys.path.append("llm-finetune/lit-gpt")
sys.path.append("llm-finetune/lit-gpt/generate")

import lora


def generate(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    lora_path = "out/lora/alpaca/lit_model_lora_finetuned.pth",
    prompt: str = ""
):
    lora.main(
        prompt = prompt,
        checkpoint_dir = Path("llm-finetune/lit-gpt/checkpoints") / model_name,
        lora_path = Path(lora_path),
        precision = "bf16-true",
        quantize = "bnb.nf4",
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(generate)
