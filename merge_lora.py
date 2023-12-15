from pathlib import Path
import sys

sys.path.append("llm-finetune/lit-gpt")
sys.path.append("llm-finetune/lit-gpt/script")

import merge_lora as merge


def merge_lora(
    model_name: str = "mistralai/Mistral-7B-v0.1",
):
    mege.merge_lora(
        checkpoint_dir = Path("llm-finetune/lit-gpt/checkpoints") / model_name,
        precision = "bf16-true",
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(merge_lora)
