from pathlib import Path
import sys

sys.path.append("llm-finetune/lit-gpt")
sys.path.append("llm-finetune/lit-gpt/scripts")

import scripts.merge_lora as merge

def perform_merge(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    lora_path: str = "out/lora/alpaca/lit_model_lora_finetuned.pth",
    out_dir: str = "out/lora/checkpoint"
):
    out_dir = Path(out_dir)
    # Make the out_dir, because the merge_lora script doesn't do that...
    out_dir.mkdir(parents=True, exist_ok=True)

    merge.merge_lora(
        checkpoint_dir = Path("llm-finetune/lit-gpt/checkpoints") / model_name,
        lora_path = Path(lora_path),
        precision = "bf16-true",
    )

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(perform_merge)
