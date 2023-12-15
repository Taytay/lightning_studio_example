#!/bin/env bash

MODEL=mistralai/Mistral-7B-v0.1
CHECKPOINT_PATH=$1

if [ ! -d lit-gpt ]; then
  git clone https://github.com/Lightning-AI/lit-gpt.git
fi

pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/test/cu118 --pre 'torch==2.1.0'
pip install huggingface_hub tokenizers sentencepiece safetensors
pip install scipy bitsandbytes

# if MODEL is absolute path, then load that
if [ $# -eq 1 ]; then
  pushd lit-gpt
  python ../collect.py --repo_id checkpoints/$MODEL --checkpoint_path $CHECKPOINT_PATH
  python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MODEL
  popd
else
  pushd lit-gpt
  python scripts/download.py --repo_id $MODEL
  python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MODEL
fi
