#!/usr/bin/env python3
"""Download the diffusers-format weights of a Hugging Face model, preferring fp16 to save space."""

import sys

from huggingface_hub import HfApi, snapshot_download

if len(sys.argv) != 3:
	sys.exit(f'Usage: {sys.argv[0]} <model-id> <target-dir>')

model_id, target_dir = sys.argv[1], sys.argv[2]

files = HfApi().list_repo_files(model_id)
has_fp16 = any(f.endswith('.fp16.safetensors') for f in files)
weights = '*/*.fp16.safetensors' if has_fp16 else '*/*.safetensors'

# component subfolders only, skipping single-file checkpoints and ONNX exports in the repo root
snapshot_download(
	model_id,
	local_dir=target_dir,
	allow_patterns=['model_index.json', '*/*.json', '*/*.txt', '*/*.model', weights],
)
print(f'Downloaded {model_id} ({"fp16" if has_fp16 else "full precision"}) to {target_dir}')
