import os, shutil, subprocess

#ADAPTER_PATH = "/kaggle/input/datasets/nicholas33/nemotron-sft-adapter-v1"
ADAPTER_PATH = "/kaggle/input/models/nicholas33/nemotron-sft-adapter-v2/pytorch/nemotron-sft-adapter-run2-valloss0369/1"
OUTPUT_DIR = "/kaggle/working"

# Copy the two adapter files into /kaggle/working
for fname in ["adapter_config.json", "adapter_model.safetensors"]:
    src = os.path.join(ADAPTER_PATH, fname)
    dst = os.path.join(OUTPUT_DIR, fname)
    shutil.copy(src, dst)
    print(f"copied {fname}  ({os.path.getsize(dst):,} bytes)")

# Sanity check the rank
import json
with open(os.path.join(OUTPUT_DIR, "adapter_config.json")) as f:
    cfg = json.load(f)
assert cfg["r"] <= 32, f"LoRA rank {cfg['r']} > 32"
print(f"LoRA rank: {cfg['r']} ✓")

# Zip them up (the -m flag moves into the zip and deletes the originals)
subprocess.run(
    ["zip", "-m", "submission.zip", "adapter_config.json", "adapter_model.safetensors"],
    cwd=OUTPUT_DIR, check=True,
)
print(f"\nSubmission ready: {os.path.join(OUTPUT_DIR, 'submission.zip')}")
print(f"Size: {os.path.getsize(os.path.join(OUTPUT_DIR, 'submission.zip')):,} bytes")