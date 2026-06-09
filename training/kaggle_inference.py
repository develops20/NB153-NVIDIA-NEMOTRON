import os, shutil, subprocess

# Set ADAPTER_PATH to your Kaggle Models input after uploading the GRPO adapter.
# Run2 SFT (0.74 LB):  .../nemotron-sft-adapter-run2-valloss0369/1
# GRPO v1 (target):    .../nemotron-sft-adapter-grpo-v1/1  (update after RunPod GRPO)
ADAPTER_PATH = os.environ.get(
    "ADAPTER_PATH",
    "/kaggle/input/models/nicholas33/nemotron-sft-adapter-v2/pytorch/nemotron-sft-adapter-run2-valloss0369/1",
)
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