#!/usr/bin/env bash
# Upload GRPO assets to RunPod. Usage: ./scripts/upload_grpo_to_runpod.sh root@POD_IP
set -euo pipefail
POD="${1:?Usage: $0 root@POD_IP}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Uploading GRPO scripts to $POD ..."
scp "$ROOT/training/grpo_train.py" "$POD:/workspace/grpo_train.py"
scp -r "$ROOT/solvers" "$POD:/workspace/solvers"
scp "$ROOT/trl_wheels/trl-0.29.1-py3-none-any.whl" "$POD:/workspace/trl_wheels/trl-0.29.1-py3-none-any.whl"
scp "$ROOT/data/sft_train.jsonl" "$POD:/workspace/data/sft_train.jsonl"

echo "Done. On pod run:"
echo "  pip install /workspace/trl_wheels/trl-0.29.1-py3-none-any.whl datasets"
echo "  export SFT_ADAPTER=/workspace/output OUTPUT_DIR=/workspace/output_grpo DATA_DIR=/workspace/data"
echo "  python -u /workspace/grpo_train.py 2>&1 | tee logs/grpo_\$(date +%Y%m%d_%H%M).log"
