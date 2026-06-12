#!/usr/bin/env bash
# Upload GRPO assets to RunPod. Usage: ./scripts/upload_grpo_to_runpod.sh root@POD_IP
set -euo pipefail
POD="${1:?Usage: $0 root@POD_IP}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Uploading GRPO scripts to $POD ..."
scp "$ROOT/training/grpo_train.py" "$POD:/workspace/grpo_train.py"
scp -r "$ROOT/solvers" "$POD:/workspace/solvers"
# trl 1.4 is installed from PyPI (see below) — the old trl-0.29.1 wheel is stale and not shipped.
# Required: ground-truth answers are embedded in assistant messages (\\boxed{...})
scp "$ROOT/data/sft_train.jsonl" "$POD:/workspace/data/sft_train.jsonl"
# Optional: only needed to regenerate JSONL on pod (generate_sft_data.py), not for grpo_train.py
if [[ -f "$ROOT/raw-data/train.csv" ]]; then
  scp "$ROOT/raw-data/train.csv" "$POD:/workspace/data/train.csv"
fi

echo "Done. On pod run:"
echo "  pip install 'trl>=1.4,<1.5' datasets   # trl 1.4 matches transformers 5.x; NOT the old 0.29.1 wheel"
echo "  pip show trl transformers              # confirm trl 1.4.x + transformers 5.11 before running"
echo "  export SFT_ADAPTER=/workspace/output OUTPUT_DIR=/workspace/output_grpo DATA_DIR=/workspace/data"
echo "  python -u /workspace/grpo_train.py 2>&1 | tee logs/grpo_\$(date +%Y%m%d_%H%M).log"
