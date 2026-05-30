# RunPod SFT plan — Nemotron-3 Nano 30B LoRA

Follow this **before** starting the GPU. Goal: configure once, then run ~17h training without wasting GPU hours.

Prior successful run: **RTX PRO 6000 Blackwell 96 GB**, val_loss **0.0388**, ~**17.5h**, peak VRAM **~94/96 GB**.

---

## Checklist (tick as you go)

### A. RunPod pod settings

- [ ] **GPU:** NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) — same class as last run
- [ ] **Template / image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (your previous choice)
- [ ] **Network volume** mounted at `/workspace` (persistent — survives pod stop)
- [ ] **Container disk:** **50 GB** (enough for OS + pip env if model lives on volume)
- [ ] **Volume disk:** **150 GB** minimum (**200 GB** safer)
- [ ] **Expose:** SSH + Jupyter (optional)
- [ ] **Start pod** only after volume is attached

### B. Upload from Mac (before or right after pod start)

- [ ] `data/sft_train.jsonl` → `/workspace/data/` (**May 30 file — NOT `sft-0.74/`**)
- [ ] `data/sft_val.jsonl` → `/workspace/data/`
- [ ] Repo clone **or** copy `training/train.py` + entire `solvers/` folder

### C. One-time setup on pod (no GPU training yet)

- [ ] System packages + Python deps (section 3)
- [ ] Download base model via `kagglehub` (section 4)
- [ ] Verify imports + GPU (section 5)
- [ ] Pre-flight directory layout (section 2)

### D. Start training

- [ ] Set env vars (section 6)
- [ ] Launch in `tmux` (section 7)
- [ ] Monitor first 30 min (section 8)

### E. After training

- [ ] Download `/workspace/output/submission.zip` + logs
- [ ] Stop pod (volume keeps checkpoints if on network volume)

---

## 1. RunPod specs (recommended)

| Setting | Value | Why |
|---------|-------|-----|
| **GPU** | RTX PRO 6000 Blackwell **96 GB** | Last run peaked ~94 GB VRAM at batch=2, seq=2048 |
| **Container image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Your prior image; includes CUDA devel for custom wheels |
| **Container disk** | **50 GB** | Pip env + temp; keep heavy files on network volume |
| **Volume disk** | **150–200 GB** | Model ~60 GB + checkpoints ~10 GB + adapter ~4 GB + token cache ~10–20 GB + headroom |
| **Volume mount** | `/workspace` | Matches all defaults in `train.py` |

### Volume size breakdown

| Item | ~Size |
|------|-------|
| Base model (BF16, kagglehub cache) | 55–65 GB |
| LoRA adapter + `submission.zip` | ~3.5 GB |
| Step checkpoints (`KEEP_LAST_N=2` + `best/`) | ~7–12 GB |
| SFT JSONL (train + val) | ~61 MB |
| Optional `TOKEN_CACHE` (.pt files) | 5–20 GB |
| Logs | < 100 MB |

**Do not** use `data/sft-0.74/` — that old set has **961 fallback** templates.

### Image note (Blackwell)

Last successful run reported **torch 2.12.dev+cu128** + **mamba_ssm 2.3.1** on Blackwell. The RunPod 2.4.0 / CUDA 12.4 image may work, but if you hit `no kernel image` or `mamba_ssm` errors, install **Blackwell-compatible** `mamba_ssm` + `causal_conv1d` wheels (Kaggle discussion **#681820**) or switch to a newer PyTorch/CUDA 12.8 image.

`train.py` does **not** need the Kaggle cutlass/ptxas shims (those are only in `kaggle_notebook.py`).

---

## 2. Directory layout (`train.py` expects)

Default paths come from `training/train.py` (run as `/workspace/train.py` or set env vars).

```
/workspace/
├── train.py                          # copy from repo training/train.py
├── solvers/                          # REQUIRED — train.py imports solvers.solver
│   ├── __init__.py
│   ├── solver.py
│   └── … (copy whole solvers/ from repo)
├── data/
│   ├── sft_train.jsonl               # required (unless train.csv fallback)
│   ├── sft_val.jsonl                 # required
│   └── train.csv                     # optional fallback only
├── output/                           # created automatically
│   ├── adapter_config.json           # best adapter
│   ├── adapter_model.safetensors
│   ├── submission.zip                # packaged at end
│   └── checkpoints/
│       ├── step_200/
│       ├── step_400/
│       └── best/
├── logs/
│   └── train_YYYYMMDD_HHMM.log
└── .cache/kagglehub/models/metric/
    └── nemotron-3-nano-30b-a3b-bf16/transformers/default/1/   # default MODEL_PATH
```

### Path → env override

| Default path | Env var |
|--------------|---------|
| `/workspace/data` | `DATA_DIR` |
| `/workspace/output` | `OUTPUT_DIR` |
| `/workspace/output/checkpoints` | `CHECKPOINT_DIR` |
| `…/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/…` | `MODEL_PATH` |
| (empty = off) | `TOKEN_CACHE` e.g. `/workspace/data/.token_cache` |

### Data loading logic

1. If `DATA_DIR/sft_train.jsonl` exists → use JSONL (your case).
2. Else if `DATA_DIR/train.csv` missing → error.
3. Else → fallback to CSV (not recommended for this run).

Training also filters JSONL at load time (drops legacy fallback marker + Result/boxed mismatches — should drop **0 rows** with May 30 JSONL).

---

## 3. One-time install (run on pod, **before** training)

SSH into pod, then:

```bash
cd /workspace

# ── Option A: clone repo (easiest) ──
git clone https://github.com/develops20/NB153-NVIDIA-NEMOTRON.git repo
cp repo/training/train.py /workspace/train.py
cp -r repo/solvers /workspace/solvers

# ── Create dirs ──
mkdir -p /workspace/data /workspace/output /workspace/logs

# ── System (zip for submission packaging) ──
apt-get update && apt-get install -y zip

# ── Python packages ──
pip install --upgrade pip
pip install \
  polars \
  peft \
  transformers \
  accelerate \
  bitsandbytes \
  kagglehub \
  safetensors \
  sentencepiece \
  protobuf

# ── Mamba (required for Nemotron hybrid layers) ──
# Try pip first on your image:
pip install causal-conv1d mamba-ssm

# If import fails on Blackwell, install wheels from Kaggle discussion #681820 instead:
# pip install /path/to/causal_conv1d-*.whl /path/to/mamba_ssm-*.whl
```

### Versions from last good run (reference)

| Package | Version |
|---------|---------|
| torch | 2.12.0.dev+cu128 (image may differ) |
| transformers | 5.8.0 |
| peft | 0.19.1 |
| mamba_ssm | 2.3.1 |
| causal_conv1d | 1.6.1 |

Pin if needed: `pip install transformers==5.8.0 peft==0.19.1`

---

## 4. Download base model (once per volume)

```bash
export KAGGLE_API_TOKEN="your_token"   # from kaggle.com/settings

python3 << 'PY'
import kagglehub
path = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
print("Model at:", path)
PY
```

Default `MODEL_PATH` in `train.py` assumes:

```
/workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
```

If download lands elsewhere, set:

```bash
export MODEL_PATH="/actual/path/from/kagglehub/print"
```

---

## 5. Pre-flight checks (5 min, uses little GPU time)

```bash
cd /workspace

# GPU
nvidia-smi

# Python imports
python3 -c "
import torch
print('torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0))
import mamba_ssm, peft, transformers, polars
from solvers.solver import extract_boxed_answer
print('imports OK')
"

# Data files
ls -lh /workspace/data/sft_train.jsonl /workspace/data/sft_val.jsonl

# Model path
ls -lh "$MODEL_PATH/config.json" 2>/dev/null || ls -lh /workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1/config.json

# Dry load (loads model — ~2–5 min, uses VRAM; cancel if OOM)
# python3 -c "
# from transformers import AutoModelForCausalLM
# import torch
# p = '${MODEL_PATH:-/workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1}'
# AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='cpu')
# print('model config OK')
# "
```

---

## 6. Training env vars

```bash
export DATA_DIR=/workspace/data
export OUTPUT_DIR=/workspace/output
export CHECKPOINT_DIR=/workspace/output/checkpoints
export MODEL_PATH=/workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1

# Patched defaults (optional — these match train.py defaults)
export NUM_EPOCHS=2
export BATCH_SIZE=2
export GRAD_ACCUM=4
export LR=2e-4
export MIN_LR=2e-5
export NUM_WORKERS=4
export SAVE_EVERY_STEPS=200
export KEEP_LAST_N=2

# Speeds up restarts after tokenization (optional, ~5–20 GB on volume)
export TOKEN_CACHE=/workspace/data/.token_cache
```

---

## 7. Start training

```bash
cd /workspace
tmux new -s train

python -u train.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M).log
```

Detach tmux: `Ctrl+B`, then `D`  
Reattach: `tmux attach -t train`  
Tail log: `tail -f logs/train_*.log`

### Expected startup log lines

```
[init] data=/workspace/data | model=...
[lora] init r=32
[tokenize] ... kept / ... skipped
[loader] batch=2 workers=4 pin_memory=True
[train] 2 epochs, ~3652 optimizer steps, warmup=182 | lr 2e-4→2e-5 cosine floor
```

---

## 8. First 30 minutes — what to watch

| Signal | OK | Problem |
|--------|-----|---------|
| VRAM | ~85–94 GB | OOM → stop; don't raise batch size |
| GPU util | 50–70%+ | Low is OK with grad checkpointing |
| `loss` in log | finite numbers | `nan` spam → check finite-loss patch present |
| `lr` | decays toward `2e-5` | — |
| Throughput | ~0.06 st/s | ~17h total at 2 epochs |

### Resume after pod restart

Checkpoints in `CHECKPOINT_DIR/step_*`. Re-run the same `python -u train.py` command — script auto-resumes from latest `step_*` + `trainer_state.json`.

---

## 9. Upload JSONL from Mac

```bash
# Replace POD_SSH with your RunPod SSH command/host
scp data/sft_train.jsonl data/sft_val.jsonl root@POD_IP:/workspace/data/
scp training/train.py root@POD_IP:/workspace/train.py
scp -r solvers root@POD_IP:/workspace/solvers
```

Or use RunPod file browser / `rsync -avz`.

---

## 10. Download results

```bash
# From Mac
scp root@POD_IP:/workspace/output/submission.zip .
scp root@POD_IP:/workspace/output/adapter_*.safetensors .
scp root@POD_IP:/workspace/logs/train_*.log .
scp -r root@POD_IP:/workspace/output/checkpoints/best ./checkpoints_best/
```

---

## 11. Training hyperparameters (fixed in train.py)

| Param | Value |
|-------|-------|
| LoRA rank | 32 (must stay ≤ 32 for submission) |
| MAX_SEQ_LEN | 2048 |
| BATCH_SIZE | 2 |
| GRAD_ACCUM | 4 (effective batch 8) |
| LR | 2e-4 → floor 2e-5 |
| Epochs | 2 |
| Precision | bfloat16 |
| Gradient checkpointing | on |

---

## 12. Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: solvers` | Copy `/workspace/solvers/`; run from `/workspace` |
| `ModuleNotFoundError: mamba_ssm` | Install Blackwell wheels (#681820) |
| `CUDA OOM` | Don't increase batch; seq already 2048 |
| `sft_train.jsonl not found` | Upload to `/workspace/data/` |
| `no kernel image` | Wrong CUDA/torch for Blackwell — update image or wheels |
| Pod idle billing | **Stop pod** when not training; keep network volume |

---

## 13. Quick reference — copy/paste block

```bash
# === ON POD (after SSH) ===
cd /workspace
git clone https://github.com/develops20/NB153-NVIDIA-NEMOTRON.git repo
cp repo/training/train.py . && cp -r repo/solvers .
mkdir -p data output logs
apt-get update && apt-get install -y zip
pip install -U pip polars peft transformers accelerate bitsandbytes kagglehub safetensors
pip install causal-conv1d mamba-ssm   # or Blackwell wheels if this fails

export KAGGLE_API_TOKEN="..."
python3 -c "import kagglehub; print(kagglehub.model_download('metric/nemotron-3-nano-30b-a3b-bf16/transformers/default'))"

export DATA_DIR=/workspace/data OUTPUT_DIR=/workspace/output NUM_WORKERS=4
export TOKEN_CACHE=/workspace/data/.token_cache

tmux new -s train
python -u train.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M).log
```

---

*Generated from `training/train.py` + prior run report. Repo: https://github.com/develops20/NB153-NVIDIA-NEMOTRON*
