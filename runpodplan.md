# RunPod SFT plan — Nemotron-3 Nano 30B LoRA

Follow this **before** starting the GPU. Goal: configure once, then run ~17h training without wasting GPU hours.

Prior successful runs (RTX PRO 6000 Blackwell 96 GB):
- **Run 1:** val_loss **0.0328**, ~**17.5h**, 14,608 train examples, peak VRAM ~94/96 GB. (LR decayed to ~0 → train-loss NaN tail at end; cosmetic.)
- **Run 2:** val_loss **0.0369**, ~**18.2h**, 15,569 train / 1,730 val, peak VRAM ~90/96 GB. (Cosine LR floored at 2e-5 → no NaN tail; cleaner data, 0 drops.)

**Current SFT data:** 15,569 train / 1,730 val examples — 0 fallbacks, 0 mismatches at load time, ~98.6% solver accuracy on train.csv. (If you regenerate the data, update the counts in section 1, section 7 expected-log, and the env note below.)

---

## Checklist (tick as you go)

### A. RunPod pod settings

- [ ] **GPU:** NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) — same class as last run
- [ ] **Template / image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (base nvcc is 12.4 — we upgrade to 12.8 in section 3)
- [ ] **Network volume** mounted at `/workspace` (persistent — survives pod stop)
- [ ] **Container disk:** **50 GB** (enough for OS + pip env if model lives on volume)
- [ ] **Volume disk:** **150 GB** minimum (**200 GB** safer)
- [ ] **Expose:** SSH + Jupyter (optional)
- [ ] **Start pod** only after volume is attached

### B. Upload from Mac (before or right after pod start)

- [ ] `data/sft_train.jsonl` → `/workspace/data/` (15,569 rows — **NOT `sft-0.74/`**)
- [ ] `data/sft_val.jsonl` → `/workspace/data/` (1,730 rows)
- [ ] Copy `training/train.py` + entire `solvers/` folder → `/workspace/`

### C. One-time setup on pod (no GPU training yet)

**Order matters — do not skip steps or reorder:**

- [ ] **Step 1:** Install CUDA 12.8 toolkit + set env vars (section 3.1)
- [ ] **Step 2:** Install PyTorch nightly cu128 (section 3.2)
- [ ] **Step 3:** Rebuild `causal_conv1d` + `mamba_ssm` against new torch (section 3.3)
- [ ] Install remaining Python deps (section 3.4)
- [ ] Download base model via `kagglehub` (section 4)
- [ ] Mamba2 smoke test + import checks (section 5)
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
| **Container image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Ubuntu 22.04 devel image; **must upgrade CUDA toolkit to 12.8** (see section 3) |
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

### Blackwell stack (verified)

The base RunPod image ships **nvcc 12.4**, but Nemotron on Blackwell needs:

| Component | Version |
|-----------|---------|
| CUDA toolkit (nvcc) | **12.8** (V12.8.93) |
| torch | **2.12.0.dev+cu128** (nightly) |
| causal_conv1d | **1.6.1** (built from git) |
| mamba_ssm | **2.3.1** (built from pip, **pinned + `--no-deps`** — see warning below) |
| torch archs | `sm_75`, `sm_80`, `sm_90`, `sm_100`, **`sm_120`** |

> ⚠️ **Critical — mamba_ssm can hijack your torch.** Newer mamba_ssm (e.g. `2.3.2.post1`) declares dependencies (`nvidia-cutlass-dsl`, `tilelang`, `quack-kernels`) that **uninstall your cu128 nightly torch and replace it with cu130 stable + a CUDA 13 stack**. `--no-build-isolation` does **not** prevent this (it only affects the build env, not runtime deps). Always install mamba as `pip install mamba_ssm==2.3.1 --no-deps --no-build-isolation` and verify torch is unchanged afterward. See section 3.3.

Plain `pip install causal-conv1d mamba-ssm` on the stock image **will fail** on Blackwell. Follow section 3 exactly.

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

Training also filters JSONL at load time (drops legacy fallback marker + Result/boxed mismatches — drops **0 rows** with current JSONL).

---

## 3. One-time install (run on pod, **before** training)

SSH into pod, then:

```bash
cd /workspace

# ── Copy code (or clone repo) ──
mkdir -p /workspace/data /workspace/output /workspace/logs
# From Mac (see section 9) or:
# git clone … && cp repo/training/train.py . && cp -r repo/solvers .

# ── System (zip for submission packaging) ──
apt-get update && apt-get install -y zip wget
```

### 3.1 CUDA 12.8 toolkit

The base image has **nvcc 12.4**, but torch nightly is built for **cu128** — you need matching nvcc.

```bash
. /etc/os-release   # confirm Ubuntu 22.04

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-toolkit-12-8

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1

nvcc --version    # expect: release 12.8, V12.8.93
```

**Persist env vars** — add the four `export` lines above to `~/.bashrc` so they survive new shells.

### 3.2 PyTorch nightly cu128 (Blackwell sm_120 support)

`train.py` only needs **torch** — not torchvision or torchaudio. Install torch alone (faster, fewer deps):

```bash
pip uninstall -y torch torchvision torchaudio
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify:

```bash
python3 -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('archs', torch.cuda.get_arch_list())"
```

Expected:

```
torch 2.12.0.dev20260407+cu128   # nightly date may differ
cuda 12.8
archs ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

### 3.3 Rebuild causal_conv1d + mamba_ssm

**Must run in the same shell** where CUDA env vars are set (section 3.1).

> ⚠️ Use `--no-deps` on both, and **pin `mamba_ssm==2.3.1`**. Without this, mamba_ssm's dependency tree will silently uninstall your cu128 nightly torch and pull in cu130 + a CUDA 13 stack, breaking the verified Blackwell setup.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0+PTX"

# Runtime deps that --no-deps would otherwise skip (safe to pre-install)
pip install einops ninja

pip uninstall -y causal_conv1d mamba_ssm

# causal_conv1d — pinned, ~5 min build
pip install git+https://github.com/Dao-AILab/causal-conv1d.git@v1.6.1 --no-build-isolation --no-deps

# mamba_ssm — PINNED to 2.3.1 + --no-deps so it can't swap your torch, ~10 min build
pip install mamba_ssm==2.3.1 --no-build-isolation --no-deps
```

**Immediately verify torch was NOT replaced:**

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
# MUST still show: 2.12.0.dev...+cu128   12.8
# If it shows +cu130 or a CUDA 13 version, mamba pulled the wrong deps —
# reinstall torch (section 3.2) and redo this step with --no-deps.
```

Expected versions:

| Package | Version |
|---------|---------|
| causal_conv1d | 1.6.1 |
| mamba_ssm | 2.3.1 |

> **Note from Run 2:** if you forget `--no-deps` and end up on `mamba_ssm 2.3.2.post1` + torch `2.12.0+cu130`, training *can* still succeed (the cu130 stable torch happened to be ABI-compatible with the kernels and supports sm_120). But this is not guaranteed — stay on the pinned cu128 stack for reproducibility.

### 3.4 Remaining Python packages

```bash
pip install --upgrade pip
pip install \
  polars \
  peft \
  transformers \
  accelerate \
  kagglehub \
  safetensors \
  sentencepiece \
  protobuf
```

**Do not install `bitsandbytes`** — `train.py` loads the model in native bfloat16 and never uses 4/8-bit quant. If bitsandbytes is present (e.g. leftover from the base image), peft may try to load it at LoRA init and print a harmless `libnvJitLink.so.13` error. Uninstall to silence it:

```bash
pip uninstall -y bitsandbytes
```

Pin if needed: `pip install transformers==5.8.0 peft==0.19.1`

### Key things to remember

| Rule | Why |
|------|-----|
| **CUDA toolkit major must match torch build** | `+cu128` torch → CUDA **12.8** nvcc. `+cu124` torch → CUDA **12.4** nvcc. |
| **`TORCH_CUDA_ARCH_LIST="12.0+PTX"`** | Blackwell-specific. Without it, mamba_ssm build may pick wrong archs and fail. |
| **`--no-build-isolation` + `--no-deps`** | `--no-build-isolation` lets the build see your installed torch. `--no-deps` stops mamba_ssm from uninstalling your cu128 torch and pulling cu130 + CUDA 13. **Use both on causal_conv1d and mamba_ssm.** Pin `mamba_ssm==2.3.1`. |
| **`FORCE_CUDA=1`** | Forces CUDA kernel compilation even if GPU isn't detected at build time. |
| **Order: CUDA → torch → mamba** | Each step depends on the previous. **Reinstalling torch requires rebuilding causal_conv1d + mamba_ssm.** Always verify `torch.__version__` still shows `+cu128` after the mamba step. |

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

## 5. Pre-flight checks (5 min)

```bash
cd /workspace

# GPU
nvidia-smi

# Mamba2 smoke test (confirms Blackwell stack)
python3 << 'PY'
import torch, mamba_ssm, causal_conv1d
from mamba_ssm import Mamba2
m = Mamba2(d_model=256, d_state=64, d_conv=4, expand=2).to('cuda', dtype=torch.bfloat16)
y = m(torch.randn(2, 128, 256, device='cuda', dtype=torch.bfloat16))
torch.cuda.synchronize()
print('Mamba2 layer OK, output shape', tuple(y.shape))
PY
# Expected: Mamba2 layer OK, output shape (2, 128, 256)

# Other imports
python3 -c "
import torch
print('torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0))
import peft, transformers, polars
from solvers.solver import extract_boxed_answer
print('imports OK')
"

# Data files
ls -lh /workspace/data/sft_train.jsonl /workspace/data/sft_val.jsonl

# Model path
ls -lh "$MODEL_PATH/config.json" 2>/dev/null || ls -lh /workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1/config.json
```

---

## 6. Training env vars

```bash
# CUDA libs (needed if bitsandbytes or other CUDA tools are on the image)
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export DATA_DIR=/workspace/data
export OUTPUT_DIR=/workspace/output
export CHECKPOINT_DIR=/workspace/output/checkpoints
export MODEL_PATH=/workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1

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
[data] train=15569 val=1730
[tokenize] ... kept / ... skipped
[loader] batch=2 workers=4 pin_memory=True
[train] 2 epochs, ~3894 optimizer steps, warmup=194 | lr 2e-4→2e-5 cosine floor
```

---

## 8. First 30 minutes — what to watch

| Signal | OK | Problem |
|--------|-----|---------|
| VRAM | ~85–94 GB | OOM → stop; don't raise batch size |
| GPU util | 50–70%+ | Low is OK with grad checkpointing |
| `loss` in log | finite numbers | `nan` spam → check finite-loss patch present |
| `lr` | decays toward `2e-5` | — |
| Throughput | ~0.05–0.06 st/s | ~17–23h total at 2 epochs |

### Harmless startup noise (safe to ignore)

| Message | Meaning |
|---------|---------|
| `bitsandbytes library load error: libnvJitLink.so.13` | peft probing bnb; training uses native bf16 — **does not block training** |
| `` `torch_dtype` is deprecated `` | transformers 5.x rename — cosmetic |
| `` `use_return_dict` is deprecated `` | transformers 5.x internal — cosmetic |

### Resume after pod restart

Checkpoints in `CHECKPOINT_DIR/step_*`. Re-run the same `python -u train.py` command — script auto-resumes from latest `step_*` + `trainer_state.json`.

---

## 9. Upload JSONL from Mac

```bash
# Replace POD_IP with your RunPod SSH host
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
| `ModuleNotFoundError: mamba_ssm` | Re-run section 3.3 with CUDA env vars + `TORCH_CUDA_ARCH_LIST="12.0+PTX"` |
| **mamba install replaced torch / pulled cu130** | mamba_ssm 2.3.2+ deps uninstall cu128 torch. Reinstall torch (section 3.2), then `pip install mamba_ssm==2.3.1 --no-deps --no-build-isolation`. Verify: `python -c "import torch;print(torch.__version__)"` must show `+cu128`. |
| `no kernel image` / CUDA arch mismatch | Wrong torch for Blackwell — use cu128 nightly (section 3.2) |
| mamba build fails | Check `nvcc --version` is 12.8; use `--no-build-isolation`; set `FORCE_CUDA=1` |
| Reinstalled torch, mamba broken | Must rebuild causal_conv1d + mamba_ssm (section 3.3) |
| `libnvJitLink.so.13` / bitsandbytes load error at LoRA init | **Harmless** if training continues — uninstall `bitsandbytes` (`pip uninstall -y bitsandbytes`); not used by `train.py`. Or set `LD_LIBRARY_PATH=$CUDA_HOME/lib64:...` before launch |
| `torch_dtype` / `use_return_dict` deprecation warnings | Harmless transformers 5.x noise — fixed in repo `train.py` for `dtype=` |
| `sft_train.jsonl not found` | Upload to `/workspace/data/` |
| Pod idle billing | **Stop pod** when not training; keep network volume |

---

## 13. Quick reference — copy/paste block

```bash
# === ON POD (after SSH) — full Blackwell stack ===
cd /workspace
mkdir -p data output logs

# 1. CUDA 12.8 toolkit
. /etc/os-release
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get install -y cuda-toolkit-12-8 zip wget
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1
nvcc --version

# 2. PyTorch nightly cu128 (torch only — no torchvision/torchaudio)
pip uninstall -y torch torchvision torchaudio
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Mamba (rebuild against new torch — PIN + --no-deps so it can't swap your torch)
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
pip install einops ninja
pip uninstall -y causal_conv1d mamba_ssm
pip install git+https://github.com/Dao-AILab/causal-conv1d.git@v1.6.1 --no-build-isolation --no-deps
pip install mamba_ssm==2.3.1 --no-build-isolation --no-deps
# VERIFY torch unchanged (must still be +cu128):
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"

# 4. Other deps (no bitsandbytes — train.py uses native bf16)
pip install -U pip polars peft transformers accelerate kagglehub safetensors sentencepiece protobuf
pip uninstall -y bitsandbytes   # optional: silence libnvJitLink warning at LoRA init

# 5. Mamba smoke test
python3 -c "
import torch; from mamba_ssm import Mamba2
m = Mamba2(d_model=256, d_state=64, d_conv=4, expand=2).to('cuda', dtype=torch.bfloat16)
y = m(torch.randn(2, 128, 256, device='cuda', dtype=torch.bfloat16))
torch.cuda.synchronize(); print('Mamba2 OK', tuple(y.shape))
"

# 6. Download model
export KAGGLE_API_TOKEN="..."
python3 -c "import kagglehub; print(kagglehub.model_download('metric/nemotron-3-nano-30b-a3b-bf16/transformers/default'))"

# 7. Train
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DATA_DIR=/workspace/data OUTPUT_DIR=/workspace/output NUM_WORKERS=4
export TOKEN_CACHE=/workspace/data/.token_cache
tmux new -s train
python -u train.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M).log
```

---

*Updated after Run 2 — Blackwell CUDA 12.8 + torch cu128 + mamba rebuild sequence. Key fix: pin `mamba_ssm==2.3.1` with `--no-deps` to prevent it from swapping torch to cu130. Run 2 result: val_loss 0.0369 on 15,569 clean examples, ~18.2h. Public LB: **0.74** (SFT plateau — see section 14 for GRPO).*

---

## 14. GRPO stage (after SFT — target 0.80+)

SFT alone plateaued at **public LB 0.74**. GRPO continues from the SFT adapter with verifiable rewards via [training/grpo_train.py](training/grpo_train.py).

### Prerequisites

- SFT adapter at `/workspace/output/` (run2: val_loss 0.0369, fingerprint `079d43f8f2f4bfd3edf351f84917d52d`)
- `data/sft_train.jsonl` on pod
- Upload: `grpo_train.py`, `solvers/`, `trl_wheels/trl-0.29.1-py3-none-any.whl`

### Install TRL on pod

```bash
pip install /workspace/trl_wheels/trl-0.29.1-py3-none-any.whl
pip install datasets
```

### Smoke test (~5 min)

```bash
export SFT_ADAPTER=/workspace/output
export OUTPUT_DIR=/workspace/output_grpo_smoke
export DATA_DIR=/workspace/data
export GRPO_MAX_ROWS=10
export GRPO_MAX_STEPS=2
python -u /workspace/grpo_train.py
```

### Full GRPO run (~8–12h)

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export SFT_ADAPTER=/workspace/output
export OUTPUT_DIR=/workspace/output_grpo
export DATA_DIR=/workspace/data
unset GRPO_MAX_ROWS GRPO_MAX_STEPS

tmux new -s grpo
python -u /workspace/grpo_train.py 2>&1 | tee logs/grpo_$(date +%Y%m%d_%H%M).log
```

### Fingerprint + download

```bash
python3 -c "
import hashlib, sys
f=sys.argv[1]
with open(f,'rb') as fh:
    fh.seek(0); a=fh.read(5_000_000)
    fh.seek(1_000_000_000); b=fh.read(5_000_000)
print(hashlib.md5(a+b).hexdigest(), f)
" /workspace/output_grpo/adapter_model.safetensors
# Must differ from run2: 079d43f8f2f4bfd3edf351f84917d52d

scp root@POD:/workspace/output_grpo/submission.zip .
```

### Submit on Kaggle

1. Upload adapter to Kaggle Models (`nemotron-sft-adapter-grpo-v1`)
2. Run [training/kaggle_inference.py](training/kaggle_inference.py) with `ADAPTER_PATH` set
3. Submit `submission.zip` (5/day limit)

See [evaluation/GRPO_ITERATION.md](evaluation/GRPO_ITERATION.md) if score stays below 0.82.

