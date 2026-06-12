# MASTER RUNBOOK — Nemotron-3-Nano-30B  (re-SFT → GRPO → Submit)

> **This is the single source of truth.** It supersedes both the old `runpodplan.md`
> and the interim `runpod_resft_grpo_runbook.txt`. Delete/ignore those to avoid confusion.
>
> Model: hybrid Mamba/MoE `NemotronHForCausalLM`, ~30B params (~63 GB bf16), 13 shards.
> GPU: RTX PRO 6000 Blackwell (sm_120, 96 GB). **Driver caps CUDA at 12.8 → stay on cu12.**

---

## 0. WHY THIS ROUND EXISTS (read first)

The previous two SFT runs over-trained to val_loss ~0.033 and **collapsed the model into
overconfidence**: token entropy ~0.04, generations near-identical. Consequences:
- Leaderboard stuck at **0.74** across two *different* adapters (different weights, same score).
- GRPO had **zero reward variance** (`reward_std=0`) → no gradient → couldn't learn.
- Diversity only appeared at temperature ~2.0 (entropy jumped 0.05 → 5.0), which is too
  hot to be useful (incoherent, rambling, slow).

**Fix:** re-SFT with **label smoothing + early stop + LR floor**, monitored by a live
**entropy gate**, to produce a checkpoint that keeps generation diversity (target val
entropy ~1–3 at temp 1.0). THEN run GRPO on that healthier checkpoint, where variance
appears at a sane temperature (~1.0–1.2).

**The 0.74 adapter is already submitted on Kaggle = the floor. This round is upside-only.**

---

## 1. POD SPECS (decided — do NOT iterate on GPU choice)

| Setting | Value |
|---|---|
| GPU | RTX PRO 6000 Blackwell, 96 GB (single GPU) |
| Image | a CUDA 12.x **devel** Ubuntu image (e.g. `runpod/pytorch ... cuda12.x devel`) |
| Container disk | 50 GB |
| Network volume | **200 GB**, mounted at `/workspace` (persists across stop) |
| Expose | SSH (+ Jupyter optional) |

**GPU decision is final:** single RTX 6000 + **HF-generate GRPO**. We are NOT using vLLM
(colocate does not fit a 30B model + training on one 96 GB GPU — proven this session: OOM).
vLLM would require a **multi-GPU pod from the start** (cannot hot-add a GPU) at ~3× cost;
only revisit if a *proven* GRPO run is too slow for the time budget. See §10 note.

### Volume size breakdown
| Item | ~Size |
|---|---|
| Base model (bf16, kagglehub cache) | 55–65 GB |
| LoRA adapter + submission.zip | ~3.5 GB |
| Step checkpoints (KEEP_LAST_N=2 + best/) | ~7–12 GB |
| SFT JSONL (train+val) | ~61 MB |
| Optional TOKEN_CACHE (.pt) | 5–20 GB |
| Logs | <100 MB |

---

## 2. DIRECTORY LAYOUT (train.py / grpo_train.py expect)

```
/workspace/
├── train.py                      # re-SFT script (anti-collapse version)
├── grpo_train.py                 # GRPO script (HF-generate, patched)
├── solvers/                      # REQUIRED — imported as solvers.solver
│   ├── __init__.py
│   └── solver.py                 # extract_boxed_answer, solve_puzzle, verify_answer,
│                                 #   reasoning_result_matches
├── data/
│   ├── sft_train.jsonl           # required (messages: system/user/assistant)
│   ├── sft_val.jsonl             # REQUIRED this round (early-stop + entropy gate need it)
│   └── train.csv                 # optional (only for regen/audit)
├── output/                       # SFT best (healthy) adapter lands here
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── checkpoints/{step_N, best}/
├── output_grpo/                  # GRPO output
├── logs/
└── .cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1/
```

| Default path | Env override |
|---|---|
| `/workspace/data` | `DATA_DIR` |
| `/workspace/output` | `OUTPUT_DIR` |
| `/workspace/output/checkpoints` | `CHECKPOINT_DIR` |
| `.../default/1` | `MODEL_PATH` |
| (empty=off) | `TOKEN_CACHE` e.g. `/workspace/data/.token_cache` |

---

## 3. ONE-TIME ENVIRONMENT SETUP (before any training)

**Order matters. Every `pip install` here risks swapping torch — follow exactly.**

### 3.1 Shell prep
```bash
cd /workspace
mkdir -p data output logs solvers
apt-get update && apt-get install -y zip wget tmux
```

### 3.2 CUDA 12.8 toolkit (to BUILD mamba; matches the driver)
```bash
# If /usr/local/cuda-12.8/bin/nvcc already exists and prints 12.8, skip the apt install.
/usr/local/cuda-12.8/bin/nvcc --version | tail -2     # want: release 12.8

# If missing: install via apt. If you hit "Conflicting values set for option Signed-By":
#   ls /etc/apt/sources.list.d/ | grep -i cuda
#   rm -f /etc/apt/sources.list.d/cuda.list      # remove the source WITHOUT signed-by
#   apt-get update && apt-get install -y cuda-toolkit-12-8
```

### 3.3 PyTorch — STABLE cu128 (driver-safe, ships stable triton 3.4.0)
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python3 -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list())"
# WANT: 2.8.0+cu128 , 12.8 , and 'sm_120' present.
```
> Why stable 2.8 not nightly: nightly cu128 pulls **triton 3.7.0** which breaks sm_120
> ("device kernel image is invalid") and forces a `triton==3.6.0` pin. Stable 2.8 → triton
> 3.4.0, no Blackwell issue, no pin needed. (If you ever DO end up on nightly+triton 3.7,
> the fix is: `pip install "triton==3.6.0"; rm -rf ~/.triton/cache`.)

### 3.4 Build causal_conv1d + mamba_ssm (against the torch you just installed)
**Rules learned the hard way — non-negotiable:**
- Always `--no-deps` AND `--no-cache-dir` on BOTH. `--no-deps` stops mamba from
  uninstalling your torch and pulling a cu130/CUDA-13 stack (which the driver can't run).
  `--no-cache-dir` stops reuse of a stale wheel built against a different torch (ABI
  mismatch → import error/segfault).
- Pin `mamba_ssm==2.3.1`.
- `CUDA_HOME` major must match torch's CUDA major (12.8 toolkit for cu128 torch).

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:/usr/bin:/bin:/usr/local/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib64
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
nvcc --version | tail -2                       # confirm 12.8

pip install einops ninja huggingface_hub
pip uninstall -y causal_conv1d mamba_ssm
pip install git+https://github.com/Dao-AILab/causal-conv1d.git@v1.6.1 \
    --no-build-isolation --no-deps --no-cache-dir
pip install mamba_ssm==2.3.1 \
    --no-build-isolation --no-deps --no-cache-dir

# VERIFY torch NOT swapped:
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"   # MUST be 2.8.0+cu128
```

### 3.5 Remaining Python deps
```bash
pip install transformers accelerate peft datasets polars safetensors \
    sentencepiece protobuf --no-cache-dir
pip uninstall -y bitsandbytes      # not used; silences libnvJitLink warning at LoRA init

python3 -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__)"
# WANT: torch 2.8.0+cu128 | triton 3.4.0
```

### 3.6 Mamba smoke test (must pass before training)
```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas
rm -rf ~/.triton/cache
python3 -c "
import torch, mamba_ssm, causal_conv1d
from mamba_ssm import Mamba2
m = Mamba2(d_model=256, d_state=64, d_conv=4, expand=2).to('cuda', dtype=torch.bfloat16)
y = m(torch.randn(2,128,256, device='cuda', dtype=torch.bfloat16)); torch.cuda.synchronize()
print('Mamba2 OK', tuple(y.shape))
"
# WANT: Mamba2 OK (2, 128, 256)
```

### 3.7 Base model (once per volume)
```bash
export KAGGLEHUB_CACHE=/workspace/.cache/kagglehub
# (kagglehub reads ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)
python3 -c "import kagglehub; print(kagglehub.model_download('metric/nemotron-3-nano-30b-a3b-bf16/transformers/default'))"
# Expected: /workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
```

### Golden rules (pin these in your head)
| Rule | Why |
|---|---|
| Every torch-dependent `pip install` → `--no-deps` (+`--no-cache-dir` for compiled) | Anything listing torch can swap it; mamba/causal_conv1d/trl/vllm/transformers all do |
| After every such install, re-check `torch.__version__` | Catch a hijack immediately |
| Driver caps CUDA 12.8 → cu130 fails "driver too old" | Stay on cu128 builds |
| Reinstalling torch ⇒ rebuild causal_conv1d + mamba_ssm | ABI is torch-version-specific |
| CUDA_HOME major == torch CUDA major | else "headers incompatible" / build fails |

---

## 4. RE-SFT  (the point of this round)

`train.py` is the anti-collapse version with: label smoothing, LR floor (MIN_LR), weight
decay, **mid-epoch eval every EVAL_EVERY_STEPS**, **entropy-gated "best" save**, and
early-stop on entropy floor / patience. Two trackers: `best_val_loss` (drives patience),
`best_healthy_val_loss` (drives what gets saved — only persists when entropy ≥ GOOD_ENTROPY).

### Env + launch
```bash
cd /workspace
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64
export TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas
export MODEL_PATH=/workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
export DATA_DIR=/workspace/data
export OUTPUT_DIR=/workspace/output

# anti-collapse knobs
export NUM_EPOCHS=1
export LABEL_SMOOTHING=0.1          # bump to 0.15 if entropy slides fast early
export WEIGHT_DECAY=0.05
export MIN_LR=2e-5                   # LR floor (don't decay to 0)
export EVAL_EVERY_STEPS=50
export EARLY_STOP_PATIENCE=2
export GOOD_ENTROPY=1.0             # only SAVE checkpoints with entropy >= this
export MIN_ENTROPY=0.5              # STOP (collapse alarm) if entropy drops below this
export TARGET_VAL_LOSS=0.0          # DISABLE loss-target stop; we stop on entropy, not loss
export ENTROPY_VAL_SAMPLES=4
export ENTROPY_MAX_NEW_TOKENS=128
export ENTROPY_TEMPERATURE=1.0

tmux new -s sft
python -u train.py 2>&1 | tee logs/sft_$(date +%Y%m%d_%H%M).log
# detach: Ctrl-b d   |   reattach: tmux attach -t sft
```

### What to watch (this is the whole game)
Healthy pattern — entropy stays 1–3 while val_loss descends; best saves while healthy:
```
[eval] val_loss=0.45 entropy=2.1 ...
[eval] val_loss=0.32 entropy=1.6 ...  [best] healthy checkpoint ...
[eval] val_loss=0.24 entropy=1.2 ...  [best] healthy checkpoint ...
[eval] val_loss=0.20 entropy=0.7 ...  [best] skipped: ... entropy 0.7 < good 1.0
[early_stop] entropy 0.38 < floor 0.5
[final] using best healthy checkpoint
```
- **OUTPUT_DIR automatically holds the last HEALTHY checkpoint** (entropy ≥ 1.0), not a
  lower-loss collapsed one.
- If the FIRST couple of evals show entropy already < ~1.5 or dropping fast → kill, set
  `LABEL_SMOOTHING=0.15` (and/or raise it further), relaunch. Cheap to catch early.
- Do NOT proceed to GRPO on a collapsed checkpoint.

### Fixed hyperparameters (in train.py)
LoRA r=32, alpha=64, target `.*\.(in_proj|out_proj|up_proj|down_proj)$`, dropout 0.05;
MAX_SEQ_LEN 2048; BATCH_SIZE 2; GRAD_ACCUM 4; LR 2e-4 → floor MIN_LR; bf16; grad checkpointing on.
(Note: 1 epoch + label smoothing + entropy-gated early stop REPLACES the old "2 epochs to
val_loss 0.033" recipe that caused the collapse.)

---

## 5. VERIFY THE NEW CHECKPOINT KEPT DIVERSITY (cheap gate before GRPO)
The `[eval]` logs already tell you, but to double-check the saved adapter: load base + the
new OUTPUT_DIR adapter, generate at temp 1.0 on a few prompts, confirm mean token entropy
~1–3 (NOT ~0.04). If healthy → GRPO. If ~0 → re-SFT was still too aggressive (raise
LABEL_SMOOTHING / stop earlier) and redo §4.

---

## 6. GRPO  (HF generate, single GPU — validated path)

### grpo_train.py REQUIRED patches (all confirmed needed)
1. **Top of file, before imports:** `import os; os.environ["HF_HUB_TRUST_REMOTE_CODE"]="1"`
   (the vLLM-only env vars / `vllm.LLM` monkeypatch are NOT needed for HF generate.)
2. **solvers import:** `sys.path.insert(0, "/workspace")` (NOT `os.path... ".."`).
3. **Adapter load (manual):** build `LoraConfig` from `adapter_config.json`, `get_peft_model`,
   copy safetensors into state_dict injecting `.default` before `.weight`.
   `PeftModel.from_pretrained` CRASHES on peft 0.19 + transformers 5.x
   ("WeightConverter ... unexpected kwarg distributed_operation").
4. **Patch modeling_nemotron_h.py** (HF generate path), in BOTH copies (cache regenerates):
   change `if not empty_past_kv:` → `if not empty_past_kv and cache_position is not None:`
   - SRC:   `.../kagglehub/.../default/1/modeling_nemotron_h.py`
   - CACHE: `.../huggingface/modules/transformers_modules/_1/<hash>/modeling_nemotron_h.py`
   (re-apply if you clear the HF modules cache)
5. **GRPOConfig:** NO `vllm_*` args, no `model_init_kwargs`.

> NOTE on transformers/trl versions for HF-generate GRPO: this works on transformers 5.11 +
> trl 1.4 with the manual adapter load + the cache_position patch. (The vLLM detour that
> forced transformers 4.57 / vllm 0.18 is abandoned — we are NOT using vLLM.)

### 6.1 Variance-gate smoke test (MANDATORY before any long run)
```bash
cd /workspace
export SFT_ADAPTER=/workspace/output
export OUTPUT_DIR=/workspace/output_grpo_smoke
export DATA_DIR=/workspace/data
export MODEL_PATH=/workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
export GRPO_MAX_ROWS=16
export GRPO_MAX_STEPS=2
export GRPO_SOLVER_BONUS=0
export GRPO_NUM_GENERATIONS=4
export GRPO_MAX_COMPLETION_LENGTH=768
export GRPO_TEMPERATURE=1.1          # healthy model should give variance here; raise only if std=0
export TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas
python -u grpo_train.py 2>&1 | tee logs/grpo_smoke.log
```
**GATE — do not launch the long run unless ALL hold:**
- `reward_std > 0` AND `rewards/reward_correctness/std > 0`  (real learning signal)
- `completions/clipped_ratio` low (completions finishing, not truncating)
- step_time ~250–360 s (HF generate, gens=4, len≈512)

If `correctness/std == 0`: raise `GRPO_TEMPERATURE` by 0.2 and retry. On the *healthy*
re-SFT model this should NOT need temp 2.0 (that was the collapsed model's symptom).

### 6.2 Full run (size to budget)
Steps that fit in T hours at S s/step = `T*3600/S`. e.g. 35h @ 300s ≈ 420 steps; 48h @ 250s ≈ 690.
Prompts consumed ≈ steps × GRAD_ACCUM(4). Set MAX_ROWS a bit above that.
```bash
cd /workspace
export SFT_ADAPTER=/workspace/output
export OUTPUT_DIR=/workspace/output_grpo
export DATA_DIR=/workspace/data
export MODEL_PATH=/workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
export GRPO_MAX_STEPS=500            # tune to budget
export GRPO_MAX_ROWS=3000
export GRPO_SOLVER_BONUS=0
export GRPO_NUM_GENERATIONS=4
export GRPO_MAX_COMPLETION_LENGTH=768
export GRPO_TEMPERATURE=1.1          # whatever passed the smoke gate
export GRPO_SAVE_STEPS=25            # checkpoint often; a crash loses <=25 steps
export TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas
tmux new -s grpo
python -u grpo_train.py 2>&1 | tee logs/grpo_$(date +%Y%m%d_%H%M).log
```
Watch `reward`, `reward_std`, `reward_correctness/mean` trending UP. If `frac_reward_zero_std`
creeps back to 1.0, variance died — stop and revisit temperature.

---

## 7. SUBMIT
```bash
# Fingerprint must DIFFER from the 0.74 adapter (079d43f8f2f4bfd3edf351f84917d52d):
python3 -c "
import hashlib, os
p='/workspace/output_grpo/adapter_model.safetensors'
f=open(p,'rb'); a=f.read(5_000_000); f.seek(1_000_000_000); b=f.read(5_000_000)
print('size', os.path.getsize(p)); print('fingerprint', hashlib.md5(a+b).hexdigest())
"
```
Then: upload `adapter_config.json` + `adapter_model.safetensors` as a NEW Kaggle dataset
version → run the **zip-only** submission notebook → commit → submit.
- `submission.zip` must contain ONLY those two files at the ROOT. rank ≤ 32.
- Submission notebook copies the two files to /kaggle/working and runs
  `zip -m submission.zip adapter_config.json adapter_model.safetensors`.
- Confirm the attached dataset version is the NEW one before committing.

### Download to Mac (optional backup)
```bash
scp root@POD_IP:/workspace/output_grpo/adapter_model.safetensors .
scp root@POD_IP:/workspace/output_grpo/adapter_config.json .
scp root@POD_IP:/workspace/logs/grpo_*.log .
```

---

## 8. TROUBLESHOOTING
| Symptom | Fix |
|---|---|
| `device kernel image is invalid` (Blackwell) | You're on nightly+triton 3.7. `pip install "triton==3.6.0"; rm -rf ~/.triton/cache`. (Stable torch 2.8 avoids this entirely.) |
| mamba install swapped torch to cu130 | Forgot `--no-deps`. Reinstall torch (§3.3), rebuild mamba with `--no-deps --no-cache-dir`. |
| `headers are incompatible` during mamba build | CUDA_HOME major ≠ torch CUDA major. Point CUDA_HOME at a toolkit matching torch (12.8 for cu128). |
| reused stale mamba wheel ("Using cached") | add `--no-cache-dir`; you want to see "Building wheel". |
| `ModuleNotFoundError: solvers` | `solvers/` at `/workspace/solvers`; run from `/workspace`; `sys.path.insert(0,"/workspace")`. |
| `WeightConverter ... distributed_operation` (GRPO adapter load) | Use the manual adapter load (§6 patch 3), not `PeftModel.from_pretrained`. |
| `cache_position[-1] ... NoneType` (GRPO generate) | Apply the modeling_nemotron_h.py patch in BOTH file copies (§6 patch 4). |
| GRPO `reward_std=0` / `frac_reward_zero_std=1` | No learning signal. On healthy model: raise temperature 0.2. If only fixable at temp~2.0, the SFT checkpoint is still collapsed → redo §4 with more label smoothing. |
| CUDA OOM during vLLM init | vLLM colocate doesn't fit a 30B + training on one 96GB GPU. We don't use vLLM — ensure no `vllm_*` args / monkeypatch are in grpo_train.py. |
| `cu130` / "driver too old (12080)" | torch is a CUDA-13 build; driver maxes at 12.8. Reinstall a cu128 torch. |
| bitsandbytes `libnvJitLink.so.13` at LoRA init | Harmless; `pip uninstall -y bitsandbytes`. |
| Pod idle billing | Stop pod when not running; network volume persists. |

---

## 9. KEY FACTS / NUMBERS
- 0.74 adapter fingerprint (the floor, already submitted): `079d43f8f2f4bfd3edf351f84917d52d`
- Model loads at ~66.7 GB (63.1 GB bf16 + 3.6 GB fp32 LoRA) — no waste; this is correct size.
- HF-generate GRPO step time seen: ~250 s (temp~1, len 512) … ~665 s (temp 2.0, len 768, rambling).
- vLLM 0.11 lacks tokenizer API for tf5.x; vLLM 0.18 needs tf4.57; vLLM 0.22 needs cu13
  (driver-incompatible). **All abandoned — HF generate on single GPU is the chosen path.**
- Competition: grader loads base + your LoRA via vLLM, greedy (temp 0), scores `\boxed{}` accuracy.
  Base/untrained ≈ 0.53; your SFT = 0.74; top ≈ 0.86–0.89 (GRPO).

---

## 10. NOTE — IF you ever want fast vLLM GRPO (future, NOT this round)
Requires a MULTI-GPU pod chosen at deploy time (can't hot-add). Run vLLM in **server mode**
(`vllm_mode="server"`, `trl vllm-serve`) on a 2nd GPU; training keeps GPU 0. That stack
needs torch 2.10+cu128 + vLLM 0.18 + transformers 4.57 + mamba rebuilt against torch 2.10.
~3× hourly cost; only justified once a single-GPU HF-generate run is proven to work but is
too slow for the remaining time. Do NOT start here.
```
```
