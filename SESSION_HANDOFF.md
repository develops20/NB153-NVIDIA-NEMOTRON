# Session handoff — 2026-06-09

Archive for **Nvidia Nemotron Kaggle / RunPod**. SFT complete at **public LB 0.74**; next stage is **GRPO**.

---

## Where you are

| Step | Task | Status |
|------|------|--------|
| 1 | Solvers + SFT JSONL | Done (98.6% solver audit, 0% fallbacks) |
| 2 | SFT RunPod run2 | Done — val_loss **0.0369**, 18.24h, **LB 0.74** |
| 3 | Per-type diagnosis | Done — see `evaluation/diagnosis_baseline.json` |
| 4 | **GRPO on RunPod** | **Next** — `training/grpo_train.py` |
| 5 | Submit GRPO adapter | After GRPO completes |
| 6 | Public notebook + write-up | Required by Jun 15 for prizes |

---

## Key results

### SFT run2 (Jun 8)

- Adapter fingerprint: `079d43f8f2f4bfd3edf351f84917d52d`
- Public LB: **0.74** (same as run1 despite better data)
- Files: `outputs/run2/`

### Why SFT plateaued

Competition scores **adapter only** via vLLM — solvers do not run at submit time. SFT teaches format; GRPO teaches correct answers via rewards.

### Solver upper bound (train.csv)

| Type | Solver accuracy |
|------|-----------------|
| cipher, gravity, roman, unit | 100% |
| bit_manipulation | 93.07% |
| symbol_equation | 62.38% (extract_boxed failures on crypt answers) |

Full JSON: `evaluation/diagnosis_baseline.json`

---

## GRPO quick start (RunPod)

```bash
# Upload
scp training/grpo_train.py trl_wheels/trl-0.29.1-py3-none-any.whl root@POD:/workspace/
scp -r solvers root@POD:/workspace/solvers

# On pod (after SFT adapter exists at /workspace/output)
pip install /workspace/trl-0.29.1-py3-none-any.whl datasets
export SFT_ADAPTER=/workspace/output
export OUTPUT_DIR=/workspace/output_grpo
export DATA_DIR=/workspace/data

tmux new -s grpo
python -u /workspace/grpo_train.py 2>&1 | tee logs/grpo_$(date +%Y%m%d_%H%M).log
```

Full checklist: [runpodplan.md](runpodplan.md) section 14.

---

## Submit after GRPO

1. Fingerprint new adapter (must ≠ `079d43f8...`)
2. Upload to Kaggle Models
3. Run [training/kaggle_inference.py](training/kaggle_inference.py) — set `ADAPTER_PATH` env or edit default
4. Submit `submission.zip`

---

## Resume in Cursor

> Read `SESSION_HANDOFF.md`. GRPO finished on RunPod — help me fingerprint, upload, and submit.

Or:

> Read `evaluation/GRPO_ITERATION.md`. GRPO v1 scored below 0.82 — plan iteration.

---

## Key paths

```
training/grpo_train.py       ← GRPO (RunPod)
training/train.py            ← SFT (done)
training/kaggle_inference.py ← Package submission.zip
evaluation/evaluate.py       ← Per-type eval (fixed boxed parser)
evaluation/diagnosis_baseline.json
data/sft_train.jsonl
outputs/run2/                  ← SFT adapter (0.74 LB)
runpodplan.md                  ← Full RunPod + GRPO guide
```
