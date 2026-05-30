# Session handoff — 2026-05-30

Archive of agent work for **Nvidia Nemotron Kaggle / RunPod SFT**. Use this when the RunPod machine is ready (~3h setup).

---

## Where you are in the plan

| Step | Task | Status |
|------|------|--------|
| 1 | Scrape | Done |
| 2 | Fix solvers + audit | Done (bit 93%, symbol tier policy) |
| 3 | Regenerate JSONL | **Done** — use `data/sft_*.jsonl` (May 30) |
| 4 | Patch `training/train.py` | Done |
| 5 | Fresh SFT on RunPod | **Next** — blocked on pod creation |
| 6 | GRPO | After SFT eval |
| 7 | Submit | After GRPO / best adapter |

---

## Use this SFT data (not the old snapshot)

| Path | Use? | Notes |
|------|------|--------|
| **`data/sft_train.jsonl`** | **YES** | 15,569 rows, 0% fallbacks, 0 mismatches |
| **`data/sft_val.jsonl`** | **YES** | 1,730 rows |
| `data/sft-0.74/sft_*.jsonl` | **NO** | Old run: 961+100 fallback templates |

Quality (current, with fixed `\boxed{}` parser):

- `fallback_template`: 0
- `missing_boxed`: 0
- `result_line_vs_boxed_mismatch`: 0
- Oracle-only real rows: 114 (bit solver misses — honest CoT, not fake reasoning)

---

## Code changes completed this session

### `training/train.py`

- `MAX_SEQ_LEN = 2048`
- `MIN_LR = 2e-5` cosine floor
- `NUM_WORKERS=4`, `pin_memory`, `persistent_workers`
- `TOKEN_CACHE` optional disk cache for tokenized tensors
- Finite-loss gating (`torch.isfinite`) — fixes NaN logging artifact
- Imports `extract_boxed_answer` / `reasoning_result_matches` from solvers

### `data_generation/generate_sft_data.py`

- `find_train_csv()` — fixed broken CSV path
- Solver miss → `_oracle_only_reasoning` (not fake fallback template)
- Uses shared boxed/Result helpers from `solvers/solver.py`

### `solvers/solver.py`

- `extract_boxed_answer()` — handles `}` inside answers (e.g. `%?}%`)
- `extract_result_line()`, `reasoning_result_matches()`, `format_boxed_answer()`

### Git

- Repo initialized on `main`, initial commit `32cc164`
- `.gitignore` excludes `venv/`, `data/`, `outputs/`, `nemotron-master/`
- **Review changes:** Source Control sidebar or `git diff`

---

## RunPod checklist (when machine is ready)

1. **GPU:** RTX 6000 Pro 96GB (or similar); same stack as prior run (torch cu128, mamba_ssm, peft)
2. **Upload:**
   ```bash
   scp data/sft_train.jsonl data/sft_val.jsonl user@pod:/workspace/data/
   scp training/train.py user@pod:/workspace/train.py
   ```
3. **Env (optional):**
   ```bash
   export DATA_DIR=/workspace/data
   export OUTPUT_DIR=/workspace/output
   export NUM_WORKERS=4
   export TOKEN_CACHE=/workspace/data/.token_cache
   ```
4. **Train:**
   ```bash
   tmux new -s train
   python -u /workspace/train.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M).log
   ```
5. **Watch:** VRAM ~94/96GB is normal at batch=2; log should show `lr 2e-4→2e-5` and no persistent `loss nan`

Prior run reference: `outputs/training_report (1).md` — val_loss 0.0388, ~17.5h, 0.06 st/s.

---

## Not done (optional later)

- 16 bit-manipulation rows where winner had `rule_found` but our solver fails
- `kaggle_notebook.py` / `evaluation/evaluate.py` — still old `\boxed{[^}]*}` regex
- GRPO stage
- Regenerate JSONL only if solvers or synthetic counts change

---

## Resume this work in Cursor

Paste into a new chat:

> Read `SESSION_HANDOFF.md`. RunPod is ready. Help me start SFT and monitor the first epoch.

Or for solver work:

> Read `SESSION_HANDOFF.md`. Investigate the 16 bit `rule_found` regressions vs `nemotron-master/problems.jsonl`.

---

## Key local paths

```
data/sft_train.jsonl          ← training data (upload this)
data/sft_val.jsonl
training/train.py             ← RunPod training script
data_generation/generate_sft_data.py
solvers/
raw-data/train.csv
outputs/training_report (1).md
nemotron-master/              ← winner reference (local only, gitignored)
```
