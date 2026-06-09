# GRPO iteration guide (if score still below 0.82)

Use after first GRPO submit. Do **not** run another full SFT regen without GRPO — SFT alone plateaued at **0.74**.

---

## Step 1 — Compare fingerprints

```bash
# run2 SFT:  079d43f8f2f4bfd3edf351f84917d52d
# GRPO v1:   (your new hash — must differ)
python3 -c "..." /path/to/adapter_model.safetensors
```

If fingerprints match, you submitted the wrong adapter.

---

## Step 2 — Per-type model eval (RunPod GPU)

```bash
export MODEL_PATH=/workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
python evaluation/evaluate.py \
  --adapter /workspace/output_grpo \
  --csv raw-data/train.csv \
  --max-samples 500 \
  --output-json evaluation/model_eval_sample.json
```

Compare per-type accuracy vs `evaluation/diagnosis_baseline.json` (solver upper bound).

---

## Step 3 — Fix highest-ROI gaps

### Bit manipulation (~7% solver misses)

```bash
python data_generation/generate_sft_data.py --audit-csv-only --csv raw-data/train.csv
```

Investigate failing IDs in audit output; fix `solvers/bit_manipulation.py`.

### Symbol equation (oracle-only rows)

884 train rows use oracle CoT — model must generalize without gold hints. Options:

- Increase synthetics: `symbol_equation_digit` count in `generate_sft_data.py`
- Short SFT refresh (1 epoch) on new JSONL, then GRPO pass 2

---

## Step 4 — GRPO pass 2

Start from GRPO v1 adapter (not base model):

```bash
export SFT_ADAPTER=/workspace/output_grpo   # previous GRPO output
export OUTPUT_DIR=/workspace/output_grpo2
export GRPO_LR=2e-6                         # lower LR
export GRPO_NUM_EPOCHS=1
python -u /workspace/grpo_train.py
```

---

## Step 5 — Submit best adapter

Keep the highest LB score among SFT / GRPO v1 / GRPO v2 (5 submissions/day).

---

## Do not spend time on

- Third full SFT regen without GRPO
- Hybrid solver-at-inference for official LB (competition uses adapter-only vLLM)
- Chasing val_loss alone

---

## Stretch goal (0.86+)

Study public 0.86 notebooks in `scraper/competition-data/codeanalysis.md`:

- `safar1/lb-score-0-86`
- `markjcooper/thk-public-fork-2026-05-14-v14-tinker-adapter`
- `mohamedamr992/end-to-end-finetuning-for-lb`

Common pattern: SFT → GRPO → replay hard failures → second GRPO.
