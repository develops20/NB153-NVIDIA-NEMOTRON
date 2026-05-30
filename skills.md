# Required Skills — NVIDIA Nemotron Competition

**Last updated:** 2026-05-29

## Overview

This document maps the technical skills required across each phase of the competition pipeline. Use it as a pre-flight checklist before implementation.

**Competition snapshot (from `scraper/competition-data/`, May 2026):**

- **Deadline:** 2026-06-15 · **Teams:** 3,644 · **~1,020** public-scored notebooks
- **Public score bar:** ~**0.86** (Models tab); active discussion on breaking the **0.86 ceiling**
- **Metric:** vLLM + Nemotron-3-Nano-30B + your LoRA; final answer in `\boxed{}`; string match or **10⁻²** relative numeric tolerance; `max_tokens=7680`, `temperature=0.0`, `max_model_len=8192`, `max_lora_rank=32`
- **Our baseline:** end-to-end submit at **0.50**; solver accuracy on `train.csv` is the current bottleneck

**Repo paths:**

| Path | Use |
|------|-----|
| `raw-data/train.csv`, `raw-data/test.csv` | Official puzzles (local EDA, solvers, audits) |
| `scraper/competition-data/*.md` | Live rules, metric text, discussions, top notebook index |
| `data/sft_*.jsonl` | Training data |
| `solvers/`, `data_generation/generate_sft_data.py` | Solvers + CoT generation |
| `training/kaggle_notebook.py`, `training/train.py` | Kaggle vs RunPod training |

---

## Core Technical Skills

### 1. Python Data Science
- **pandas** — loading/slicing large CSV files, groupby for puzzle type analysis
- **numpy** — numerical computations (gravity fitting, unit ratios, bit operations)
- **scipy** — least squares curve fitting for physics puzzles
- **re** — regex for extracting `\boxed{}` answers from model output
- **collections** — Counter, defaultdict for cipher word mapping

### 2. Bit Manipulation Algorithms
- Python bitwise operators: `&`, `|`, `^`, `~`, `<<`, `>>`
- Rotation functions: `rotate_left(x, n, bits=8)`, `rotate_right(x, n, bits=8)`
- Brute-force search over composition of operations
- SHA-256 helper functions: Majority (`Maj`) and Choice (`Ch`) functions
- Constraint satisfaction: testing candidate operations against all examples
- **Competition note:** Top community solvers report very high symbolic/bit accuracy; aim for high 90s%+ on `raw-data/train.csv` before trusting CoT labels

### 3. NLP / Cipher Solving
- Word alignment and tokenization
- Frequency analysis on word-level and character-level
- Bijective mapping construction (build dict from examples)
- Handling OOV (out-of-vocabulary) words via character-level pattern matching

### 4. LLM Fine-Tuning
- **Hugging Face Transformers** — `AutoModelForCausalLM`, `AutoTokenizer`, `BitsAndBytesConfig`
- **PEFT (Parameter-Efficient Fine-Tuning)** — `LoraConfig`, `get_peft_model`, `TaskType.CAUSAL_LM`
- **TRL (Transformer Reinforcement Learning)** — `SFTTrainer`, `GRPOTrainer`, `GRPOConfig` (**pin 0.29.1** in production/Kaggle)
- **Unsloth** — memory-efficient patching for Nemotron architecture (MoE/Mamba); optional vs plain TRL+PEFT on RunPod
- `TrainingArguments` hyperparameter tuning
- Gradient checkpointing, mixed precision (BF16)
- Nemotron chat template: `tokenizer.apply_chat_template()`; CoT in `<think>...</think>`

### 5. Reinforcement Learning from Verifiable Rewards (RLVR)
- GRPO algorithm mechanics (group sampling, advantage estimation)
- Writing reward functions compatible with TRL's `GRPOTrainer`
- Format reward design (penalize missing `\boxed{}`, reward `<think>` blocks)
- KL divergence penalty tuning to prevent reward hacking
- Monitoring reward curves and detecting training instability (see discussions on H20 / exploding `grad_norm`)

### 6. Prompt Engineering
- Designing chain-of-thought (CoT) prompts for each puzzle type
- System prompt crafting for reasoning models
- Formatting training examples in conversation template format (ChatML / Nemotron format)
- Temperature and sampling parameter selection for GRPO exploration (rollout); **0.0** for submission inference

### 7. Data Generation / Synthetic Data
- Programmatic generation of bit manipulation puzzle instances
- Random cipher dictionary generation + sentence construction
- Physics puzzle generation (random g values, random time values)
- Unit conversion puzzle generation (random scale factors)
- CoT annotation: writing step-by-step solution templates per puzzle type
- **Trusted CoT gating:** only emit training rows when solvers match ground truth (`generate_sft_data.py`)

### 8. Evaluation & Metrics
- String exact match comparison
- Numerical comparison with relative tolerance (**10⁻²**, same as competition metric)
- Per-category accuracy breakdown
- Local harness: `evaluation/evaluate.py` (`--audit-solvers`, `--audit-sft`, `--solver-only`, optional `--adapter`)
- Solver batch test: `solvers/test_solvers.py` on `raw-data/train.csv`
- Understanding `\boxed{}` extraction heuristics from the [NVIDIA Nemotron Metric](https://www.kaggle.com/code/metric/nvidia-nemotron-metric)

### 9. Model Packaging & Submission
- Understanding LoRA adapter file structure (`adapter_config.json`, `adapter_model.safetensors`)
- `adapter_config.json` constraints: `r ≤ 32`, correct `base_model_name_or_path`
- Creating `submission.zip` with correct file structure
- Using `model.save_pretrained()` and `peft.save_pretrained()`
- Reference notebook: [NVIDIA Nemotron Submission Demo](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo)

### 10. Kaggle / RunPod Environment
- **Kaggle:** GPU notebook (RTX PRO 6000 Blackwell), `/kaggle/input/...`, offline wheels (`trl_wheels/`), utility-script workarounds in `kaggle_notebook.py`
- **RunPod:** `training/train.py` with `DATA_DIR`, `MODEL_PATH`, step checkpoints; no Kaggle-specific ptxas hack unless needed
- Attaching or downloading base weights: `kagglehub` + `KAGGLE_API_TOKEN` or cached model path
- `mamba_ssm` and custom CUDA wheel installation on Blackwell (see pinned discussions in scrape)
- Handling inference non-determinism (same adapter can score differently across runs)
- Local vs Kaggle score gap — validate on Kaggle when possible

### 11. Competition Intelligence (scraped docs)
- Running `scraper/competitionscraper.py` to refresh `scraper/competition-data/`
- Mining `codeanalysis.md` for high-scoring notebook patterns (GRPO, Unsloth, Tinker adapters)
- Reading `discussions.md` for metric rescoring, solver threads, training stability

---

## Skill Prioritization by Phase

| Phase | Critical Skills | Supporting Skills |
|-------|----------------|-------------------|
| **Solvers (now)** | Bit manipulation, symbol equations, cipher, audits | `test_solvers.py`, `evaluate.py --audit-solvers` |
| EDA | pandas, numpy, re | `raw-data/train.csv` |
| Data Gen | Synthetic data, trusted CoT gating | `generate_sft_data.py` |
| SFT | TRL SFTTrainer, PEFT/LoRA | Kaggle notebook or `train.py` |
| GRPO | TRL GRPOTrainer, reward functions, RLVR | Stable lr, KL, wall-clock guards |
| Submission | PEFT saving, zip packaging | Metric-aligned inference params |

---

## Key Libraries & Versions

```
transformers>=4.46.0
peft>=0.13.0
trl==0.29.1          # Pinned in Kaggle notebook / trl_wheels (GRPOTrainer)
torch>=2.4.0
bitsandbytes>=0.44.0
accelerate>=1.0.0
datasets>=3.0.0
scipy>=1.12.0
numpy>=1.26.0
pandas>=2.0.0
mamba_ssm            # Custom Blackwell wheel on Kaggle when needed
causal_conv1d        # Dependency for mamba_ssm
polars               # Used in kaggle_notebook.py for CSV load
```

---

## Key References

- [TRL GRPO Documentation](https://huggingface.co/docs/trl/grpo_trainer)
- [PEFT LoRA Guide](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [Unsloth Nemotron Guide](https://github.com/unslothai/unsloth)
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948) — GRPO for reasoning
- [NVIDIA Nemotron Submission Demo](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo)
- Competition metric: [NVIDIA Nemotron Metric on Kaggle](https://www.kaggle.com/code/metric/nvidia-nemotron-metric)
- Local scrape: `scraper/competition-data/overview.md` (evaluation parameters)

---

## Known Technical Pitfalls

| Issue | Solution |
|-------|----------|
| `CUDA error: no kernel image` on Blackwell | Custom CUDA 12.8 PyTorch / mamba wheel (discussion #681820; see scrape) |
| `mamba_ssm` import failure | Pre-built wheel in offline Kaggle environment |
| `pip install` fails on Kaggle RTX Pro 6000 | `--find-links` with `trl_wheels/` and other offline wheels |
| OOM during GRPO with 30B model | Reduce `num_generations`; 4-bit + LoRA; Unsloth patches |
| LoRA rank > 32 rejection | Always set `r=32` exactly |
| Metric updated mid-competition | Re-read metric code + pinned *Metric Update* / *Rescore* threads |
| Inference non-determinism | Same adapter can score differently; multiple submits |
| Local training → lower Kaggle scores | Validate on Kaggle when possible |
| No internet at Kaggle inference | Pre-install or bundle wheels |
| Large `submission.zip` | Rank 32 cap; verify zip contents before submit |
| **Weak solvers → bad SFT** | Fix `symbol_equation` / `bit_manipulation` before regenerating JSONL |
| **Public ~0.86 ceiling** | Expect diminishing returns; study top notebooks in `codeanalysis.md` |
| GRPO instability (grad norm, loss flatlines) | Lower lr, KL penalty, fewer generations; see recent discussions |
