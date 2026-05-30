# NVIDIA Nemotron Model Reasoning Challenge — Competition Plan

**Last updated:** 2026-05-29 (from live scrape in `scraper/competition-data/`)

**Deadline:** 2026-06-15 (final submission) | **Entry / team merge:** 2026-06-08  
**Prize pool:** $106,388 + 8× DGX Spark systems  
**Public leaderboard bar (May 2026):** ~**0.86** on Models tab; community threads on breaking the **0.86 ceiling**; notebook titles cite **0.87** experiments  
**Our validated pipeline score:** **0.50** (end-to-end SFT submit); **next focus:** solver accuracy → better SFT/GRPO data  
**Stretch target:** competitive with top public (~0.86+), not necessarily 0.95+ on private LB  

**Participation (Kaggle, May 2026):** 3,644 teams · 16,313 entrants · 49,441 submissions · **1,020** public-scored notebooks  

---

## Repository layout

| Path | Role |
|------|------|
| `raw-data/train.csv`, `raw-data/test.csv` | Official competition CSVs (static; not re-scraped) |
| `scraper/competition-data/*.md` | **Source of truth** for rules, overview, data description, discussions, models, code index |
| `data/sft_train.jsonl`, `data/sft_val.jsonl` | Generated SFT dataset |
| `solvers/` | Deterministic puzzle solvers (accuracy work in progress) |
| `data_generation/generate_sft_data.py` | Synthetic CoT + audits |
| `training/kaggle_notebook.py` | Kaggle training notebook (cells) |
| `training/train.py` | RunPod / local training |
| `evaluation/evaluate.py` | Local metric + `--audit-solvers` / `--audit-sft` |

Refresh competition intel: `cd scraper && python competitionscraper.py nvidia-nemotron-model-reasoning-challenge`

---

## Core Idea

All 6 puzzle types are **algorithmically solvable with perfect accuracy**. Every puzzle can be solved by a deterministic Python function. This gives two decisive advantages:

1. **Unlimited synthetic training data** — generate thousands of examples with perfect chain-of-thought reasoning, tuned exactly to each puzzle type.
2. **Perfect verifiers** — each puzzle solver becomes a GRPO reward function (reinforcement learning with verifiable rewards — the same approach behind DeepSeek-R1).

**The plan:** SFT on programmatically-generated CoT data → GRPO with verifiable rewards → near-perfect accuracy.

**Competitive landscape (updated 2026-05-29):** Public scores cluster at **~0.86** (base model + top community adapters on the Models tab). GRPO, Unsloth, Tinker-style adapter packaging, and **high-accuracy symbolic solvers** (e.g. discussion threads on 97%+ solver pipelines) are widespread among top notebooks. Midpoint Open Progress prize (**April 9**) has passed. The differentiator is **solver correctness on `symbol_equation` and `bit_manipulation`**, volume/quality of verified CoT, and stable GRPO — not merely having solvers or GRPO at all.

---

## The 6 Puzzle Types

### 1. `bit_manipulation`
8-bit binary → 8-bit binary via a hidden reversible operation (XOR, NOT, shift, rotate, AND, OR, Majority, Choice).

**Solver:** Given 7–10 input/output pairs, enumerate candidate single operations and compositions until one is consistent with all examples. Apply to query input.

**CoT example:**
> "Step 1: Test rotate_left(1)... matches example 1 but fails example 3. Step 2: Test XOR(0xFF)... matches all 8 examples. Apply: XOR(00110100, 11111111) = 11001011. → \boxed{11001011}"

### 2. `cipher` (word substitution)
Word-level substitution cipher. Each unique ciphertext word maps to a unique plaintext word; examples provide the mapping.

**Solver:** Build a bijective word→word lookup table from examples. Apply word-by-word to query.

**CoT example:**
> "Mapping: ucoov→queen, pwgtfyoqg→discovers, vorq→near, yrjjoe→valley. Query: trb wzrswvog hffk → cat imagines book. → \boxed{cat imagines book}"

### 3. `roman_numeral`
Convert a decimal integer to Roman numerals (standard algorithm).

**Solver:** Greedy subtraction from {M=1000, CM=900, D=500, CD=400, C=100, XC=90, L=50, XL=40, X=10, IX=9, V=5, IV=4, I=1}.

**CoT example:**
> "38: 38≥X=10 → X, rem 28. 28≥X → X, rem 18. 18≥X → X, rem 8. 8≥V=5 → V, rem 3. 3≥I → III. Result: XXXVIII → \boxed{XXXVIII}"

### 4. `gravity`
Given (t, d) observations following d = 0.5 × g × t², find g, then predict distance for new t.

**Solver:** Compute g = 2d/t² for each example pair, average, then d_new = 0.5 × g_avg × t_new². Round to 2 decimal places.

**CoT example:**
> "g from examples: 2×14.92/1.37²=15.90, 2×144.96/4.27²=15.91. avg g=15.90. d=0.5×15.90×4.41²=154.62 → \boxed{154.62}"

### 5. `unit_conversion`
A hidden linear scale factor. Given (input_value, output_value) pairs, find the ratio and apply to query.

**Solver:** ratio = mean(output/input) across all examples. Apply query × ratio, round to 2 decimal places.

**CoT example:**
> "Ratios: 6.69/10.08=0.664, 11.83/17.83=0.663, 23.79/35.85=0.663. avg=0.663. 25.09×0.663=16.63 → \boxed{16.65}"

### 6. `symbol_equation`
Symbol-level algebra. Substitution rules are given as equations; solve the query expression.

**Solver:** Parse LHS=RHS pairs into a symbol→value mapping via constraint propagation. Substitute into query.

**CoT example:**
> "From rules: `=A, [=B, !=C... Apply to [[-!': B·B·C·' → \boxed{@&}"

---

## Strategy: SFT → GRPO

```
train.csv + synthetic data (20K examples)
         ↓
  [Stage 1: SFT — ~4–6h on RTX 6000]
  Nemotron-3-Nano + LoRA (rank 32)
  Trained on: high-quality CoT per puzzle type
  Goal: strong val / solver-audit accuracy, correct <think>/\boxed{} format
         ↓
  [Stage 2: GRPO — ~6–8h on RTX 6000 / RunPod]
  Continues from SFT checkpoint
  Reward: programmatic verifier (1.0 correct / 0.1 format / 0.0 wrong)
  Goal: close gap toward public ~0.86+ (private LB unknown)
         ↓
  submission.zip  (adapter_config.json + adapter_model.safetensors)
```

---

## Data Generation Pipeline

### Phase A — EDA & Solver Validation
1. Parse all `raw-data/train.csv` rows, classify by puzzle type (keyword detection on prompt).
2. Implement and unit-test each Python solver (`solvers/test_solvers.py`, `evaluation/evaluate.py --audit-solvers`).
3. Validate 100% accuracy on `gravity`, `unit_conversion`, `roman_numeral`; high accuracy on `cipher`, `symbol_equation`; raise `bit_manipulation` from ~90% and `symbol_equation` from ~42% (current audit baseline).

### Phase B — Synthetic CoT Dataset
Generate **2,000–5,000 examples per puzzle type** (target: 20K total):

```json
{
  "messages": [
    {"role": "system", "content": "You are a systematic reasoning assistant..."},
    {"role": "user", "content": "<puzzle prompt>"},
    {"role": "assistant", "content": "<think>\n<step-by-step reasoning>\n</think>\n\nThe answer is \\boxed{<answer>}"}
  ]
}
```

Synthetic generators:
- **Gravity:** Random g ∈ [2.0, 20.0], random t values, 4–6 example pairs + query
- **Unit conversion:** Random ratio ∈ [0.5, 5.0], 4–6 example pairs
- **Roman numerals:** Random integers 1–3999, 3–5 examples + query
- **Cipher:** Fixed Alice-in-Wonderland wordlist, random permutation mappings
- **Bit manipulation:** Random operation from known set, 8–10 example pairs
- **Symbol equations:** Random symbol substitution rules

Save: `data/sft_train.jsonl` (90%) + `data/sft_val.jsonl` (10%)

### Phase C — GRPO Reward Functions

```python
def extract_boxed_answer(text: str) -> str | None:
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else None

def reward_correctness(completions, ground_truths, **kwargs) -> list[float]:
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        predicted = extract_boxed_answer(completion)
        if predicted == gt:
            rewards.append(1.0)
        else:
            try:
                rel_diff = abs(float(predicted) - float(gt)) / (abs(float(gt)) + 1e-9)
                rewards.append(1.0 if rel_diff < 0.01 else 0.0)
            except:
                rewards.append(0.0)
    return rewards

def reward_format(completions, **kwargs) -> list[float]:
    # 0.3 for <think>...</think>, 0.7 for \boxed{}
    ...
```

---

## Training Configuration

### Model & LoRA
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/kaggle/input/nemotron-3-nano-30b-a3b-bf16",
    max_seq_length=4096,
    dtype=None,       # auto BF16
    load_in_4bit=False,  # BF16 for best quality; switch to True if OOM
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
```

### SFT Hyperparameters
| Parameter | Value |
|-----------|-------|
| epochs | 3 |
| batch size | 2 (per device) |
| gradient accumulation | 8 |
| learning rate | 2e-4 |
| scheduler | cosine |
| optimizer | adamw_8bit |
| max seq length | 4096 |
| packing | True |

### GRPO Hyperparameters
| Parameter | Value |
|-----------|-------|
| epochs | 1 |
| learning rate | 5e-6 |
| num_generations (group G) | 8 |
| temperature (rollout) | 0.7 |
| kl_coef | 0.01 |
| max new tokens | 1024 |

### Inference (submission)
Matches competition metric exactly:
- temperature=0.0 (greedy), top_p=1.0, max_tokens=7680, max_model_len=8192

---

## System Prompt for Inference
```
You are a systematic reasoning assistant. Analyze the puzzle carefully, identify the hidden rule from the examples, show your reasoning step by step inside <think>...</think> tags, and always place your final answer inside \boxed{}. Do not include \boxed{} anywhere else in your response.
```

---

## Implementation Roadmap (revised 2026-05-29)

| Phase | Milestone | Target | Status |
|-------|-----------|--------|--------|
| Done | SFT JSONL pipeline, Kaggle/RunPod training scripts, submission packaging | — | **Done** |
| Done | End-to-end scored submit (SFT-only) | **0.50** | **Done** |
| **Now** | **Fix solvers** (`symbol_equation`, `bit_manipulation`); re-audit `raw-data/train.csv` | Solver audits → high 90s%+ overall | **In progress** |
| Next | Regenerate / refresh SFT with trusted CoT only; optional GRPO on Kaggle or RunPod | Beat 0.50 → toward **0.70+** then **0.86** public bar | Pending |
| Jun 8 | Entry deadline (accept rules, merge teams) | — | Upcoming |
| Jun 8–15 | Final adapter + **public notebook + write-up** (required for prizes) | Best effort LB | Upcoming |

> **~17 days to competition end (per overview scrape).** Prioritize solver fixes before another long training run.

---

## Deliverables

```
project/
├── raw-data/
│   ├── train.csv            # Official training puzzles
│   └── test.csv             # Sample test prompts
├── scraper/competition-data/
│   ├── overview.md          # Rules, metric params, timeline (scraped)
│   ├── data.md, rules.md, discussions.md, models.md, codeanalysis.md
├── solvers/
│   ├── solver.py            # Classifier + dispatcher
│   ├── bit_manipulation.py, cipher.py, gravity.py, roman_numeral.py
│   ├── unit_conversion.py, symbol_equation.py
│   └── test_solvers.py      # Batch accuracy on train.csv
├── data_generation/
│   └── generate_sft_data.py # Synthetic CoT + audits
├── data/
│   ├── sft_train.jsonl
│   └── sft_val.jsonl
├── training/
│   ├── kaggle_notebook.py   # Kaggle GPU notebook
│   ├── train.py             # RunPod SFT
│   └── kaggle_inference.py
├── evaluation/
│   └── evaluate.py
└── outputs/                 # Adapters, logs, submission artifacts
```

---

## Technical Stack & Dependencies

### Core Training Libraries
| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.4+ | Base deep learning |
| `unsloth` | latest | Fast LoRA fine-tuning (2× speed, memory patches for Mamba) |
| `trl` | **0.29.1** (pinned in `trl_wheels/` / Kaggle notebook) | `SFTTrainer` + `GRPOTrainer` |
| `transformers` | 4.46+ | Model loading, tokenizer, chat templates |
| `peft` | 0.13+ | LoRA adapter management, `save_pretrained` |
| `accelerate` | 1.0+ | Distributed training, gradient checkpointing |
| `bitsandbytes` | 0.44+ | 4-bit quantization (QLoRA fallback) |
| `datasets` | 3.0+ | HuggingFace dataset handling |
| `mamba_ssm` | custom wheel | Required for Nemotron MoE/Mamba layers |
| `causal_conv1d` | custom wheel | Dependency for mamba_ssm |

### Solver / Data Libraries
| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.0+ | CSV parsing, EDA |
| `numpy` | 1.26+ | Numerical ops (gravity/unit solvers, bit ops) |
| `scipy` | 1.12+ | Least-squares fitting fallback for gravity |
| `tqdm` | latest | Progress bars during data generation |
| `re` | stdlib | Extract `\boxed{}` answers from model output |

### Install
```bash
pip install unsloth trl transformers peft accelerate bitsandbytes datasets pandas numpy scipy tqdm
# Then install mamba_ssm + causal_conv1d from Blackwell wheel (see discussion #681820)
```

### Blackwell CUDA Fix (RTX PRO 6000)
The default `mamba_ssm`/`causal_conv1d` wheels are compiled for older CUDA architectures. Install Blackwell-compatible builds per discussion #681820.

### Solver Skills Required
- **Bit manipulation:** Python `&`, `|`, `^`, `~`, `<<`, `>>`; rotation functions; brute-force op enumeration; SHA-256 Majority/Choice functions
- **Cipher:** Bijective word-map construction; OOV handling via character-level fallback
- **Gravity/Unit:** `numpy` least-squares; ratio averaging; float rounding to 2 dp
- **Roman/Symbol:** Greedy algorithm; constraint propagation

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| CUDA kernel error on Blackwell | Custom wheel from discussion #681820 |
| GRPO training instability | Low lr (5e-6), KL penalty kl_coef=0.01 |
| Bit manipulation solver coverage | Exhaustive single/double op search; fall back to top-k voting across multiple candidate ops |
| OOM during GRPO rollouts | Reduce `num_generations` to 4; enable 4-bit |
| LoRA rank > 32 rejection | Always verify `adapter_config.json` before zipping |
| Test set distribution unknown | Discussion #681793 + organizer reply confirms same 6 types in test |
| `\boxed{}` in `<think>` blocks captured as final answer | System prompt instructs to put `\boxed{}` only at end; reward_format penalizes otherwise |
| **Metric was updated (rescoring)** | Review latest metric code; verify `\boxed{}` extraction and numerical tolerance match |
| **Inference non-determinism** | Same adapter can score differently across runs; submit multiple times and take best |
| **Local vs Kaggle score gap** | Scores trained locally are lower than on Kaggle; validate on Kaggle environment |
| **Pip install with internet disabled** | Kaggle inference environment has no internet; pre-install all deps or use offline wheels |
| **Approach no longer unique** | At least one competitor using deterministic solvers + CoT; differentiate via data quality and GRPO |

---

## Prize Strategy

- **Open Progress Prize (April 9):** **Closed** — not a current focus.
- **Open Contribution Awards (3× DGX Spark):** Best data/synthetic, RL, fine-tuning — requires top **10%** final LB + form submission; document solver-backed CoT and GRPO if we reach that tier.
- **Final Leaderboard:** $25K / $15K / $5K + DGX Spark placements; realistic aim is strong improvement from **0.50**, not assuming 0.95+ without private-LB evidence.
- **Required for any prize:** Public Kaggle notebook + write-up before **June 15**.

Pinned discussions to re-read: *Metric Update*, *Rescore After Metric Update* (`scraper/competition-data/discussions.md`).

---

## Competitive Edge Summary (revised 2026-05-29)

1. **Verified solvers → trusted CoT only** — fallbacks hurt SFT; fixing `symbol_equation` and `bit_manipulation` is the highest-leverage step before retraining.
2. **Solver-verified GRPO** — still valuable if solvers are accurate; many public notebooks struggle with GRPO stability.
3. **Synthetic + train CoT dataset** — already built; refresh after solver fixes.
4. **Open Contribution / documentation** — strong narrative if solver audits and data pipeline are published clearly.

**What changed since April:** The public bar moved from ~0.80 to **~0.86**. Solver+CoT and GRPO are table stakes; winning execution requires **near-perfect solvers** and training data that matches the official metric (`\boxed{}`, 10⁻² numeric tolerance, vLLM params in `overview.md`).
