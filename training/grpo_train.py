#!/usr/bin/env python
"""
Nemotron-3 Nano 30B GRPO training — RunPod edition.

Continues from an SFT LoRA checkpoint and optimizes with verifiable rewards
(correctness + format), matching the competition metric helpers in solvers/.

Prerequisites:
  - SFT adapter at SFT_ADAPTER (adapter_config.json + adapter_model.safetensors)
  - sft_train.jsonl in DATA_DIR (prompt + ground-truth answers)
  - trl==0.29.1 (pip install trl_wheels/trl-0.29.1-py3-none-any.whl)
  - Same CUDA 12.8 + torch cu128 + mamba stack as train.py

Run:
    export SFT_ADAPTER=/workspace/output
    export OUTPUT_DIR=/workspace/output_grpo
    export DATA_DIR=/workspace/data
    tmux new -s grpo
    python -u grpo_train.py 2>&1 | tee logs/grpo_$(date +%Y%m%d_%H%M).log
"""

import gc
import json
import os
import sys
import time

SESSION_START = time.time()

# ─── Paths ─────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "/workspace/data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/output_grpo")
SFT_ADAPTER = os.environ.get("SFT_ADAPTER", "/workspace/output")
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
)
SFT_TRAIN_PATH = os.path.join(DATA_DIR, "sft_train.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Hyperparameters ───────────────────────────────────────────────────
LR = float(os.environ.get("GRPO_LR", "5e-6"))
NUM_GENERATIONS = int(os.environ.get("GRPO_NUM_GENERATIONS", "4"))
TEMPERATURE = float(os.environ.get("GRPO_TEMPERATURE", "0.7"))
BETA = float(os.environ.get("GRPO_BETA", "0.01"))
MAX_COMPLETION_LENGTH = int(os.environ.get("GRPO_MAX_COMPLETION_LENGTH", "1024"))
GRAD_ACCUM = int(os.environ.get("GRPO_GRAD_ACCUM", "4"))
NUM_EPOCHS = int(os.environ.get("GRPO_NUM_EPOCHS", "1"))
MAX_STEPS = int(os.environ.get("GRPO_MAX_STEPS", "0"))  # 0 = use num_train_epochs
SAVE_STEPS = int(os.environ.get("GRPO_SAVE_STEPS", "500"))
LOGGING_STEPS = int(os.environ.get("GRPO_LOGGING_STEPS", "10"))
SOLVER_BONUS = float(os.environ.get("GRPO_SOLVER_BONUS", "0.1"))
GRPO_MAX_ROWS = os.environ.get("GRPO_MAX_ROWS")  # optional cap for smoke tests
GRPO_MAX_ROWS = int(GRPO_MAX_ROWS) if GRPO_MAX_ROWS else None

print(f"[init] SFT_ADAPTER={SFT_ADAPTER}", flush=True)
print(f"[init] OUTPUT_DIR={OUTPUT_DIR} | DATA_DIR={DATA_DIR}", flush=True)
print(f"[init] model={MODEL_PATH}", flush=True)
print(
    f"[init] lr={LR} gens={NUM_GENERATIONS} temp={TEMPERATURE} beta={BETA} "
    f"max_completion={MAX_COMPLETION_LENGTH}",
    flush=True,
)

# ─── Imports ───────────────────────────────────────────────────────────
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from solvers.solver import extract_boxed_answer, solve_puzzle, verify_answer

# ─── Load base + SFT adapter ───────────────────────────────────────────
for req in ("adapter_config.json", "adapter_model.safetensors"):
    path = os.path.join(SFT_ADAPTER, req)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SFT adapter missing: {path}")

print(f"[model] loading base from {MODEL_PATH}", flush=True)
_t = time.time()
base = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"[model] base loaded in {time.time()-_t:.1f}s", flush=True)

print(f"[model] loading SFT adapter from {SFT_ADAPTER}", flush=True)
model = PeftModel.from_pretrained(base, SFT_ADAPTER, is_trainable=True)
model.print_trainable_parameters()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# ─── GRPO dataset from SFT JSONL ───────────────────────────────────────
def load_grpo_examples(path: str, max_rows: int | None = None) -> list[dict]:
    examples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            examples.append(json.loads(line))
    return examples


def build_grpo_dataset(raw_examples: list[dict]) -> Dataset:
    prompts, answers, user_prompts = [], [], []
    skipped = 0
    for ex in raw_examples:
        msgs = ex["messages"]
        prompt_text = tokenizer.apply_chat_template(
            msgs[:2], tokenize=False, add_generation_prompt=True
        )
        gt = extract_boxed_answer(msgs[2]["content"])
        if not gt:
            skipped += 1
            continue
        prompts.append(prompt_text)
        answers.append(gt)
        user_prompts.append(msgs[1]["content"])
    print(f"[data] GRPO prompts={len(prompts)} skipped={skipped}", flush=True)
    return Dataset.from_dict({
        "prompt": prompts,
        "ground_truth": answers,
        "user_prompt": user_prompts,
    })


raw_examples = load_grpo_examples(SFT_TRAIN_PATH, GRPO_MAX_ROWS)
grpo_dataset = build_grpo_dataset(raw_examples)

# ─── Reward functions ────────────────────────────────────────────────────
def reward_correctness(prompts, completions, ground_truth, user_prompt=None, **kwargs):
    rewards = []
    user_prompts = user_prompt or prompts
    for puzzle_prompt, completion, gt in zip(user_prompts, completions, ground_truth):
        predicted = extract_boxed_answer(completion)
        if predicted is None:
            rewards.append(0.0)
            continue
        score = verify_answer(predicted, gt)
        if score >= 1.0 and SOLVER_BONUS > 0:
            solver_answer, _ = solve_puzzle(puzzle_prompt)
            if solver_answer and verify_answer(predicted, solver_answer) >= 1.0:
                score = min(1.0 + SOLVER_BONUS, 1.5)
        rewards.append(float(score))
    return rewards


def reward_format(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        score = 0.0
        if "<think>" in completion and "</think>" in completion:
            score += 0.3
        if "\\boxed{" in completion:
            score += 0.7
        rewards.append(score)
    return rewards


class GRPOLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        parts = [f"step {state.global_step}"]
        for key in (
            "rewards/reward_correctness",
            "rewards/reward_format",
            "reward",
            "reward_std",
            "kl",
            "loss",
        ):
            if key in logs:
                parts.append(f"{key.split('/')[-1]}={logs[key]:.4f}")
        elapsed = time.time() - SESSION_START
        if state.global_step > 0:
            rate = elapsed / state.global_step
            remaining_steps = max(0, (state.max_steps or 0) - state.global_step)
            parts.append(f"ETA={(remaining_steps * rate)/60:.0f}min")
        print("  GRPO | " + " | ".join(parts), flush=True)


# ─── Train ─────────────────────────────────────────────────────────────
grpo_kwargs = dict(
    output_dir=OUTPUT_DIR,
    learning_rate=LR,
    num_generations=NUM_GENERATIONS,
    temperature=TEMPERATURE,
    beta=BETA,
    max_completion_length=MAX_COMPLETION_LENGTH,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    bf16=True,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    report_to="none",
    remove_unused_columns=False,
)

if MAX_STEPS > 0:
    grpo_kwargs["max_steps"] = MAX_STEPS
else:
    grpo_kwargs["num_train_epochs"] = NUM_EPOCHS

grpo_config = GRPOConfig(**grpo_kwargs)

print(f"[train] starting GRPO on {len(grpo_dataset)} examples", flush=True)
grpo_trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=grpo_dataset,
    reward_funcs=[reward_correctness, reward_format],
    processing_class=tokenizer,
    callbacks=[GRPOLoggingCallback()],
)

grpo_trainer.train()
grpo_trainer.save_model(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print(f"[save] adapter written to {OUTPUT_DIR}", flush=True)

# ─── Verify + package ──────────────────────────────────────────────────
cfg_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
with open(cfg_path) as cf:
    cfg = json.load(cf)
assert cfg.get("r", 0) <= 32, f"LoRA rank {cfg.get('r')} > 32 — submission rejected"
print(f"[verify] LoRA rank r={cfg.get('r')} OK", flush=True)

import subprocess

sub_zip = os.path.join(OUTPUT_DIR, "submission.zip")
if os.path.exists(sub_zip):
    os.remove(sub_zip)
subprocess.run(
    ["zip", "submission.zip", "adapter_config.json", "adapter_model.safetensors"],
    cwd=OUTPUT_DIR,
    check=True,
)
print(f"[package] {sub_zip}", flush=True)
print(f"[done] total {(time.time() - SESSION_START)/3600:.2f}h", flush=True)

del grpo_trainer, grpo_dataset, model
gc.collect()
torch.cuda.empty_cache()
