########################### cell1 — Load training data
###########################

import polars as pl

train = pl.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv')
print(f"Training samples: {len(train)}")
train.head()

########################### cell2 — Model setup (from updated boilerplate + utility script)
###########################
import site
import gc
import json
import math
import os
import time

# ─── Pipeline mode configuration ───
# Quickmode: uses the **same** uploaded full sft_train.jsonl / sft_val.jsonl dataset as production,
# but only the first N **examples** (file order after JSON parse) after optional filtering — no second dataset.
# Default ON for fast Kaggle dry-runs; set env QUICKMODE=0 for a full run on all examples.
QUICKMODE = os.environ.get("QUICKMODE", "1").lower() in ("1", "true", "yes")

QUICKMODE_MAX_TRAIN = int(os.environ.get("QUICKMODE_MAX_TRAIN", "100"))
QUICKMODE_MAX_VAL = int(os.environ.get("QUICKMODE_MAX_VAL", "100"))
QUICKMODE_GRPO_MAX_STEPS = int(os.environ.get("QUICKMODE_GRPO_MAX_STEPS", "48"))

# Second stage: set ENABLE_GRPO=1 for the GRPO run. Default off for SFT-only submission first.
ENABLE_GRPO = os.environ.get("ENABLE_GRPO", "0").lower() in ("1", "true", "yes")

# Session wall-clock guard (GPU quota). Default 30h; disable early-stop with DISABLE_TIME_GUARD=1.
SESSION_START = time.time()
SESSION_LIMIT_SEC = int(os.environ.get("SESSION_LIMIT_HOURS", "30")) * 3600
TIME_RESERVE_SEC = 15 * 60  # leave 15 min for saving + packaging
DISABLE_TIME_GUARD = os.environ.get("DISABLE_TIME_GUARD", "0").lower() in ("1", "true", "yes")


def time_remaining():
    return SESSION_LIMIT_SEC - (time.time() - SESSION_START)


if QUICKMODE:
    print(
        f"*** QUICKMODE — first {QUICKMODE_MAX_TRAIN} train + first {QUICKMODE_MAX_VAL} val examples "
        f"(same full JSONL upload as production); GRPO steps cap={QUICKMODE_GRPO_MAX_STEPS} when GRPO on ***"
    )
else:
    print("*** PRODUCTION MODE — all examples in JSONL, 2 SFT epochs, full GRPO schedule when GRPO on ***")

if ENABLE_GRPO:
    print("*** ENABLE_GRPO=1 — GRPO runs after SFT ***")
else:
    print("*** ENABLE_GRPO=0 — SFT only; package adapter after supervised training ***")

# Fix: /kaggle/usr/lib is read-only, so we copy ptxas-blackwell to /tmp,
# chmod it there, and point Triton to it via the env var.
import shutil
_ptxas_src = (
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script"
    "/triton/backends/nvidia/bin/ptxas-blackwell"
)
_ptxas_dst = "/tmp/ptxas-blackwell"
if os.path.exists(_ptxas_src):
    shutil.copy2(_ptxas_src, _ptxas_dst)
    os.chmod(_ptxas_dst, 0o755)
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = _ptxas_dst

cutlass_pkg_path = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
site.addsitedir(cutlass_pkg_path)

import kagglehub
import mamba_ssm
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# ─── Hyperparameters ───
LORA_RANK = 32
MAX_SEQ_LEN = 2048
NUM_EPOCHS = 1 if QUICKMODE else 2
GRAD_ACCUM = 4
LR = 2e-4
BATCH_SIZE = 2
WARMUP_RATIO = 0.05

MODEL_PATH = kagglehub.model_download(
    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
)
OUTPUT_DIR = "/kaggle/working"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Model loaded successfully.")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─── LoRA config (same for Quickmode and production; submission must keep r<=32) ───
print(f"Initializing LoRA adapter with rank={LORA_RANK}...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=64,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable gradient checkpointing to save memory during training
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

########################### cell3 — Data preparation
###########################

SYSTEM_PROMPT = (
    "You are a systematic reasoning assistant. For each puzzle, carefully "
    "analyze the examples to discover the underlying rule, show your reasoning "
    "step by step inside <think>...</think> tags, and always place your final "
    "answer inside \\boxed{}. Do not include \\boxed{} anywhere else in your response."
)

import re as _re_sft

# ─── Load SFT data (pre-generated with CoT reasoning) ───
SFT_TRAIN_PATH = "/kaggle/input/datasets/nicholas33/nb153-nemotron-puzzle-cot-sft/sft_train.jsonl"
SFT_VAL_PATH = "/kaggle/input/datasets/nicholas33/nb153-nemotron-puzzle-cot-sft/sft_val.jsonl"
USE_RAW_CSV = not os.path.exists(SFT_TRAIN_PATH)

if USE_RAW_CSV:
    print("SFT data not found — falling back to train.csv with simple CoT template")


def build_text_from_messages(messages):
    """Build training text using the tokenizer's chat template."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def _metric_match(pred: str, gt: str) -> bool:
    """Same semantics as competition metric / solvers.verify_answer on strings."""
    p, g = pred.strip(), gt.strip()
    if p == g:
        return True
    try:
        pf, gf = float(p), float(g)
        if abs(gf) < 1e-9:
            return abs(pf) < 1e-9
        return abs(pf - gf) / (abs(gf) + 1e-9) < 0.01
    except (ValueError, TypeError):
        return False


def _extract_boxed(assistant_content: str) -> str | None:
    m = _re_sft.findall(r"\\boxed\{([^}]*)\}", assistant_content)
    return m[-1].strip() if m else None


FALLBACK_MARKER = "I can identify the transformation rule"


def load_sft_data(path, drop_fallbacks=True, drop_result_mismatch=True):
    texts = []
    raw_examples = []
    dropped_fb = 0
    dropped_mm = 0
    with open(path, "r") as f:
        for line in f:
            ex = json.loads(line)
            assistant_content = ex["messages"][2]["content"]
            if drop_fallbacks and FALLBACK_MARKER in assistant_content:
                dropped_fb += 1
                continue
            if drop_result_mismatch:
                boxed = _extract_boxed(assistant_content)
                if boxed is None:
                    dropped_mm += 1
                    continue
                results = _re_sft.findall(
                    r"(?:^|\n)Result:\s*(\S+)\s*(?:\n|$)",
                    assistant_content,
                    _re_sft.MULTILINE,
                )
                if results and not _metric_match(results[-1].strip(), boxed):
                    dropped_mm += 1
                    continue
            text = build_text_from_messages(ex["messages"])
            texts.append(text)
            raw_examples.append(ex)
    if dropped_fb:
        print(f"  Dropped {dropped_fb} fallback-CoT examples (no real reasoning)")
    if dropped_mm:
        print(f"  Dropped {dropped_mm} examples (missing \\boxed or Result vs \\boxed mismatch)")
    return texts, raw_examples


def load_raw_csv_data():
    texts = []
    for row in train.iter_rows(named=True):
        prompt = row["prompt"]
        answer = row["answer"]
        assistant_msg = (
            f"<think>\nAnalyzing the puzzle examples to find the pattern.\n"
            f"After careful analysis, I can determine the answer.\n</think>\n\n"
            f"The answer is \\boxed{{{answer}}}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_msg},
        ]
        texts.append(build_text_from_messages(messages))
    return texts


grpo_train_examples = None

if USE_RAW_CSV:
    all_texts = load_raw_csv_data()
    split_idx = int(len(all_texts) * 0.95)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    train_raw_examples = None
    val_raw_examples = None
    if QUICKMODE:
        train_texts = train_texts[:QUICKMODE_MAX_TRAIN]
        val_texts = val_texts[:QUICKMODE_MAX_VAL]
        print(
            f"  Quickmode truncated raw-CSV texts to {len(train_texts)} train / {len(val_texts)} val"
        )
else:
    print(f"Loading SFT JSONL (drop fallbacks + Result/\\boxed mismatches)...")
    train_texts, train_raw_examples = load_sft_data(SFT_TRAIN_PATH)
    val_texts, val_raw_examples = load_sft_data(SFT_VAL_PATH)
    if QUICKMODE:
        train_texts = train_texts[:QUICKMODE_MAX_TRAIN]
        train_raw_examples = train_raw_examples[:QUICKMODE_MAX_TRAIN]
        val_texts = val_texts[:QUICKMODE_MAX_VAL]
        val_raw_examples = val_raw_examples[:QUICKMODE_MAX_VAL]
        print(
            f"  Quickmode: using first {len(train_texts)} train / {len(val_texts)} val examples "
            f"(from uploaded JSONL order)"
        )
    grpo_train_examples = list(train_raw_examples)

print(f"Training examples: {len(train_texts)}")
print(f"Validation examples: {len(val_texts)}")
print(f"Example text (first 500 chars):\n{train_texts[0][:500]}")


def _find_assistant_start(full_text: str, messages: list[dict] | None) -> int:
    """Return the character offset where the assistant turn begins in the
    formatted ``full_text``.  We build the prompt-only prefix (system + user
    with a generation prompt) and measure its length; if that fails we fall
    back to scanning for common assistant-turn markers.
    """
    if messages is not None:
        try:
            prefix = tokenizer.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True
            )
            if prefix and full_text.startswith(prefix):
                return len(prefix)
        except Exception:
            pass

    for marker in ("<|im_start|>assistant\n", "assistant\n"):
        idx = full_text.find(marker)
        if idx != -1:
            return idx + len(marker)
    return 0


class SFTDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, raw_examples=None):
        self.encodings = []
        skipped = 0
        for i, text in enumerate(texts):
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)

            if ids.sum() == 0:
                skipped += 1
                continue

            labels = ids.clone()
            labels[mask == 0] = -100

            msgs = raw_examples[i]["messages"] if raw_examples and i < len(raw_examples) else None
            assistant_char_offset = _find_assistant_start(text, msgs)
            if assistant_char_offset > 0:
                prefix_ids = tokenizer(
                    text[:assistant_char_offset],
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_tensors="pt",
                )["input_ids"].squeeze(0)
                labels[:len(prefix_ids)] = -100

            self.encodings.append({
                "input_ids": ids,
                "attention_mask": mask,
                "labels": labels,
            })

        print(f"Tokenized {len(self.encodings)} examples (skipped {skipped})")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def _collate_and_pad(batch):
    max_len = max(x["input_ids"].size(0) for x in batch)
    pad_id = tokenizer.pad_token_id
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for x in batch:
        length = x["input_ids"].size(0)
        pad_len = max_len - length
        out["input_ids"].append(torch.nn.functional.pad(x["input_ids"], (0, pad_len), value=pad_id))
        out["attention_mask"].append(torch.nn.functional.pad(x["attention_mask"], (0, pad_len), value=0))
        out["labels"].append(torch.nn.functional.pad(x["labels"], (0, pad_len), value=-100))
    return {k: torch.stack(v) for k, v in out.items()}


train_dataset = SFTDataset(train_texts, tokenizer, MAX_SEQ_LEN, raw_examples=train_raw_examples)
val_dataset = SFTDataset(val_texts, tokenizer, MAX_SEQ_LEN, raw_examples=val_raw_examples)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate_and_pad)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate_and_pad)

del train_texts, val_texts, train_raw_examples, val_raw_examples
gc.collect()

########################### cell4 — Training loop with cosine scheduler and validation
###########################

model.train()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=0.01,
)

total_steps = math.ceil(len(train_loader) / GRAD_ACCUM) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)


def cosine_lr(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr)

print(
    f"Training: {NUM_EPOCHS} epochs, ~{total_steps} optimizer steps, "
    f"{len(train_dataset)} examples, warmup={warmup_steps} steps"
)

best_val_loss = float("inf")
_t0 = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        running_loss += outputs.loss.item()

        if (i + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step = (i + 1) // GRAD_ACCUM
            steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
            global_step = epoch * steps_per_epoch + step
            if step % 10 == 0:
                avg = running_loss / (i + 1)
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - _t0
                rate = global_step / elapsed if elapsed > 0 else 0
                remaining = (total_steps - global_step) / rate if rate > 0 else 0
                print(
                    f"  epoch {epoch+1} | step {step}/{steps_per_epoch} "
                    f"(global {global_step}/{total_steps}) | "
                    f"avg_loss {avg:.4f} | lr {lr_now:.2e} | "
                    f"{rate:.2f} steps/s | ETA {remaining/60:.0f}min"
                )

    if (i + 1) % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss_total += outputs.loss.item()
    avg_val_loss = val_loss_total / max(1, len(val_loader))

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} — "
        f"train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(OUTPUT_DIR)
        print(f"  Saved best model (val_loss={best_val_loss:.4f})")

    gc.collect()
    torch.cuda.empty_cache()

########################### cell5 — Save SFT adapter
###########################

if best_val_loss == float("inf"):
    model.save_pretrained(OUTPUT_DIR)
    print(f"No validation improvement recorded — saved final adapter to {OUTPUT_DIR}")
else:
    print(f"Using best checkpoint already saved (val_loss={best_val_loss:.4f})")
print("SFT stage complete.")

########################### cell6 — GRPO RL Stage
###########################

import re as _re
import subprocess as _sp
from transformers import TrainerCallback

TRL_WHEEL_DIR = "/kaggle/input/datasets/nicholas33/nb153-trl-0291-offline-wheelhouse/wheels"
_sp.check_call(["pip", "install", "-q", "--no-index", "--find-links", TRL_WHEEL_DIR, "--no-deps", "trl==0.29.1"])
from trl import GRPOTrainer, GRPOConfig
print(f"trl 0.29.1 installed from {TRL_WHEEL_DIR}")


def extract_boxed_answer(text):
    matches = _re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else None


def reward_correctness(prompts, completions, ground_truth, **kwargs):
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        predicted = extract_boxed_answer(completion)
        if predicted is None:
            rewards.append(0.0)
            continue
        p, g = predicted.strip(), gt.strip()
        if p == g:
            rewards.append(1.0)
            continue
        if _re.fullmatch(r'[01]{8}', p) and _re.fullmatch(r'[01]{8}', g):
            rewards.append(0.0)
            continue
        try:
            rel_diff = abs(float(p) - float(g)) / (abs(float(g)) + 1e-9)
            rewards.append(1.0 if rel_diff < 0.01 else 0.0)
        except (ValueError, TypeError):
            rewards.append(0.0)
    return rewards


def reward_format(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        score = 0.0
        if '<think>' in completion and '</think>' in completion:
            score += 0.3
        if '\\boxed{' in completion:
            score += 0.7
        rewards.append(score)
    return rewards


def build_grpo_dataset(raw_examples):
    """Build a HuggingFace Dataset with 'prompt' and 'ground_truth' columns."""
    from datasets import Dataset as HFDataset
    prompts = []
    answers = []
    for ex in raw_examples:
        msgs = ex["messages"]
        prompt_text = tokenizer.apply_chat_template(
            msgs[:2], tokenize=False, add_generation_prompt=True
        )
        answer_content = msgs[2]["content"]
        gt = extract_boxed_answer(answer_content)
        if gt:
            prompts.append(prompt_text)
            answers.append(gt)
    return HFDataset.from_dict({"prompt": prompts, "ground_truth": answers})


class GRPOLoggingCallback(TrainerCallback):
    """Log reward statistics and time estimates during GRPO training."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        remaining = time_remaining()
        parts = [f"step {state.global_step}"]
        for key in ["rewards/reward_correctness", "rewards/reward_format",
                     "reward", "reward_std", "kl", "loss"]:
            if key in logs:
                parts.append(f"{key.split('/')[-1]}={logs[key]:.4f}")
        elapsed = time.time() - SESSION_START
        if state.global_step > 0:
            rate = elapsed / state.global_step
            eta_steps = (state.max_steps - state.global_step) * rate
            parts.append(f"ETA={min(eta_steps, remaining)/60:.0f}min")
        parts.append(f"remaining={remaining/60:.0f}min")
        print("  GRPO | " + " | ".join(parts))

class TimeGuardCallback(TrainerCallback):
    """Stop GRPO before the session expires (unless DISABLE_TIME_GUARD is set)."""
    def on_step_end(self, args, state, control, **kwargs):
        if DISABLE_TIME_GUARD:
            return control
        remaining = time_remaining()
        if remaining < TIME_RESERVE_SEC:
            print(
                f"\n*** TIME GUARD: {remaining/60:.1f} min remaining "
                f"(reserve={TIME_RESERVE_SEC/60:.0f} min) — stopping GRPO early ***"
            )
            control.should_training_stop = True
        return control


if ENABLE_GRPO and not USE_RAW_CSV and grpo_train_examples:
    print(f"\n=== Starting GRPO RL Stage (time remaining: {time_remaining()/60:.0f} min) ===")

    raw_for_grpo = list(grpo_train_examples)

    grpo_dataset = build_grpo_dataset(raw_for_grpo)
    print(f"GRPO dataset: {len(grpo_dataset)} examples")

    if QUICKMODE:
        grpo_config = GRPOConfig(
            output_dir="/kaggle/working/grpo_adapter",
            learning_rate=5e-6,
            num_generations=4,
            temperature=0.7,
            beta=0.01,
            max_completion_length=1024,
            max_steps=QUICKMODE_GRPO_MAX_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            bf16=True,
            logging_steps=10,
            save_steps=max(50, QUICKMODE_GRPO_MAX_STEPS),
        )
    else:
        grpo_config = GRPOConfig(
            output_dir="/kaggle/working/grpo_adapter",
            learning_rate=5e-6,
            num_generations=4,
            temperature=0.7,
            beta=0.01,
            max_completion_length=1024,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            bf16=True,
            logging_steps=10,
            save_steps=500,
        )

    _grpo_callbacks = [GRPOLoggingCallback()]
    if not DISABLE_TIME_GUARD:
        _grpo_callbacks.insert(0, TimeGuardCallback())

    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=grpo_dataset,
        reward_funcs=[reward_correctness, reward_format],
        processing_class=tokenizer,
        callbacks=_grpo_callbacks,
    )

    grpo_trainer.train()
    grpo_trainer.save_model("/kaggle/working/grpo_adapter")

    model.save_pretrained(OUTPUT_DIR)
    print("GRPO training complete — adapter saved.")

    del grpo_trainer, grpo_dataset, raw_for_grpo
    gc.collect()
    torch.cuda.empty_cache()
elif ENABLE_GRPO and not USE_RAW_CSV:
    print("GRPO skipped — no SFT train examples in memory (empty JSONL after filters?)")
elif ENABLE_GRPO:
    print("GRPO skipped — SFT data not available (using raw CSV fallback)")
else:
    print("GRPO disabled — set ENABLE_GRPO = True to enable")

########################### cell7 — Verify and package submission
###########################

# If we skipped SFT and GRPO didn't run (e.g. missing SFT data), ensure
# the loaded adapter is saved to OUTPUT_DIR so submission packaging works.
if not os.path.exists(os.path.join(OUTPUT_DIR, "adapter_config.json")):
    print("No adapter in OUTPUT_DIR yet — saving current model...")
    model.save_pretrained(OUTPUT_DIR)

config_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
with open(config_path, "r") as cf:
    config = json.load(cf)
assert config.get("r", 0) <= 32, f"LoRA rank {config.get('r')} > 32 — will be rejected!"
print(f"LoRA rank verified: r={config.get('r')}")

import subprocess

subprocess.run(
    [
        "zip",
        "-m",
        "submission.zip",
        "adapter_config.json",
        "adapter_model.safetensors",
    ],
    cwd=OUTPUT_DIR,
    check=True,
)

elapsed_total = time.time() - SESSION_START
print(f"Done. Total wall time: {elapsed_total/3600:.1f} hours")
