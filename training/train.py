#!/usr/bin/env python
"""
Nemotron-3 Nano 30B SFT training — RunPod edition.

Differences from the Kaggle notebook:
  * No /kaggle/input or /kaggle/working paths
  * No kagglehub model download (model is already on disk)
  * No ptxas-blackwell shim, no cutlass site.addsitedir hack
  * No offline TRL wheelhouse install (GRPO disabled in this script)
  * Step-level checkpointing so a pod restart doesn't lose hours of work
  * Plain prints flushed line-by-line so `tail -f` works under nohup/tmux

Run with:
    cd /workspace
    tmux new -s train
    python -u train.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M).log
    # Ctrl+B, D to detach
"""

import gc
import json
import math
import os
import re
import sys
import time

# ─── Paths (RunPod layout) ─────────────────────────────────────────────
DATA_DIR        = os.environ.get("DATA_DIR", "/workspace/data")
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", "/workspace/output")
CHECKPOINT_DIR  = os.environ.get("CHECKPOINT_DIR", "/workspace/output/checkpoints")
MODEL_PATH      = os.environ.get(
    "MODEL_PATH",
    "/workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
)

SFT_TRAIN_PATH = os.path.join(DATA_DIR, "sft_train.jsonl")
SFT_VAL_PATH   = os.path.join(DATA_DIR, "sft_val.jsonl")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Save a checkpoint every N optimizer steps (0 disables intra-epoch saves)
SAVE_EVERY_STEPS = int(os.environ.get("SAVE_EVERY_STEPS", "200"))
KEEP_LAST_N      = int(os.environ.get("KEEP_LAST_N", "2"))

SESSION_START = time.time()
print(f"[init] data={DATA_DIR} | model={MODEL_PATH}", flush=True)
print(f"[init] output={OUTPUT_DIR} | checkpoints={CHECKPOINT_DIR} | save_every={SAVE_EVERY_STEPS}", flush=True)

# ─── Imports (heavy) ───────────────────────────────────────────────────
import torch
import polars as pl
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from solvers.solver import extract_boxed_answer, reasoning_result_matches

# ─── Hyperparameters ───────────────────────────────────────────────────
LORA_RANK    = 32
MAX_SEQ_LEN  = 2048
NUM_EPOCHS   = int(os.environ.get("NUM_EPOCHS", "1"))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", "4"))
LR           = float(os.environ.get("LR", "2e-4"))
MIN_LR       = float(os.environ.get("MIN_LR", str(LR * 0.1)))  # cosine floor = 10% of peak
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "2"))
WARMUP_RATIO = 0.05
MIN_LR_RATIO = MIN_LR / LR
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.05"))
LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", "0.1"))
TARGET_VAL_LOSS = float(os.environ.get("TARGET_VAL_LOSS", "0.0"))  # 0 = disabled; stop on entropy, not loss
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "2"))
ENTROPY_VAL_SAMPLES = int(os.environ.get("ENTROPY_VAL_SAMPLES", "16"))  # 4 was too noisy a gate
ENTROPY_MAX_NEW_TOKENS = int(os.environ.get("ENTROPY_MAX_NEW_TOKENS", "128"))
ENTROPY_TEMPERATURE = float(os.environ.get("ENTROPY_TEMPERATURE", "1.0"))
EVAL_EVERY_STEPS = int(os.environ.get("EVAL_EVERY_STEPS", "50"))
MIN_ENTROPY = float(os.environ.get("MIN_ENTROPY", "0.5"))
GOOD_ENTROPY = float(os.environ.get("GOOD_ENTROPY", "1.0"))
NUM_WORKERS  = int(os.environ.get("NUM_WORKERS", "4"))
PIN_MEMORY   = torch.cuda.is_available()
TOKEN_CACHE  = os.environ.get("TOKEN_CACHE", "")  # dir for pre-tokenized .pt caches; empty = disabled

# ─── Load model + tokenizer ────────────────────────────────────────────
print(f"[model] loading from {MODEL_PATH}", flush=True)
_t = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"[model] loaded in {time.time()-_t:.1f}s", flush=True)

# ─── LoRA ──────────────────────────────────────────────────────────────
print(f"[lora] init r={LORA_RANK}", flush=True)
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
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# ─── Resume from latest checkpoint if present ──────────────────────────
def find_latest_ckpt(d):
    if not os.path.isdir(d):
        return None
    cands = []
    for name in os.listdir(d):
        m = re.match(r"step_(\d+)$", name)
        if m and os.path.isfile(os.path.join(d, name, "adapter_config.json")):
            cands.append((int(m.group(1)), os.path.join(d, name)))
    return max(cands)[1] if cands else None

resume_path = find_latest_ckpt(CHECKPOINT_DIR)
resume_state = None
if resume_path:
    print(f"[resume] found checkpoint {resume_path}", flush=True)
    # Load adapter weights into the existing PeftModel
    model.load_adapter(resume_path, adapter_name="default", is_trainable=True)
    state_path = os.path.join(resume_path, "trainer_state.json")
    if os.path.isfile(state_path):
        with open(state_path) as f:
            resume_state = json.load(f)
        print(f"[resume] state: {resume_state}", flush=True)
else:
    print("[resume] no checkpoint — starting fresh", flush=True)

# ─── Data ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a systematic reasoning assistant. For each puzzle, carefully "
    "analyze the examples to discover the underlying rule, show your reasoning "
    "step by step inside <think>...</think> tags, and always place your final "
    "answer inside \\boxed{}. Do not include \\boxed{} anywhere else in your response."
)

USE_RAW_CSV = not os.path.exists(SFT_TRAIN_PATH)
if USE_RAW_CSV:
    print(f"[data] {SFT_TRAIN_PATH} not found — falling back to {TRAIN_CSV_PATH}", flush=True)

def build_text_from_messages(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def _metric_match(pred, gt):
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

def _extract_boxed(s):
    return extract_boxed_answer(s)

FALLBACK_MARKER = "I can identify the transformation rule"

def load_sft_data(path, drop_fallbacks=True, drop_result_mismatch=True):
    texts, raw, dropped_fb, dropped_mm = [], [], 0, 0
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            assistant = ex["messages"][2]["content"]
            if drop_fallbacks and FALLBACK_MARKER in assistant:
                dropped_fb += 1
                continue
            if drop_result_mismatch:
                boxed = _extract_boxed(assistant)
                if boxed is None:
                    dropped_mm += 1
                    continue
                if not reasoning_result_matches(assistant, boxed):
                    dropped_mm += 1
                    continue
            texts.append(build_text_from_messages(ex["messages"]))
            raw.append(ex)
    if dropped_fb: print(f"  dropped {dropped_fb} fallback examples", flush=True)
    if dropped_mm: print(f"  dropped {dropped_mm} examples (boxed/mismatch)", flush=True)
    return texts, raw

def load_raw_csv_data():
    train_df = pl.read_csv(TRAIN_CSV_PATH)
    texts = []
    for row in train_df.iter_rows(named=True):
        msgs = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": row["prompt"]},
            {"role": "assistant", "content":
                f"<think>\nAnalyzing the puzzle examples to find the pattern.\n"
                f"After careful analysis, I can determine the answer.\n</think>\n\n"
                f"The answer is \\boxed{{{row['answer']}}}"},
        ]
        texts.append(build_text_from_messages(msgs))
    return texts

if USE_RAW_CSV:
    all_texts = load_raw_csv_data()
    split = int(len(all_texts) * 0.95)
    train_texts, val_texts = all_texts[:split], all_texts[split:]
    train_raw_examples = val_raw_examples = None
else:
    print("[data] loading SFT JSONL", flush=True)
    train_texts, train_raw_examples = load_sft_data(SFT_TRAIN_PATH)
    val_texts,   val_raw_examples   = load_sft_data(SFT_VAL_PATH)

print(f"[data] train={len(train_texts)} val={len(val_texts)}", flush=True)

def _find_assistant_start(full_text, messages):
    if messages is not None:
        try:
            prefix = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
            if prefix and full_text.startswith(prefix):
                return len(prefix)
        except Exception:
            pass
    for marker in ("<|im_start|>assistant\n", "assistant\n"):
        idx = full_text.find(marker)
        if idx != -1:
            return idx + len(marker)
    return 0

def _tokenize_examples(texts, tok, max_len, raw_examples=None):
    encodings, skipped = [], 0
    for i, text in enumerate(texts):
        enc = tok(text, truncation=True, max_length=max_len, padding=False, return_tensors="pt")
        ids  = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        if ids.sum() == 0:
            skipped += 1
            continue
        labels = ids.clone()
        labels[mask == 0] = -100
        msgs = raw_examples[i]["messages"] if raw_examples and i < len(raw_examples) else None
        off = _find_assistant_start(text, msgs)
        if off > 0:
            pref = tok(text[:off], truncation=True, max_length=max_len, padding=False, return_tensors="pt")["input_ids"].squeeze(0)
            labels[:len(pref)] = -100
        encodings.append({"input_ids": ids, "attention_mask": mask, "labels": labels})
    return encodings, skipped

def _cache_meta(n_texts, max_len):
    return {"max_len": max_len, "n_texts": n_texts}

def _load_token_cache(cache_path, n_texts, max_len):
    if not cache_path or not os.path.isfile(cache_path):
        return None
    cached = torch.load(cache_path, weights_only=False)
    if cached.get("meta") == _cache_meta(n_texts, max_len):
        print(f"[tokenize] cache hit {cache_path} ({len(cached['encodings'])} examples)", flush=True)
        return cached["encodings"]
    print(f"[tokenize] cache stale {cache_path} — rebuilding", flush=True)
    return None

def _save_token_cache(cache_path, encodings, n_texts, max_len):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save({"encodings": encodings, "meta": _cache_meta(n_texts, max_len)}, cache_path)
    print(f"[tokenize] saved cache {cache_path}", flush=True)

def build_sft_dataset(texts, tok, max_len, raw_examples=None, cache_path=None):
    encodings = _load_token_cache(cache_path, len(texts), max_len) if cache_path else None
    if encodings is None:
        encodings, skipped = _tokenize_examples(texts, tok, max_len, raw_examples)
        print(f"[tokenize] {len(encodings)} kept / {skipped} skipped", flush=True)
        if cache_path:
            _save_token_cache(cache_path, encodings, len(texts), max_len)
    return SFTDataset(encodings)

class SFTDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]

def _collate(batch):
    max_len = max(x["input_ids"].size(0) for x in batch)
    pad_id = tokenizer.pad_token_id
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for x in batch:
        pad_len = max_len - x["input_ids"].size(0)
        out["input_ids"]     .append(torch.nn.functional.pad(x["input_ids"],      (0, pad_len), value=pad_id))
        out["attention_mask"].append(torch.nn.functional.pad(x["attention_mask"], (0, pad_len), value=0))
        out["labels"]        .append(torch.nn.functional.pad(x["labels"],         (0, pad_len), value=-100))
    return {k: torch.stack(v) for k, v in out.items()}

_train_cache = os.path.join(TOKEN_CACHE, f"train_len{MAX_SEQ_LEN}.pt") if TOKEN_CACHE else None
_val_cache   = os.path.join(TOKEN_CACHE, f"val_len{MAX_SEQ_LEN}.pt")   if TOKEN_CACHE else None
if TOKEN_CACHE:
    os.makedirs(TOKEN_CACHE, exist_ok=True)
    print(f"[tokenize] cache dir={TOKEN_CACHE}", flush=True)

train_dataset = build_sft_dataset(
    train_texts, tokenizer, MAX_SEQ_LEN, raw_examples=train_raw_examples, cache_path=_train_cache,
)
val_dataset = build_sft_dataset(
    val_texts, tokenizer, MAX_SEQ_LEN, raw_examples=val_raw_examples, cache_path=_val_cache,
)

_loader_kw = dict(
    batch_size=BATCH_SIZE,
    collate_fn=_collate,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=NUM_WORKERS > 0,
)
train_loader = DataLoader(train_dataset, shuffle=True,  **_loader_kw)
val_loader   = DataLoader(val_dataset,   shuffle=False, **_loader_kw)
print(f"[loader] batch={BATCH_SIZE} workers={NUM_WORKERS} pin_memory={PIN_MEMORY}", flush=True)

def _build_entropy_val_prompts(raw_examples, texts, n_samples):
    n = min(n_samples, len(texts))
    if n == 0:
        return []
    step = max(1, len(texts) // n)
    prompts = []
    for i in range(0, len(texts), step):
        if len(prompts) >= n:
            break
        if raw_examples and i < len(raw_examples):
            msgs = raw_examples[i]["messages"]
            prompts.append(
                tokenizer.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
            )
        else:
            off = _find_assistant_start(texts[i], None)
            prompts.append(texts[i][:off] if off > 0 else texts[i])
    return prompts

entropy_val_prompts = _build_entropy_val_prompts(val_raw_examples, val_texts, ENTROPY_VAL_SAMPLES)

del train_texts, val_texts, train_raw_examples, val_raw_examples
gc.collect()

_sft_ce = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)

def _sft_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return _sft_ce(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

@torch.no_grad()
def _mean_token_entropy(
    prompts,
    temperature=ENTROPY_TEMPERATURE,
    max_new_tokens=ENTROPY_MAX_NEW_TOKENS,
):
    """Mean per-token distribution entropy along a sampled rollout (do_sample=True)."""
    if not prompts:
        return float("nan")
    entropies = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = enc["input_ids"].to(model.device)
        for _ in range(max_new_tokens):
            logits = model(input_ids).logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropies.append(-(probs * log_probs).sum(dim=-1).item())
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    return sum(entropies) / len(entropies)

# ─── Optimizer + scheduler ─────────────────────────────────────────────
model.train()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY,
)

steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

def cosine_lr(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    decay = 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * decay

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr)

# Fast-forward scheduler if resuming
start_global_step = 0
start_epoch = 0
if resume_state:
    start_global_step = resume_state.get("global_step", 0)
    start_epoch       = resume_state.get("epoch", 0)
    for _ in range(start_global_step):
        scheduler.step()
    print(f"[resume] fast-forwarded scheduler to step {start_global_step}, epoch {start_epoch}", flush=True)

print(
    f"[train] {NUM_EPOCHS} epochs, ~{total_steps} optimizer steps, warmup={warmup_steps} | "
    f"lr {LR:.0e}→{MIN_LR:.0e} ({MIN_LR_RATIO:.0%} floor) | wd={WEIGHT_DECAY} | "
    f"label_smoothing={LABEL_SMOOTHING} | early_stop target={TARGET_VAL_LOSS} patience={EARLY_STOP_PATIENCE} | "
    f"eval_every={EVAL_EVERY_STEPS} min_entropy={MIN_ENTROPY} good_entropy={GOOD_ENTROPY}",
    flush=True,
)

# ─── Checkpointing helpers ─────────────────────────────────────────────
def save_checkpoint(global_step, epoch, val_loss=None, tag=None, extra_state=None):
    name = tag or f"step_{global_step}"
    path = os.path.join(CHECKPOINT_DIR, name)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    state = {"global_step": global_step, "epoch": epoch, "val_loss": val_loss}
    if extra_state:
        state.update(extra_state)
    with open(os.path.join(path, "trainer_state.json"), "w") as f:
        json.dump(state, f)
    print(f"[ckpt] saved {path}", flush=True)
    # Prune old step_* checkpoints, keep best/ and final/
    if KEEP_LAST_N > 0:
        steps = sorted(
            (int(re.match(r"step_(\d+)", n).group(1)), n)
            for n in os.listdir(CHECKPOINT_DIR)
            if re.match(r"step_\d+$", n)
        )
        for _, old in steps[:-KEEP_LAST_N]:
            old_path = os.path.join(CHECKPOINT_DIR, old)
            try:
                import shutil; shutil.rmtree(old_path)
                print(f"[ckpt] pruned {old_path}", flush=True)
            except Exception as e:
                print(f"[ckpt] prune failed for {old_path}: {e}", flush=True)

# ─── Training loop ─────────────────────────────────────────────────────
best_val_loss = float("inf")
best_healthy_val_loss = float("inf")
epochs_without_improvement = 0
if resume_state:
    best_val_loss = resume_state.get("best_val_loss", best_val_loss)
    best_healthy_val_loss = resume_state.get("best_healthy_val_loss", best_healthy_val_loss)
    epochs_without_improvement = resume_state.get("epochs_without_improvement", 0)
_t0 = time.time()
global_step = start_global_step

def _to_device(batch):
    return {k: v.to(model.device, non_blocking=PIN_MEMORY) for k, v in batch.items()}

def _compute_val_metrics():
    val_total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = _to_device(batch)
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            vloss = _sft_loss(logits, batch["labels"])
            if torch.isfinite(vloss):
                val_total += vloss.item()
    return val_total / max(1, len(val_loader)), _mean_token_entropy(entropy_val_prompts)

def _run_eval(global_step, epoch, *, at_epoch_end=False):
    """Validate, log entropy, save best, return (should_stop, reason)."""
    global best_val_loss, best_healthy_val_loss, epochs_without_improvement
    avg_val_loss, avg_entropy = _compute_val_metrics()
    print(
        f"[eval] val_loss={avg_val_loss:.4f} entropy={avg_entropy:.4f} "
        f"(sampled T={ENTROPY_TEMPERATURE})",
        flush=True,
    )
    val_improved = avg_val_loss < best_val_loss
    if val_improved:
        best_val_loss = avg_val_loss
        if at_epoch_end:
            epochs_without_improvement = 0
    elif at_epoch_end:
        epochs_without_improvement += 1

    _early_state = {
        "best_val_loss": best_val_loss,
        "best_healthy_val_loss": best_healthy_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
    }
    if avg_entropy >= GOOD_ENTROPY and avg_val_loss < best_healthy_val_loss:
        best_healthy_val_loss = avg_val_loss
        _early_state["best_healthy_val_loss"] = best_healthy_val_loss
        save_checkpoint(
            global_step, epoch + 1 if at_epoch_end else epoch,
            avg_val_loss, tag="best", extra_state=_early_state,
        )
        model.save_pretrained(OUTPUT_DIR)
        print(
            f"[best] healthy checkpoint val_loss={avg_val_loss:.4f} "
            f"entropy={avg_entropy:.4f} → {OUTPUT_DIR}",
            flush=True,
        )
    elif val_improved and avg_entropy < GOOD_ENTROPY:
        print(
            f"[best] skipped: val_loss={avg_val_loss:.4f} improved but "
            f"entropy={avg_entropy:.4f} < good {GOOD_ENTROPY}",
            flush=True,
        )

    if avg_entropy < MIN_ENTROPY:
        return True, f"entropy {avg_entropy:.4f} < floor {MIN_ENTROPY}"
    if TARGET_VAL_LOSS > 0 and avg_val_loss <= TARGET_VAL_LOSS:
        return True, f"val_loss {avg_val_loss:.4f} <= target {TARGET_VAL_LOSS}"
    if at_epoch_end and epochs_without_improvement >= EARLY_STOP_PATIENCE:
        return True, f"no val improvement for {EARLY_STOP_PATIENCE} epoch(s)"
    return False, None

stop_training = False

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    finite_batches = 0
    nonfinite_skipped = 0
    micro_in_accum = 0
    epoch_opt_steps = 0
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        batch = _to_device(batch)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        batch_loss = _sft_loss(outputs.logits, batch["labels"])

        if torch.isfinite(batch_loss):
            (batch_loss / GRAD_ACCUM).backward()
            running_loss += batch_loss.item()
            finite_batches += 1
            micro_in_accum += 1
        else:
            nonfinite_skipped += 1
            if nonfinite_skipped <= 3 or nonfinite_skipped % 100 == 0:
                print(f"  [warn] non-finite loss at batch {i}, skipping backward", flush=True)

        if micro_in_accum >= GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            micro_in_accum = 0
            epoch_opt_steps += 1

            if epoch_opt_steps % 10 == 0:
                avg = running_loss / max(1, finite_batches)
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - _t0
                rate = (global_step - start_global_step) / elapsed if elapsed > 0 else 0
                eta = (total_steps - global_step) / rate if rate > 0 else 0
                print(
                    f"  e{epoch+1}/{NUM_EPOCHS} step {epoch_opt_steps}/{steps_per_epoch} "
                    f"(global {global_step}/{total_steps}) | loss {avg:.4f} | lr {lr_now:.2e} | "
                    f"{rate:.2f} st/s | ETA {eta/60:.0f}min",
                    flush=True,
                )

            if SAVE_EVERY_STEPS and global_step % SAVE_EVERY_STEPS == 0:
                save_checkpoint(
                    global_step, epoch,
                    extra_state={
                        "best_val_loss": best_val_loss,
                        "best_healthy_val_loss": best_healthy_val_loss,
                        "epochs_without_improvement": epochs_without_improvement,
                    },
                )

            if EVAL_EVERY_STEPS and global_step % EVAL_EVERY_STEPS == 0:
                model.eval()
                should_stop, stop_reason = _run_eval(global_step, epoch)
                model.train()
                gc.collect()
                torch.cuda.empty_cache()
                if should_stop:
                    print(f"[early_stop] {stop_reason}", flush=True)
                    stop_training = True
                    break

        if stop_training:
            break

    if stop_training:
        break

    # Flush remaining grads at epoch end
    if micro_in_accum > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    if nonfinite_skipped:
        print(f"[epoch {epoch+1}] skipped {nonfinite_skipped} non-finite batches", flush=True)
    avg_train_loss = running_loss / max(1, finite_batches)

    print(f"[epoch {epoch+1}] train_loss {avg_train_loss:.4f}", flush=True)
    model.eval()
    should_stop, stop_reason = _run_eval(global_step, epoch, at_epoch_end=True)
    model.train()
    if should_stop:
        print(f"[early_stop] {stop_reason}", flush=True)
        break

    gc.collect()
    torch.cuda.empty_cache()

# ─── Final save + verify ───────────────────────────────────────────────
if best_healthy_val_loss == float("inf"):
    model.save_pretrained(OUTPUT_DIR)
    print(
        f"[final] no healthy checkpoint (entropy >= {GOOD_ENTROPY}) — "
        f"saved current adapter to {OUTPUT_DIR}",
        flush=True,
    )
else:
    print(
        f"[final] using best healthy checkpoint "
        f"(val_loss={best_healthy_val_loss:.4f}, entropy >= {GOOD_ENTROPY})",
        flush=True,
    )

cfg_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
with open(cfg_path) as cf:
    cfg = json.load(cf)
assert cfg.get("r", 0) <= 32, f"LoRA rank {cfg.get('r')} > 32 — submission rejected"
print(f"[verify] LoRA rank r={cfg.get('r')} OK", flush=True)

# Package
import subprocess
sub_zip = os.path.join(OUTPUT_DIR, "submission.zip")
if os.path.exists(sub_zip):
    os.remove(sub_zip)
subprocess.run(
    ["zip", "submission.zip", "adapter_config.json", "adapter_model.safetensors"],
    cwd=OUTPUT_DIR, check=True,
)
print(f"[package] {sub_zip}", flush=True)
print(f"[done] total {(time.time()-SESSION_START)/3600:.2f}h", flush=True)
