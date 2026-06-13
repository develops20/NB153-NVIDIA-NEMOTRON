#!/usr/bin/env python
"""
GRPO-readiness probe for an SFT/GRPO LoRA checkpoint — RunPod, single GPU.

The SFT [eval] entropy is measured on a free-running T=1.0 rollout, which a brittle model
derails into garbage — so a high number there is misleading. This probe answers the question
GRPO actually cares about: at a given sampling temperature, does the checkpoint produce
*coherent*, *varied* outputs that are *sometimes right and sometimes wrong*? Only prompt-groups
with mixed correct/incorrect give GRPO a non-zero reward std to learn from.

For each val prompt it draws SAMPLES completions at each temperature and reports, per temp:
  * coherent     — produced a \\boxed{} answer (didn't derail)
  * correct      — \\boxed{} matched ground truth
  * var-groups   — prompts whose SAMPLES are NOT all-same-correctness  <-- the reward_std>0 proxy
  * mean entropy — sanity vs the training [eval] numbers

Loads base + adapter via the SAME manual load as grpo_train.py (so it also smoke-tests that
path), and samples cache-free (token-by-token, no KV cache) so it needs NO modeling_nemotron_h.py
patch.

Run (on pod, GPU free):
    scp training/probe_checkpoint.py root@POD_IP:/workspace/
    cd /workspace
    export MODEL_PATH=/workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
    export PROBE_ADAPTER=/workspace/output       # checkpoint to inspect
    export DATA_DIR=/workspace/data
    # optional knobs:  PROBE_TEMPS=0.7,0.8,0.9  PROBE_SAMPLES=4  PROBE_N_PROMPTS=3
    python -u probe_checkpoint.py
"""

import os
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"  # before transformers/trl, like grpo_train.py

import json
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace")  # solvers package lives at /workspace/solvers on the pod
from solvers.solver import extract_boxed_answer, verify_answer

# ─── Tee all stdout to logs/probe_<ts>.log (tmux copy-paste is painful) ─
_LOG_DIR = os.environ.get("PROBE_LOG_DIR", "logs")  # relative to cwd; run from /workspace -> /workspace/logs
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_PATH = os.path.join(_LOG_DIR, time.strftime("probe_%Y%m%d_%H%M%S.log"))


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


_log_fh = open(_LOG_PATH, "a", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)
print(f"[probe] logging to {_LOG_PATH}", flush=True)

# ─── Config ────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/workspace/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
)
ADAPTER = os.environ.get("PROBE_ADAPTER", "/workspace/output")
DATA_DIR = os.environ.get("DATA_DIR", "/workspace/data")
VAL_PATH = os.path.join(DATA_DIR, "sft_val.jsonl")

N_PROMPTS = int(os.environ.get("PROBE_N_PROMPTS", "3"))
SAMPLES = int(os.environ.get("PROBE_SAMPLES", "4"))  # = GRPO num_generations (group size)
MAX_NEW_TOKENS = int(os.environ.get("PROBE_MAX_NEW_TOKENS", "220"))
MAX_SEQ_LEN = int(os.environ.get("PROBE_MAX_SEQ_LEN", "2048"))
TEMPS = [float(t) for t in os.environ.get("PROBE_TEMPS", "0.7,0.8,0.9").split(",")]
SHOW_CHARS = int(os.environ.get("PROBE_SHOW_CHARS", "320"))

print(
    f"[probe] adapter={ADAPTER} | temps={TEMPS} | n_prompts={N_PROMPTS} | samples/group={SAMPLES}",
    flush=True,
)

# ─── Load base + adapter (manual load — mirrors grpo_train.py B1) ───────
for req in ("adapter_config.json", "adapter_model.safetensors"):
    p = os.path.join(ADAPTER, req)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"adapter missing: {p}")

print(f"[probe] loading base from {MODEL_PATH}", flush=True)
_t = time.time()
base = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map={"": 0}, trust_remote_code=True, dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"[probe] base loaded in {time.time()-_t:.1f}s", flush=True)

with open(os.path.join(ADAPTER, "adapter_config.json")) as f:
    acfg = json.load(f)
lora_config = LoraConfig(
    r=acfg["r"],
    lora_alpha=acfg["lora_alpha"],
    lora_dropout=acfg.get("lora_dropout", 0.0),
    bias=acfg.get("bias", "none"),
    target_modules=acfg["target_modules"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, lora_config)

adapter_sd = load_file(os.path.join(ADAPTER, "adapter_model.safetensors"))
remapped = {}
for k, v in adapter_sd.items():
    if "lora_" in k and k.endswith(".weight"):
        remapped[k[: -len(".weight")] + ".default.weight"] = v
    else:
        remapped[k] = v
missing, unexpected = model.load_state_dict(remapped, strict=False)
missing_lora = [k for k in missing if "lora_" in k]
assert not unexpected, f"adapter keys not consumed: {unexpected[:5]} (+{len(unexpected)} total)"
assert not missing_lora, f"LoRA params left uninitialized: {missing_lora[:5]} (+{len(missing_lora)} total)"
print(f"[probe] adapter loaded: {len(remapped)} tensors, all LoRA slots filled", flush=True)
model.eval()

# ─── Pick a spread of val prompts ──────────────────────────────────────
examples = []
with open(VAL_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            examples.append(json.loads(line))
if not examples:
    raise RuntimeError(f"no examples in {VAL_PATH}")
step = max(1, len(examples) // N_PROMPTS)
picked = examples[::step][:N_PROMPTS]


@torch.no_grad()
def sample(prompt, temperature):
    """Cache-free token-by-token sampling; returns (text, mean_token_entropy)."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    input_ids = enc["input_ids"].to(model.device)
    start = input_ids.shape[1]
    ents = []
    for _ in range(MAX_NEW_TOKENS):
        logits = model(input_ids).logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        logp = torch.log_softmax(logits, dim=-1)
        ents.append(float(-(probs * logp).sum()))
        nxt = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, nxt], dim=-1)
        if nxt.item() == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(input_ids[0, start:], skip_special_tokens=True)
    return text, (sum(ents) / max(1, len(ents)))


# ─── Probe: SAMPLES per (prompt, temp) ─────────────────────────────────
# agg[temp] = dict of running totals across all prompts/samples
agg = {t: {"coherent": 0, "correct": 0, "ent": [], "var_groups": 0, "total": 0} for t in TEMPS}

for i, ex in enumerate(picked):
    msgs = ex["messages"]
    gt = extract_boxed_answer(msgs[2]["content"])
    prompt = tokenizer.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
    user = msgs[1]["content"]
    print("\n" + "=" * 78, flush=True)
    print(f"[{i+1}/{len(picked)}] gt={gt!r}", flush=True)
    print("  user:", (user[:160] + " …") if len(user) > 160 else user, flush=True)
    for t in TEMPS:
        answers, corrects, ents = [], [], []
        first_text = None
        for _k in range(SAMPLES):
            text, ent = sample(prompt, t)
            if first_text is None:
                first_text = text
            pred = extract_boxed_answer(text)
            answers.append(pred)
            corrects.append(1 if (pred and verify_answer(pred, gt) >= 1.0) else 0)
            ents.append(ent)
            agg[t]["total"] += 1
            agg[t]["ent"].append(ent)
            if pred:
                agg[t]["coherent"] += 1
            agg[t]["correct"] += corrects[-1]
        # reward_std>0 proxy: this group's correctness values are not all identical
        group_has_variance = len(set(corrects)) > 1
        if group_has_variance:
            agg[t]["var_groups"] += 1
        n_distinct = len({a for a in answers if a is not None})
        flag = "  <-- reward variance" if group_has_variance else ""
        print(
            f"  T={t}: correct {sum(corrects)}/{SAMPLES} | distinct answers {n_distinct} "
            f"| mean-ent {sum(ents)/len(ents):.3f}{flag}",
            flush=True,
        )
        snip = (first_text or "").strip().replace("\n", " ")
        print("       e.g. " + (snip[:SHOW_CHARS] + " …" if len(snip) > SHOW_CHARS else snip), flush=True)

# ─── Summary / verdict ─────────────────────────────────────────────────
print("\n" + "=" * 78, flush=True)
print(f"[probe] summary  ({len(picked)} prompts × {SAMPLES} samples per temp)", flush=True)
best_t, best_score = None, -1
for t in TEMPS:
    a = agg[t]
    tot = max(1, a["total"])
    mean_ent = sum(a["ent"]) / max(1, len(a["ent"]))
    print(
        f"  T={t}:  coherent {a['coherent']}/{tot} | correct {a['correct']}/{tot} "
        f"| var-groups {a['var_groups']}/{len(picked)} | mean entropy {mean_ent:.3f}",
        flush=True,
    )
    # a usable GRPO temp needs BOTH coherence and reward variance
    score = a["var_groups"] + a["coherent"] / tot
    if a["coherent"] / tot >= 0.5 and a["var_groups"] >= 1 and score > best_score:
        best_t, best_score = t, score

print("", flush=True)
if best_t is not None:
    print(
        f"VERDICT: T={best_t} looks usable — coherent AND has reward-variance groups. "
        f"Try GRPO at this temperature on THIS checkpoint (skip re-SFT).",
        flush=True,
    )
else:
    print(
        "VERDICT: no temperature is both coherent and reward-varied — this checkpoint can't "
        "feed GRPO. Re-SFT needed (and fix the entropy gate: add an UPPER bound + measure at "
        "the coherent temp / teacher-forced).",
        flush=True,
    )
