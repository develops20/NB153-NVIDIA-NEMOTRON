# === CELL 1 — must run first ===
import os, shutil, site

# 1. ptxas-blackwell (underscore path)
_ptxas_src = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/triton/backends/nvidia/bin/ptxas-blackwell"
_ptxas_dst = "/tmp/ptxas-blackwell"
assert os.path.exists(_ptxas_src), f"ptxas not found at {_ptxas_src}"
if not os.path.exists(_ptxas_dst):
    shutil.copy2(_ptxas_src, _ptxas_dst)
    os.chmod(_ptxas_dst, 0o755)
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = _ptxas_dst

# 2. Cutlass (underscore path)
cutlass_pkg_path = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/nvidia_cutlass_dsl/python_packages/"
assert os.path.isdir(cutlass_pkg_path + "cutlass"), f"cutlass dir missing at {cutlass_pkg_path}"
site.addsitedir(cutlass_pkg_path)

# 3. Verify
import cutlass
print("cutlass OK:", cutlass.__file__)
import mamba_ssm
print("mamba_ssm OK:", mamba_ssm.__file__)
import subprocess
out = subprocess.run([_ptxas_dst, "--version"], capture_output=True, text=True)
print("ptxas-blackwell:", out.stdout.strip().split("\n")[-1])


# =====================================================================
# CELL 2 — Inference. Assumes CELL 1 already ran (cutlass + ptxas paths set).
#
# Inputs attached to the notebook:
#   - Model:    metric/nemotron-3-nano-30b-a3b-bf16
#   - Dataset:  nicholas33/nemotron-sft-adapter-v1
#   - Competition data: nvidia-nemotron-model-reasoning-challenge (auto-attached)
# =====================================================================

import os, re, json, time
import torch
import polars as pl
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ─── Paths ─────────────────────────────────────────────────────────────
BASE_MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
ADAPTER_PATH    = "/kaggle/input/datasets/nicholas33/nemotron-sft-adapter-v1"
TEST_CSV_PATH   = "/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/test.csv"
OUTPUT_CSV      = "/kaggle/working/submission.csv"

# Generation settings
MAX_NEW_TOKENS = 512
MAX_INPUT_LEN  = 4096

# ─── Sanity-check inputs exist ─────────────────────────────────────────
for p in [BASE_MODEL_PATH, ADAPTER_PATH, TEST_CSV_PATH]:
    assert os.path.exists(p), f"Missing input: {p}"

# ─── System prompt (must match SFT training) ───────────────────────────
SYSTEM_PROMPT = (
    "You are a systematic reasoning assistant. For each puzzle, carefully "
    "analyze the examples to discover the underlying rule, show your reasoning "
    "step by step inside <think>...</think> tags, and always place your final "
    "answer inside \\boxed{}. Do not include \\boxed{} anywhere else in your response."
)

# ─── Load tokenizer + base model ───────────────────────────────────────
print("Loading tokenizer ...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

print("Loading base model (this takes a few minutes) ...", flush=True)
t0 = time.time()
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
print(f"  base model loaded in {time.time()-t0:.1f}s", flush=True)

# ─── Attach LoRA structure + load saved tensors ────────────────────────
print("Attaching LoRA and loading adapter weights ...", flush=True)
with open(os.path.join(ADAPTER_PATH, "adapter_config.json")) as f:
    cfg = json.load(f)
print(f"  adapter: r={cfg['r']}, alpha={cfg['lora_alpha']}", flush=True)

lora_config = LoraConfig(
    r=cfg["r"],
    lora_alpha=cfg["lora_alpha"],
    target_modules=cfg["target_modules"],
    lora_dropout=cfg.get("lora_dropout", 0.0),
    bias=cfg.get("bias", "none"),
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base, lora_config)
model.eval()

saved = load_file(os.path.join(ADAPTER_PATH, "adapter_model.safetensors"))
model_sd = model.state_dict()
loaded = 0
for k, v in saved.items():
    candidates = [k]
    m = re.match(r"(.*\.lora_[AB])\.weight$", k)
    if m:
        candidates.append(f"{m.group(1)}.default.weight")
    for ck in candidates:
        if ck in model_sd:
            model_sd[ck].copy_(v.to(model_sd[ck].dtype).to(model_sd[ck].device))
            loaded += 1
            break

print(f"  loaded {loaded}/{len(saved)} adapter tensors", flush=True)
assert loaded == len(saved), f"Adapter loading incomplete: {loaded}/{len(saved)}"
print("  adapter ready", flush=True)

# ─── Load test data ────────────────────────────────────────────────────
test_df = pl.read_csv(TEST_CSV_PATH)
print(f"Test rows: {len(test_df)}", flush=True)
print(f"Test columns: {test_df.columns}", flush=True)
assert "id" in test_df.columns and "prompt" in test_df.columns

# ─── Inference helper ──────────────────────────────────────────────────
def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else ""

def run_one(prompt):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    response = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_boxed(response), response

# ─── Run inference ─────────────────────────────────────────────────────
predictions = []
raw_responses = []

t0 = time.time()
for i, row in enumerate(test_df.iter_rows(named=True)):
    answer, raw = run_one(row["prompt"])
    predictions.append(answer)
    raw_responses.append(raw)

    if (i + 1) % 10 == 0 or i == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(test_df) - i - 1) / rate if rate > 0 else 0
        print(f"  {i+1}/{len(test_df)}  rate={rate:.2f}/s  eta={eta/60:.1f}min", flush=True)

print(f"\nInference complete in {(time.time()-t0)/60:.1f} min", flush=True)

# ─── Diagnostics ───────────────────────────────────────────────────────
empty_count = sum(1 for p in predictions if not p)
print(f"Empty predictions (no \\boxed{{}} found): {empty_count}/{len(predictions)}", flush=True)
if empty_count > 0:
    for i, p in enumerate(predictions):
        if not p:
            print(f"\n[debug] empty prediction at row {i}, raw response was:")
            print(raw_responses[i][:500])
            break

# ─── Show a sample of all 3 raw responses (visible test is tiny) ───────
if len(predictions) <= 5:
    for i, (pred, raw) in enumerate(zip(predictions, raw_responses)):
        print(f"\n--- row {i} ---")
        print(f"  prediction: {pred!r}")
        print(f"  raw (first 300 chars): {raw[:300]}")

# ─── Write submission ──────────────────────────────────────────────────
submission = pl.DataFrame({
    "id":     test_df["id"],
    "answer": predictions,
})
submission.write_csv(OUTPUT_CSV)
print(f"\nWrote {OUTPUT_CSV} with {len(submission)} rows", flush=True)
print(submission.head(5))