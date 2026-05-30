"""Evaluation script for the NVIDIA Nemotron reasoning challenge.

Loads a LoRA adapter + base model, runs inference on training data,
and reports per-type accuracy matching the competition metric.
"""

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from solvers.solver import classify_puzzle

# Match training / SFT system prompt (evaluation/inference fidelity).
SYSTEM_PROMPT = (
    "You are a systematic reasoning assistant. For each puzzle, carefully "
    "analyze the examples to discover the underlying rule, show your reasoning "
    "step by step inside <think>...</think> tags, and always place your final "
    "answer inside \\boxed{}. Do not include \\boxed{} anywhere else in your response."
)


def extract_boxed_answer(text: str) -> str | None:
    """Extract the LAST \\boxed{} answer from model output (matches competition metric)."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None


def verify_answer(predicted: str, ground_truth: str) -> float:
    """1.0 for exact match or float within 1e-2 relative tolerance, else 0.0."""
    if predicted is None:
        return 0.0
    if predicted.strip() == ground_truth.strip():
        return 1.0
    try:
        pred_f = float(predicted)
        gt_f = float(ground_truth)
        if abs(gt_f) < 1e-9:
            return 1.0 if abs(pred_f) < 1e-9 else 0.0
        rel_diff = abs(pred_f - gt_f) / (abs(gt_f) + 1e-9)
        return 1.0 if rel_diff < 0.01 else 0.0
    except (ValueError, TypeError):
        return 0.0


def evaluate_predictions(predictions: list[dict]) -> dict:
    """Compute overall and per-type accuracy."""
    per_type = defaultdict(lambda: {"correct": 0, "total": 0, "failures": []})
    overall_correct = 0
    overall_total = 0

    for pred in predictions:
        prompt = pred["prompt"]
        ground_truth = pred["ground_truth"]
        model_output = pred.get("model_output", "")
        predicted = extract_boxed_answer(model_output)

        puzzle_type = classify_puzzle(prompt)
        score = verify_answer(predicted, ground_truth)

        per_type[puzzle_type]["total"] += 1
        overall_total += 1

        if score >= 1.0:
            per_type[puzzle_type]["correct"] += 1
            overall_correct += 1
        elif len(per_type[puzzle_type]["failures"]) < 3:
            per_type[puzzle_type]["failures"].append({
                "id": pred.get("id", "?"),
                "predicted": predicted,
                "expected": ground_truth[:50],
            })

    results = {
        "overall_accuracy": overall_correct / max(1, overall_total),
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "per_type": {},
    }

    for ptype in sorted(per_type.keys()):
        info = per_type[ptype]
        results["per_type"][ptype] = {
            "accuracy": info["correct"] / max(1, info["total"]),
            "correct": info["correct"],
            "total": info["total"],
            "failures": info["failures"],
        }

    return results


def print_results(results: dict):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(
        f"\nOverall: {results['overall_correct']}/{results['overall_total']} "
        f"= {results['overall_accuracy']:.2%}"
    )
    print("\nPer-type breakdown:")
    print(f"  {'Type':<20s} {'Correct':>7s} {'Total':>7s} {'Accuracy':>10s}")
    print("  " + "-" * 46)

    for ptype, info in sorted(results["per_type"].items()):
        print(
            f"  {ptype:<20s} {info['correct']:>7d} {info['total']:>7d} "
            f"{info['accuracy']:>9.2%}"
        )

    print("\nFailure examples:")
    for ptype, info in sorted(results["per_type"].items()):
        if info["failures"]:
            print(f"\n  --- {ptype} ---")
            for f in info["failures"]:
                print(f"    ID={f['id']}: predicted={f['predicted']}, expected={f['expected']}")


def evaluate_from_csv(
    csv_path: str,
    adapter_path: str | None = None,
    max_samples: int | None = None,
):
    """Full evaluation pipeline: load model, run inference, compute metrics."""
    predictions = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            predictions.append({
                "id": row["id"],
                "prompt": row["prompt"],
                "ground_truth": row["answer"],
                "model_output": "",
            })

    if adapter_path:
        predictions = _run_inference(predictions, adapter_path)
    else:
        print("No adapter path provided — evaluating with solver predictions")
        predictions = _run_solver_predictions(predictions)

    results = evaluate_predictions(predictions)
    print_results(results)
    return results


def _run_solver_predictions(predictions: list[dict]) -> list[dict]:
    """Use deterministic solvers instead of model inference."""
    from solvers.solver import solve_puzzle

    for pred in predictions:
        answer, reasoning = solve_puzzle(pred["prompt"])
        pred["model_output"] = (
            f"<think>\n{reasoning}\n</think>\n\n"
            f"The answer is \\boxed{{{answer}}}"
        )
    return predictions


def _run_inference(predictions: list[dict], adapter_path: str) -> list[dict]:
    """Run model inference with the LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model...")
    base_model_path = "/kaggle/input/nemotron-3-nano-30b-a3b-bf16"
    if not os.path.exists(base_model_path):
        import kagglehub
        base_model_path = kagglehub.model_download(
            "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True
    )

    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    system_prompt = SYSTEM_PROMPT

    for i, pred in enumerate(predictions):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pred["prompt"]},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=7680,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        pred["model_output"] = generated

        if (i + 1) % 50 == 0:
            print(f"  Inference: {i+1}/{len(predictions)}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate Nemotron reasoning adapter")
    parser.add_argument(
        "--csv",
        default=os.path.join(
            os.path.dirname(__file__), "..",
            "competition-data", "nvidia-nemotron-model-reasoning-challenge", "train.csv",
        ),
        help="Path to evaluation CSV",
    )
    parser.add_argument(
        "--audit-sft",
        action="store_true",
        help="Report SFT JSONL quality (fallback %, Result vs boxed) — no GPU/model.",
    )
    parser.add_argument(
        "--sft-train",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "sft_train.jsonl"),
        help="Train JSONL for --audit-sft",
    )
    parser.add_argument(
        "--sft-val",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "sft_val.jsonl"),
        help="Val JSONL for --audit-sft",
    )
    parser.add_argument(
        "--audit-solvers",
        action="store_true",
        help="Phase 0: solver accuracy & trusted-CoT stats on --csv (no GPU/model).",
    )
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument(
        "--solver-only", action="store_true",
        help="Evaluate using deterministic solvers only (no model needed)",
    )
    args = parser.parse_args()

    if args.audit_solvers:
        from data_generation.generate_sft_data import audit_train_csv, print_audit_train_csv

        report = audit_train_csv(max_rows=args.max_samples, csv_path=args.csv)
        print_audit_train_csv(report)
        return

    if args.audit_sft:
        from data_generation.generate_sft_data import print_sft_jsonl_audit

        print_sft_jsonl_audit(args.sft_train, args.sft_val)
        return

    if args.solver_only:
        args.adapter = None

    evaluate_from_csv(args.csv, args.adapter, args.max_samples)


if __name__ == "__main__":
    main()
