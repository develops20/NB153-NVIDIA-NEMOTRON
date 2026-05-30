#!/usr/bin/env python3
"""Batch-test solvers against competition train.csv.

Loads train.csv from several candidate paths, runs classify_puzzle → solve_puzzle →
verify_answer on each row, prints per-type stats, sample failures, and a summary table.
"""

from __future__ import annotations

import csv
import os
import sys

# Repo root (parent of solvers/)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from solvers.solver import classify_puzzle, solve_puzzle, verify_answer

TRAIN_CSV_CANDIDATES = [
    os.path.join(_ROOT, "raw-data", "train.csv"),
    os.path.join(
        _ROOT,
        "competition-data",
        "nvidia-nemotron-model-reasoning-challenge",
        "train.csv",
    ),
    os.path.join(_ROOT, "data", "train.csv"),
    os.path.join(_ROOT, "competition-data", "train.csv"),
    os.path.join(_ROOT, "train.csv"),
]


def find_train_csv() -> str:
    for p in TRAIN_CSV_CANDIDATES:
        if os.path.isfile(p):
            return p
    tried = "\n  ".join(TRAIN_CSV_CANDIDATES)
    raise FileNotFoundError(
        "Could not find train.csv. Tried:\n  " + tried
    )


def prompt_snippet(prompt: str, max_len: int = 220) -> str:
    one_line = " ".join(prompt.split())
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 3] + "..."


def failure_reason(predicted: str, ground_truth: str, reasoning: str) -> str:
    pred_stripped = predicted.strip()
    if not pred_stripped:
        msg = reasoning.strip() if reasoning else ""
        return (msg[:400] + ("..." if len(msg) > 400 else "")) or "empty prediction"
    return f"pred={pred_stripped!r} gt={ground_truth!r}"


def main() -> None:
    path = find_train_csv()
    print(f"Training CSV: {path}\n")

    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    n_total = len(rows)
    per_type_total: dict[str, int] = {}
    per_type_ok: dict[str, int] = {}
    failures_by_type: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        prompt = row["prompt"]
        gt = row["answer"]
        puzzle_type = classify_puzzle(prompt)
        predicted, reasoning = solve_puzzle(prompt)
        ok = verify_answer(predicted, gt) >= 1.0

        per_type_total[puzzle_type] = per_type_total.get(puzzle_type, 0) + 1
        if ok:
            per_type_ok[puzzle_type] = per_type_ok.get(puzzle_type, 0) + 1
        else:
            bucket = failures_by_type.setdefault(puzzle_type, [])
            if len(bucket) < 3:
                bucket.append(
                    {
                        "snippet": prompt_snippet(prompt),
                        "reason": failure_reason(predicted, gt, reasoning),
                    }
                )

    # Per-type success rates and sample failures
    types_sorted = sorted(per_type_total.keys(), key=lambda t: (-per_type_total[t], t))
    overall_ok = sum(per_type_ok.values())

    print("=" * 72)
    print("PER-TYPE RESULTS")
    print("=" * 72)
    for t in types_sorted:
        tot = per_type_total[t]
        ok = per_type_ok.get(t, 0)
        rate = 100.0 * ok / tot if tot else 0.0
        print(f"  {t:20s}  {ok:5d} / {tot:5d}  ({rate:6.2f}%)")

    print()
    print("=" * 72)
    print("SAMPLE FAILURES (up to 3 per type)")
    print("=" * 72)
    for t in types_sorted:
        fails = failures_by_type.get(t, [])
        if not fails:
            continue
        print(f"\n--- {t} ({len(fails)} sample(s) shown) ---")
        for i, item in enumerate(fails, 1):
            print(f"  [{i}] {item['snippet']}")
            print(f"      → {item['reason']}")

    print()
    print("=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    hdr = f"{'puzzle_type':<22} {'ok':>8} {'total':>8} {'rate %':>10}"
    print(hdr)
    print("-" * len(hdr))
    for t in types_sorted:
        tot = per_type_total[t]
        ok = per_type_ok.get(t, 0)
        rate = 100.0 * ok / tot if tot else 0.0
        print(f"{t:<22} {ok:8d} {tot:8d} {rate:10.2f}")
    print("-" * len(hdr))
    overall_rate = 100.0 * overall_ok / n_total if n_total else 0.0
    print(
        f"{'ALL':<22} {overall_ok:8d} {n_total:8d} {overall_rate:10.2f}"
    )
    print()


if __name__ == "__main__":
    main()
