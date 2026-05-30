#!/usr/bin/env python3
"""Fast regression test for symbol_equation solver on train.csv."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from solvers.solver import classify_puzzle, solve_puzzle, verify_answer

TRAIN_CSV_CANDIDATES = [
    os.path.join(_ROOT, "raw-data", "train.csv"),
    os.path.join(_ROOT, "data", "train.csv"),
    os.path.join(_ROOT, "competition-data", "train.csv"),
]


def find_train_csv() -> str:
    for path in TRAIN_CSV_CANDIDATES:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("train.csv not found")


def run_audit(csv_path: str, max_rows: int | None = None) -> dict:
    per_type_total: dict[str, int] = {}
    per_type_ok: dict[str, int] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            if classify_puzzle(row["prompt"]) != "symbol_equation":
                continue
            ptype = "symbol_equation"
            pred, _ = solve_puzzle(row["prompt"])
            ok = verify_answer(pred, row["answer"]) >= 1.0
            per_type_total[ptype] = per_type_total.get(ptype, 0) + 1
            if ok:
                per_type_ok[ptype] = per_type_ok.get(ptype, 0) + 1

    total = per_type_total.get("symbol_equation", 0)
    ok = per_type_ok.get("symbol_equation", 0)
    return {
        "total": total,
        "correct": ok,
        "accuracy": ok / max(1, total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test symbol_equation solver accuracy")
    parser.add_argument("--csv", default=None, help="Path to train.csv")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="Exit 1 if accuracy drops below this (regression guard)",
    )
    args = parser.parse_args()

    csv_path = args.csv or find_train_csv()
    t0 = time.time()
    report = run_audit(csv_path, args.max_rows)
    elapsed = time.time() - t0

    acc = report["accuracy"]
    print(f"CSV: {csv_path}")
    print(f"symbol_equation: {report['correct']}/{report['total']} = {acc:.2%}")
    print(f"elapsed: {elapsed:.1f}s")

    if args.min_accuracy is not None and acc + 1e-9 < args.min_accuracy:
        print(f"FAIL: accuracy {acc:.2%} < min {args.min_accuracy:.2%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
