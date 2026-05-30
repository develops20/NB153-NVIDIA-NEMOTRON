#!/usr/bin/env python3
"""Fast bit_manipulation regression test."""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
from solvers.solver import classify_puzzle, verify_answer
from solvers.bit_manipulation import solve_bit_manipulation

CSV = os.path.join(_ROOT, "raw-data", "train.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test bit_manipulation solver accuracy")
    parser.add_argument("--csv", default=CSV)
    parser.add_argument("--min-accuracy", type=float, default=None)
    args = parser.parse_args()

    t0 = time.time()
    tot = ok = 0
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if classify_puzzle(row["prompt"]) != "bit_manipulation":
                continue
            tot += 1
            pred, _ = solve_bit_manipulation(row["prompt"])
            if verify_answer(pred, row["answer"]) >= 1.0:
                ok += 1
    acc = ok / max(1, tot)
    print(f"bit_manipulation: {ok}/{tot} = {acc:.4f} ({100 * acc:.2f}%)")
    print(f"elapsed: {time.time() - t0:.1f}s")
    if args.min_accuracy is not None and acc + 1e-9 < args.min_accuracy:
        print(f"FAIL: accuracy {acc:.2%} < min {args.min_accuracy:.2%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
