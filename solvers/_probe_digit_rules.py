"""One-off probe: measure how many train rows match a digit-rule function library."""

from __future__ import annotations

import csv
import itertools
import math
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from solvers.solver import classify_puzzle

TRAIN = os.path.join(
    _ROOT,
    "competition-data",
    "nvidia-nemotron-model-reasoning-challenge",
    "train.csv",
)

LINE_PAT = re.compile(r"^(\d{2})([^\d\s])(\d{2})\s*=\s*(.+)$")
QUERY_PAT = re.compile(r"result for:\s*(\d{2})([^\d\s])(\d{2})", re.IGNORECASE)


def digit_funcs(a: str, b: str) -> list[str]:
    """Generate candidate output strings for two-digit strings a,b."""
    out: list[str] = []
    seen: set[str] = set()

    def add(x: object) -> None:
        if x is None:
            return
        s = str(x)
        if s not in seen:
            seen.add(s)
            out.append(s)

    ia, ib = int(a), int(b)
    ba0, ba1 = int(a[0]), int(a[1])
    bb0, bb1 = int(b[0]), int(b[1])

    # --- numeric ---
    add(ia + ib)
    add(abs(ia - ib))
    add(ia * ib)
    if ib:
        add(ia // ib)
    if ia:
        add(ib // ia)
    add(math.gcd(ia, ib))
    g = math.gcd(ia, ib)
    if g:
        add(ia * ib // g)
    add(max(ia, ib))
    add(min(ia, ib))
    add(ia % ib if ib else None)
    add(ib % ia if ia else None)
    add(ia**2)
    add(ib**2)
    add(ia + ib if ia >= ib else ib - ia)
    add(ia * ib // 100 + ia * ib % 100)

    # --- string concat / reorder ---
    add(a + b)
    add(b + a)
    add(a[0] + a[1] + b[0] + b[1])
    add(b[0] + b[1] + a[0] + a[1])
    add(a[0] + b[0] + a[1] + b[1])
    add(b[0] + a[0] + b[1] + a[1])
    add(a[0] + b[1] + b[0] + a[1])
    add(b[0] + a[1] + a[0] + b[1])
    add(a[1] + b[0] + a[0] + b[1])
    add(b[1] + a[0] + b[0] + a[1])

    # --- digit-wise (2+2 char) ---
    add(str(ba0 + bb0) + str(ba1 + bb1))
    add(str(abs(ba0 - bb0)) + str(abs(ba1 - bb1)))
    add(str(ba0 * bb0) + str(ba1 * bb1))
    add(str(ba0 + bb0) + str(abs(ba1 - bb1)))
    add(str(max(ba0, bb0)) + str(max(ba1, bb1)))
    add(str(min(ba0, bb0)) + str(min(ba1, bb1)))
    add(str(ba0 + bb0) + str(bb1 + ba1))
    add(str(ba0 * bb0) + str(ba1 * bb1))

    # --- sorted / reversed ---
    add("".join(sorted(a + b)))
    add("".join(sorted(a + b, reverse=True)))
    add(a[::-1] + b[::-1])
    add(b[::-1] + a[::-1])

    # sum of all four digits as string
    add(sum(int(c) for c in a + b))

    # permutations of 4 digit characters (each used once) -> 24 strings
    for perm in itertools.permutations([a[0], a[1], b[0], b[1]], 4):
        add("".join(perm))

    return out


def extract_digit_puzzle(prompt: str) -> tuple[list[tuple[str, str, str, str]], tuple[str, str, str]] | None:
    rules: list[tuple[str, str, str, str]] = []
    for line in prompt.splitlines():
        m = LINE_PAT.match(line.strip())
        if m:
            rules.append((m.group(1), m.group(2), m.group(3), m.group(4).strip()))
    qm = QUERY_PAT.search(prompt)
    if not qm or not rules:
        return None
    return rules, (qm.group(1), qm.group(2), qm.group(3))


def main() -> None:
    total = 0
    digit_format = 0
    solved = 0
    for row in csv.DictReader(open(TRAIN, newline="", encoding="utf-8")):
        if classify_puzzle(row["prompt"]) != "symbol_equation":
            continue
        total += 1
        ex = extract_digit_puzzle(row["prompt"])
        if not ex:
            continue
        rules, q = ex
        digit_format += 1
        a0, _, b0 = q
        common: set[str] | None = None
        for A, _op, B, rhs in rules:
            opts = {s for s in digit_funcs(A, B) if s == rhs}
            if not opts:
                common = None
                break
            common = opts if common is None else common & opts
        if not common:
            continue
        pred = min(common)
        if pred == row["answer"]:
            solved += 1

    print("symbol_equation rows:", total)
    print("digit line+query format:", digit_format)
    print("solved with digit_funcs intersection:", solved)


if __name__ == "__main__":
    main()
