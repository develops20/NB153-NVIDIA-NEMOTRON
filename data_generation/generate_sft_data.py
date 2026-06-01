"""Generate SFT training data with chain-of-thought reasoning.

1. Process all train.csv rows using solvers for CoT (fallback: oracle-only minimal CoT with ground truth)
2. Generate synthetic examples for each puzzle type
3. Save as data/sft_train.jsonl and data/sft_val.jsonl using a stratified split by puzzle type

CLI:
  python generate_sft_data.py                    # full production JSONL (upload this single dataset to Kaggle)
  python generate_sft_data.py --audit-csv-only # solver stats on train.csv (no JSONL write)
  python generate_sft_data.py --train-csv-max-rows 500  # optional local limit for faster dev

Symbol equation rows use ``symbol_equation_sft_tier`` (trusted / oracle_only / exclude):
trusted → solver CoT when consistent; oracle_only → minimal GT template; exclude → omitted.

Kaggle Quickmode does **not** use a second dataset: use the full JSONL upload and let
``kaggle_notebook.py`` take the first N examples when QUICKMODE is on.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solvers.solver import classify_puzzle, solve_puzzle, verify_answer
from solvers.solver import extract_boxed_answer, reasoning_result_matches, format_boxed_answer
from solvers.roman_numeral import int_to_roman
from solvers.symbol_equation import symbol_equation_sft_tier

SYSTEM_PROMPT = (
    "You are a systematic reasoning assistant. For each puzzle, carefully "
    "analyze the examples to discover the underlying rule, show your reasoning "
    "step by step inside <think>...</think> tags, and always place your final "
    "answer inside \\boxed{}. Do not include \\boxed{} anywhere else in your response."
)

TRAIN_CSV_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "raw-data", "train.csv"),
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "competition-data",
        "nvidia-nemotron-model-reasoning-challenge",
        "train.csv",
    ),
    os.path.join(os.path.dirname(__file__), "..", "data", "train.csv"),
    os.path.join(os.path.dirname(__file__), "..", "nemotron-master", "train.csv"),
    os.path.join(os.path.dirname(__file__), "..", "train.csv"),
]


def find_train_csv(explicit: str | None = None) -> str:
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"train.csv not found: {explicit}")
        return explicit
    for path in TRAIN_CSV_CANDIDATES:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not find train.csv. Tried:\n  " + "\n  ".join(TRAIN_CSV_CANDIDATES)
    )


def make_message(prompt: str, answer: str, reasoning: str) -> dict:
    assistant_content = (
        f"<think>\n{reasoning}\n</think>\n\n"
        f"{format_boxed_answer(answer)}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def _reasoning_final_result_consistent(reasoning: str, gt: str) -> bool:
    """If the trace ends with an explicit Result line, it must match ground truth."""
    return reasoning_result_matches(reasoning, gt)


def stratified_train_val_split(
    examples: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
    min_val_per_type: int = 1,
) -> tuple[list[dict], list[dict]]:
    """Split so each puzzle type contributes ~``val_ratio`` of its rows to validation."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        ptype = classify_puzzle(ex["messages"][1]["content"])
        by_type[ptype].append(ex)
    train: list[dict] = []
    val: list[dict] = []
    for _ptype, items in sorted(by_type.items()):
        rng.shuffle(items)
        n = len(items)
        if n <= 1:
            train.extend(items)
            continue
        n_val = max(min_val_per_type, int(round(n * val_ratio)))
        n_val = min(n_val, n - 1)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def audit_train_csv(max_rows: int | None = None, csv_path: str | None = None) -> dict:
    """Phase 0: solver coverage and CoT-consistency stats on competition train.csv."""
    path = find_train_csv(csv_path)
    per_type_total: dict[str, int] = defaultdict(int)
    per_type_solver_ok: dict[str, int] = defaultdict(int)
    per_type_solver_cot: dict[str, int] = defaultdict(int)
    sym_tier_counts: dict[str, int] = defaultdict(int)
    sym_excluded = 0
    sym_gold_ok = 0
    total = 0
    solver_ok = 0
    solver_cot = 0
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            total += 1
            prompt = row["prompt"]
            gt = row["answer"]
            ptype = classify_puzzle(prompt)
            per_type_total[ptype] += 1
            solver_answer, reasoning = solve_puzzle(prompt)
            ok = verify_answer(solver_answer, gt) == 1.0
            if not ok and ptype == "symbol_equation":
                solver_answer2, reasoning2 = solve_puzzle(prompt, answer_hint=gt)
                ok2 = verify_answer(solver_answer2, gt) == 1.0
                if ok2:
                    sym_gold_ok += 1
                    ok = True
                    solver_answer = solver_answer2
                    reasoning = reasoning2
            if ok:
                solver_ok += 1
                per_type_solver_ok[ptype] += 1

            if ptype == "symbol_equation":
                tier = symbol_equation_sft_tier(prompt, solver_correct=ok)
                sym_tier_counts[tier] += 1
                if tier == "exclude":
                    sym_excluded += 1
                use_cot = (
                    ok
                    and bool(reasoning)
                    and _reasoning_final_result_consistent(reasoning, gt)
                    and (tier == "trusted" or "Symbol mapping:" in reasoning or "Cryptarithm:" in reasoning or "Gold-matched" in reasoning)
                )
            else:
                use_cot = ok and bool(reasoning) and _reasoning_final_result_consistent(reasoning, gt)

            if use_cot:
                solver_cot += 1
                per_type_solver_cot[ptype] += 1
    out = {
        "total_rows": total,
        "solver_correct": solver_ok,
        "solver_correct_rate": solver_ok / max(1, total),
        "rows_with_trusted_solver_cot": solver_cot,
        "trusted_cot_rate": solver_cot / max(1, total),
        "symbol_equation_tiers": dict(sym_tier_counts),
        "symbol_equation_excluded": sym_excluded,
        "symbol_equation_gold_conditioned": sym_gold_ok,
        "per_type": {},
    }
    for p in sorted(per_type_total.keys()):
        n = per_type_total[p]
        out["per_type"][p] = {
            "total": n,
            "solver_correct": per_type_solver_ok[p],
            "trusted_cot": per_type_solver_cot[p],
            "solver_accuracy": per_type_solver_ok[p] / max(1, n),
            "trusted_cot_rate": per_type_solver_cot[p] / max(1, n),
        }
    return out


def print_audit_train_csv(report: dict) -> None:
    print("\n=== Phase 0 — train.csv solver audit ===")
    print(f"Rows: {report['total_rows']}")
    print(
        f"Solver exact/numeric match: {report['solver_correct']} "
        f"({report['solver_correct_rate']:.2%})"
    )
    print(
        f"Trusted solver CoT (correct + consistent Result line): "
        f"{report['rows_with_trusted_solver_cot']} ({report['trusted_cot_rate']:.2%})"
    )
    if report.get("symbol_equation_tiers"):
        print("\nSymbol equation SFT tiers (train.csv):")
        tiers = report["symbol_equation_tiers"]
        for tier in ("trusted", "oracle_only", "exclude"):
            if tier in tiers:
                print(f"  {tier}: {tiers[tier]}")
        print(f"  excluded from JSONL: {report.get('symbol_equation_excluded', 0)}")
        if report.get("symbol_equation_gold_conditioned"):
            print(f"  gold-conditioned solves: {report['symbol_equation_gold_conditioned']}")
    print("\nPer type:")
    for ptype, info in sorted(report["per_type"].items()):
        print(
            f"  {ptype:<20s}  n={info['total']:<5d}  "
            f"solver_acc={info['solver_accuracy']:.2%}  trusted_cot={info['trusted_cot_rate']:.2%}"
        )


def count_jsonl_quality_issues(examples: list[dict]) -> dict:
    """Post-generation checks: fallbacks, missing boxed, Result vs boxed conflicts."""
    fallback = 0
    missing_boxed = 0
    result_mismatch = 0
    for ex in examples:
        content = ex["messages"][2]["content"]
        if "After examining all the given examples, I can identify the transformation rule." in content:
            fallback += 1
        boxed = extract_boxed_answer(content)
        if boxed is None:
            missing_boxed += 1
            continue
        if not reasoning_result_matches(content, boxed):
            result_mismatch += 1
    return {
        "n": len(examples),
        "fallback_template": fallback,
        "missing_boxed": missing_boxed,
        "result_line_vs_boxed_mismatch": result_mismatch,
    }


def print_jsonl_quality(report: dict) -> None:
    print("\n=== JSONL quality (train+val combined) ===")
    for k, v in report.items():
        print(f"  {k}: {v}")


def load_sft_jsonl(path: str) -> list[dict]:
    examples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def print_sft_jsonl_audit(train_path: str, val_path: str | None = None) -> None:
    """Report SFT quality metrics after generation (non-fallback rate, CoT consistency)."""
    train_ex = load_sft_jsonl(train_path)
    val_ex = load_sft_jsonl(val_path) if val_path and os.path.isfile(val_path) else []
    print("\n=== SFT JSONL quality audit ===")
    print(f"  Train file: {train_path} ({len(train_ex)} examples)")
    if val_ex:
        print(f"  Val file:   {val_path} ({len(val_ex)} examples)")
    q_train = count_jsonl_quality_issues(train_ex)
    print("\n  Train shard:")
    for k, v in q_train.items():
        print(f"    {k}: {v}")
    if q_train["n"]:
        fb_pct = 100.0 * q_train["fallback_template"] / q_train["n"]
        print(f"    approx_non_fallback_pct: {100.0 - fb_pct:.2f}%")
    if val_ex:
        q_val = count_jsonl_quality_issues(val_ex)
        print("\n  Val shard:")
        for k, v in q_val.items():
            print(f"    {k}: {v}")
        if q_val["n"]:
            fb_pct = 100.0 * q_val["fallback_template"] / q_val["n"]
            print(f"    approx_non_fallback_pct: {100.0 - fb_pct:.2f}%")
    combined = train_ex + val_ex
    if combined:
        qc = count_jsonl_quality_issues(combined)
        print("\n  Train+val combined:")
        for k, v in qc.items():
            print(f"    {k}: {v}")
        if qc["n"]:
            fb_pct = 100.0 * qc["fallback_template"] / qc["n"]
            print(f"    approx_non_fallback_pct: {100.0 - fb_pct:.2f}%")


def _fallback_reasoning(prompt: str, answer: str, puzzle_type: str) -> str:
    """Generate template-based CoT when solver fails."""
    lines = [
        f"This is a {puzzle_type.replace('_', ' ')} puzzle.",
        "Let me analyze the examples carefully to find the pattern.",
        "",
        "After examining all the given examples, I can identify the transformation rule.",
        "",
        f"Applying this rule to the query gives: {answer}",
    ]
    return "\n".join(lines)


def _oracle_only_reasoning(answer: str) -> str:
    """Minimal CoT for oracle-only rows (no solver-derived reasoning)."""
    return (
        "The query operator or puzzle constraints do not allow a unique rule from "
        "the examples alone, or the automated solver did not match ground truth.\n"
        f"Result: {answer}"
    )


def _symbol_equation_train_example(
    prompt: str, gt: str, solver_answer: str, reasoning: str
) -> dict | None:
    """Apply 3-tier policy for symbol_equation rows from train.csv."""
    solver_correct = verify_answer(solver_answer, gt) == 1.0
    tier = symbol_equation_sft_tier(prompt, solver_correct=solver_correct)

    if tier == "exclude":
        return None

    if tier == "trusted":
        use_solver_cot = (
            solver_correct
            and bool(reasoning)
            and _reasoning_final_result_consistent(reasoning, gt)
        )
        if use_solver_cot:
            return make_message(prompt, gt, reasoning)
        return make_message(prompt, gt, _oracle_only_reasoning(gt))

    return make_message(prompt, gt, _oracle_only_reasoning(gt))


# ─── Process train.csv ───

def process_train_csv(max_rows: int | None = None, csv_path: str | None = None) -> list[dict]:
    examples = []
    sym_skipped = 0
    oracle_only = 0
    path = find_train_csv(csv_path)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            prompt = row["prompt"]
            gt = row["answer"]
            puzzle_type = classify_puzzle(prompt)

            solver_answer, reasoning = solve_puzzle(prompt)

            if puzzle_type == "symbol_equation":
                ex = _symbol_equation_train_example(
                    prompt, gt, solver_answer, reasoning
                )
                if ex is None:
                    sym_skipped += 1
                    continue
                examples.append(ex)
                continue

            use_solver_cot = (
                verify_answer(solver_answer, gt) == 1.0
                and bool(reasoning)
                and _reasoning_final_result_consistent(reasoning, gt)
            )
            if use_solver_cot:
                examples.append(make_message(prompt, gt, reasoning))
            else:
                examples.append(make_message(prompt, gt, _oracle_only_reasoning(gt)))
                oracle_only += 1

    if sym_skipped:
        print(f"  symbol_equation excluded (ambiguous crypt): {sym_skipped}", flush=True)
    if oracle_only:
        print(f"  oracle-only CoT (solver miss or inconsistent trace): {oracle_only}", flush=True)
    return examples


# ─── Synthetic generators ───

def gen_gravity(n: int = 500) -> list[dict]:
    examples = []
    for _ in range(n):
        g = round(random.uniform(2.0, 20.0), 2)
        num_pairs = random.randint(4, 6)
        t_values = [round(random.uniform(0.5, 6.0), 2) for _ in range(num_pairs)]
        d_values = [round(0.5 * g * t * t, 2) for t in t_values]
        t_query = round(random.uniform(0.5, 6.0), 2)
        d_answer = round(0.5 * g * t_query * t_query, 2)

        lines = [
            "In Alice's Wonderland, the gravitational constant has been secretly changed. "
            "Here are some example observations:"
        ]
        for t, d in zip(t_values, d_values):
            lines.append(f"For t = {t}s, distance = {d} m")
        lines.append(
            f"Now, determine the distance for t = {t_query}s."
        )
        prompt = "\n".join(lines)

        reasoning_parts = ["Computing gravitational constant g from each example pair:"]
        g_estimates = []
        for t, d in zip(t_values, d_values):
            ge = round(2 * d / (t * t), 4)
            g_estimates.append(ge)
            reasoning_parts.append(f"  t={t}, d={d} → g = 2×{d}/{t}² = {ge}")
        g_avg = sum(g_estimates) / len(g_estimates)
        reasoning_parts.append(f"\nAverage g = {g_avg:.4f}")
        d_predicted = round(0.5 * g_avg * t_query * t_query, 2)
        reasoning_parts.append(
            f"For t = {t_query}s: d = 0.5 × {g_avg:.4f} × {t_query}² = {d_predicted}"
        )
        answer = f"{d_answer:.2f}"
        examples.append(make_message(prompt, answer, "\n".join(reasoning_parts)))

    return examples


def gen_unit_conversion(n: int = 500) -> list[dict]:
    examples = []
    for _ in range(n):
        ratio = round(random.uniform(0.3, 5.0), 4)
        num_pairs = random.randint(3, 6)
        in_vals = [round(random.uniform(1.0, 50.0), 2) for _ in range(num_pairs)]
        out_vals = [round(v * ratio, 2) for v in in_vals]
        q_val = round(random.uniform(1.0, 50.0), 2)
        q_answer = round(q_val * ratio, 2)

        lines = [
            "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
            "For example:"
        ]
        for inv, outv in zip(in_vals, out_vals):
            lines.append(f"{inv} m becomes {outv}")
        lines.append(f"Now, convert the following measurement: {q_val} m")
        prompt = "\n".join(lines)

        reasoning_parts = ["Computing conversion ratio from each example pair:"]
        ratios = []
        for inv, outv in zip(in_vals, out_vals):
            r = round(outv / inv, 6)
            ratios.append(r)
            reasoning_parts.append(f"  {inv} → {outv}, ratio = {r}")
        avg_r = sum(ratios) / len(ratios)
        reasoning_parts.append(f"\nAverage ratio = {avg_r:.6f}")
        reasoning_parts.append(f"For {q_val}: {q_val} × {avg_r:.6f} = {q_answer}")
        answer = f"{q_answer:.2f}"
        examples.append(make_message(prompt, answer, "\n".join(reasoning_parts)))

    return examples


def gen_roman_numeral(n: int = 500) -> list[dict]:
    examples = []
    for _ in range(n):
        num_examples = random.randint(3, 5)
        example_nums = random.sample(range(1, 3999), num_examples)
        query_num = random.randint(1, 3999)

        lines = [
            "In Alice's Wonderland, numbers are secretly converted into a different "
            "numeral system. Some examples are given below:"
        ]
        for num in example_nums:
            lines.append(f"{num} -> {int_to_roman(num)}")
        lines.append(
            f"Now, write the number {query_num} in the Wonderland numeral system."
        )
        prompt = "\n".join(lines)

        answer = int_to_roman(query_num)
        reasoning_parts = [
            f"Converting {query_num} to Roman numerals using greedy subtraction:"
        ]
        from solvers.roman_numeral import ROMAN_MAP
        remainder = query_num
        for value, numeral in ROMAN_MAP:
            while remainder >= value:
                reasoning_parts.append(
                    f"  {remainder} >= {value} → {numeral}, remainder {remainder - value}"
                )
                remainder -= value
        reasoning_parts.append(f"Result: {answer}")
        examples.append(make_message(prompt, answer, "\n".join(reasoning_parts)))

    return examples


def gen_cipher(n: int = 500) -> list[dict]:
    wordlist = [
        "alice", "queen", "king", "hatter", "rabbit", "cat", "mouse",
        "turtle", "dragon", "wizard", "knight", "princess", "student",
        "bird", "garden", "castle", "palace", "forest", "mountain",
        "valley", "tower", "door", "mirror", "book", "key", "secret",
        "magical", "golden", "mysterious", "hidden", "ancient", "wise",
        "discovers", "creates", "imagines", "follows", "reads", "watches",
        "chases", "draws", "dreams", "inside", "under", "near", "through",
        "the", "a", "in", "above", "beyond", "cave", "island", "ocean",
        "library", "treasure", "puzzle", "silver", "crystal", "dark",
        "bright", "curious", "strange", "explores", "writes", "sees",
        "village", "school", "teacher", "found", "story", "potion",
        "message", "map", "colorful", "studies", "clever",
    ]

    examples = []
    for _ in range(n):
        letters = list("abcdefghijklmnopqrstuvwxyz")
        shuffled = letters[:]
        random.shuffle(shuffled)
        cipher_map = dict(zip(letters, shuffled))
        reverse_map = dict(zip(shuffled, letters))

        num_example_sents = random.randint(4, 7)
        example_sents = []
        for _ in range(num_example_sents):
            sent_len = random.randint(2, 5)
            words = random.sample(wordlist, min(sent_len, len(wordlist)))
            plain = " ".join(words)
            encrypted = " ".join(
                "".join(cipher_map.get(c, c) for c in w) for w in words
            )
            example_sents.append((encrypted, plain))

        query_len = random.randint(2, 4)
        query_words = random.sample(wordlist, min(query_len, len(wordlist)))
        query_plain = " ".join(query_words)
        query_encrypted = " ".join(
            "".join(cipher_map.get(c, c) for c in w) for w in query_words
        )

        lines = [
            "In Alice's Wonderland, secret encryption rules are used on text. "
            "Here are some examples:"
        ]
        for enc, plain in example_sents:
            lines.append(f"{enc} -> {plain}")
        lines.append(f"Now, decrypt the following text: {query_encrypted}")
        prompt = "\n".join(lines)

        reasoning_parts = ["Building character substitution map from examples:"]
        known = {}
        for enc, plain in example_sents:
            for ec, pc in zip(enc.replace(" ", ""), plain.replace(" ", "")):
                if ec.isalpha() and pc.isalpha():
                    known[ec] = pc
        map_str = ", ".join(f"{k}→{v}" for k, v in sorted(known.items()))
        reasoning_parts.append(f"  {map_str}")
        reasoning_parts.append(f"\nDecrypting: {query_encrypted}")
        for w in query_words:
            enc_w = "".join(cipher_map.get(c, c) for c in w)
            reasoning_parts.append(f"  {enc_w} → {w}")
        reasoning_parts.append(f"\nResult: {query_plain}")

        examples.append(make_message(prompt, query_plain, "\n".join(reasoning_parts)))

    return examples


def gen_bit_manipulation(n: int = 500) -> list[dict]:
    """Generate bit manipulation puzzles with known operations."""

    def random_single_op():
        ops = []
        c = random.randint(0, 255)
        ops.append((f"XOR with 0x{c:02x}", lambda x, c=c: x ^ c))
        ops.append((f"AND with 0x{c:02x}", lambda x, c=c: x & c))
        ops.append((f"OR with 0x{c:02x}", lambda x, c=c: x | c))
        ops.append(("NOT", lambda x: (~x) & 0xFF))
        shift = random.randint(1, 7)
        ops.append(
            (f"rotate left by {shift}", lambda x, s=shift: ((x << s) | (x >> (8 - s))) & 0xFF)
        )
        ops.append(
            (f"rotate right by {shift}", lambda x, s=shift: ((x >> s) | (x << (8 - s))) & 0xFF)
        )
        ops.append(
            ("reverse bits", lambda x: int(f"{x:08b}"[::-1], 2))
        )
        ops.append(
            (f"ADD {c} mod 256", lambda x, c=c: (x + c) & 0xFF)
        )
        return random.choice(ops)

    examples = []
    for _ in range(n):
        op_name, op_func = random_single_op()
        num_pairs = random.randint(7, 10)
        inputs = random.sample(range(256), num_pairs + 1)
        query_input = inputs[-1]
        pair_inputs = inputs[:-1]

        lines = [
            "In Alice's Wonderland, a secret bit manipulation rule transforms "
            "8-bit binary numbers. The transformation involves operations like "
            "bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or "
            "choice functions.\n\nHere are some examples of input -> output:"
        ]
        for inp in pair_inputs:
            out = op_func(inp)
            lines.append(f"{inp:08b} -> {out:08b}")
        lines.append(f"\nNow, determine the output for: {query_input:08b}")
        prompt = "\n".join(lines)

        answer = f"{op_func(query_input):08b}"
        reasoning_parts = [
            "Testing operations against all examples:",
            f"  Found: {op_name} matches all examples.",
        ]
        for inp in pair_inputs[:3]:
            reasoning_parts.append(
                f"  Verify: {op_name}({inp:08b}) = {op_func(inp):08b}"
            )
        reasoning_parts.append(
            f"\nApply to query: {op_name}({query_input:08b}) = {answer}"
        )
        examples.append(make_message(prompt, answer, "\n".join(reasoning_parts)))

    return examples


_FN2_NAMES = {
    0: "0", 1: "AND", 2: "a AND NOT b", 3: "a",
    4: "NOT a AND b", 5: "b", 6: "XOR", 7: "OR",
    8: "NOR", 9: "XNOR", 10: "NOT b", 11: "a OR NOT b",
    12: "NOT a", 13: "NOT a OR b", 14: "NAND", 15: "1",
}


def gen_bit_manipulation_perbit(n: int = 750) -> list[dict]:
    """Generate per-bit boolean function puzzles matching real competition format."""
    examples_out = []
    for _ in range(n):
        base_offset = random.choice(range(8))
        fn_pool = random.sample(range(1, 15), min(4, 14))

        bit_specs = []
        for out_bit in range(8):
            fn_id = random.choice(fn_pool)
            b1 = (out_bit + base_offset) % 8
            b2 = (out_bit + base_offset + random.choice([1, 2, 3])) % 8
            if b1 == b2:
                b2 = (b1 + 1) % 8
            bit_specs.append((fn_id, b1, b2))

        def apply_transform(x, specs=bit_specs):
            result = 0
            for out_bit, (fn_id, b1, b2) in enumerate(specs):
                idx = ((x >> b1) & 1) * 2 + ((x >> b2) & 1)
                val = (fn_id >> idx) & 1
                result |= val << out_bit
            return result

        num_pairs = random.randint(7, 10)
        inputs = random.sample(range(256), num_pairs + 1)
        query_input = inputs[-1]
        pair_inputs = inputs[:-1]

        lines = [
            "In Alice's Wonderland, a secret bit manipulation rule transforms "
            "8-bit binary numbers. The transformation involves operations like "
            "bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or "
            "choice functions.\n\nHere are some examples of input -> output:"
        ]
        for inp in pair_inputs:
            lines.append(f"{inp:08b} -> {apply_transform(inp):08b}")
        lines.append(f"\nNow, determine the output for: {query_input:08b}")
        prompt = "\n".join(lines)

        answer = f"{apply_transform(query_input):08b}"
        reasoning_parts = [
            "Analyzing each output bit independently:",
        ]
        for out_bit, (fn_id, b1, b2) in enumerate(bit_specs):
            fn_name = _FN2_NAMES.get(fn_id, f"f{fn_id}")
            reasoning_parts.append(
                f"  Output bit {out_bit} = {fn_name}(bit{b1}, bit{b2}) "
                f"→ {(apply_transform(query_input) >> out_bit) & 1}"
            )
        reasoning_parts.append(f"\nResult: {answer}")
        examples_out.append(make_message(prompt, answer, "\n".join(reasoning_parts)))

    return examples_out


def gen_symbol_equation(n: int = 500) -> list[dict]:
    """Generate simple symbol equation puzzles (character substitution with deletion)."""
    import string

    pool = list(string.punctuation + string.digits)

    examples_list = []
    for _ in range(n):
        chars = random.sample(pool, min(12, len(pool)))
        mapping = {}
        num_empty = random.randint(1, 3)
        empty_chars = random.sample(chars, num_empty)
        remaining = [c for c in chars if c not in empty_chars]

        for c in empty_chars:
            mapping[c] = ""
        mapped_to = random.sample(pool, len(remaining))
        for c, v in zip(remaining, mapped_to):
            mapping[c] = v

        num_rules = random.randint(3, 5)
        rules = []
        for _ in range(num_rules):
            lhs = "".join(random.choices(chars, k=5))
            rhs = "".join(mapping.get(c, c) for c in lhs)
            if rhs:
                rules.append((lhs, rhs))

        if len(rules) < 3:
            continue

        query = "".join(random.choices(chars, k=5))
        answer = "".join(mapping.get(c, c) for c in query)
        if not answer:
            continue

        lines = [
            "In Alice's Wonderland, a secret set of transformation rules is "
            "applied to equations. Below are a few examples:"
        ]
        for lhs, rhs in rules:
            lines.append(f"{lhs} = {rhs}")
        lines.append(f"Now, determine the result for: {query}")
        prompt = "\n".join(lines)

        reasoning_parts = ["Analyzing the transformation rules:"]
        for lhs, rhs in rules:
            reasoning_parts.append(f"  {lhs} = {rhs}")
        reasoning_parts.append("\nDerived character mapping:")
        for c in sorted(mapping.keys()):
            val = mapping[c] if mapping[c] else "ε (deleted)"
            reasoning_parts.append(f"  '{c}' → {val}")
        reasoning_parts.append(f"\nApplying to query: {query} → {answer}")
        examples_list.append(make_message(prompt, answer, "\n".join(reasoning_parts)))

    return examples_list


def gen_symbol_equation_digit(n: int = 750) -> list[dict]:
    """Generate digit-arithmetic symbol equation puzzles matching competition format.
    Multiple operators per puzzle, each mapping to a different function.
    """
    import string

    FUNCTIONS = [
        ("addition", lambda a, b: str(a + b)),
        ("subtraction (a-b)", lambda a, b: str(a - b)),
        ("absolute difference", lambda a, b: str(abs(a - b))),
        ("multiplication", lambda a, b: str(a * b)),
        ("concatenation", lambda a, b: f"{a:02d}{b:02d}"),
        ("reverse concatenation", lambda a, b: f"{b:02d}{a:02d}"),
    ]
    FUNCTION_NAMES = {
        "addition": "a + b",
        "subtraction (a-b)": "a - b",
        "absolute difference": "|a - b|",
        "multiplication": "a * b",
        "concatenation": "a||b",
        "reverse concatenation": "b||a",
    }

    op_chars = list(string.punctuation)

    examples_list = []
    for _ in range(n):
        num_ops = random.randint(1, 3)
        chosen_ops = random.sample(op_chars, num_ops)
        chosen_funcs = random.sample(FUNCTIONS, num_ops)
        op_func_map = dict(zip(chosen_ops, chosen_funcs))

        num_rules = random.randint(3, 6)
        rules = []
        for _ in range(num_rules):
            op = random.choice(chosen_ops)
            fname, func = op_func_map[op]
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            result = func(a, b)
            rules.append((f"{a:02d}", op, f"{b:02d}", result, fname))

        query_op = random.choice(chosen_ops)
        qa = random.randint(10, 99)
        qb = random.randint(10, 99)
        qfname, qfunc = op_func_map[query_op]
        answer = qfunc(qa, qb)

        lines = [
            "In Alice's Wonderland, a secret set of transformation rules is "
            "applied to equations. Below are a few examples:"
        ]
        for a, op, b, result, _ in rules:
            lines.append(f"{a}{op}{b} = {result}")
        lines.append(f"Now, determine the result for: {qa:02d}{query_op}{qb:02d}")
        prompt = "\n".join(lines)

        reasoning_parts = [
            "We need to infer the transformation rule from the examples.",
            "",
            "Examples:",
        ]
        for a, op, b, result, _ in rules:
            reasoning_parts.append(f"  {a}{op}{b} = {result}")
        reasoning_parts.append("")
        all_outputs = [result for _, _, _, result, _ in rules]
        reasoning_parts.append(f"The outputs are {', '.join(all_outputs)}")
        reasoning_parts.append("")

        # Show operator analysis with candidates tried
        for op in chosen_ops:
            fname, func = op_func_map[op]
            display = FUNCTION_NAMES.get(fname, fname)
            op_exs = [(a, b, result) for a, o, b, result, _ in rules if o == op]
            if not op_exs:
                reasoning_parts.append(f"Operator '{op}' → {display} (no examples, inferred from query)")
                reasoning_parts.append("")
                continue
            examples_str = ", ".join(f"{a}{op}{b} = {r}" for a, b, r in op_exs)
            reasoning_parts.append(f"Looking at operator '{op}' [{examples_str}]:")
            reasoning_parts.append("  Trying operations on identity:")
            first_a, first_b, first_exp = op_exs[0]
            # Show a few wrong candidates then the correct one
            wrong_candidates = [
                ("concatenation", f"{first_a}{first_b}"),
                ("reverse concatenation", f"{first_b}{first_a}"),
                ("addition", str(int(first_a) + int(first_b))),
                ("absolute difference", str(abs(int(first_a) - int(first_b)))),
                ("subtraction (a-b)", str(int(first_a) - int(first_b))),
                ("multiplication", str(int(first_a) * int(first_b))),
            ]
            for cname, cval in wrong_candidates:
                status = "match" if cval == first_exp else "wrong"
                reasoning_parts.append(f"    {cname} f({first_a}, {first_b}) = {cval} {status}")
                if cname == fname:
                    reasoning_parts.append(f"    correct, actions: {display}")
                    break
            reasoning_parts.append("")

        reasoning_parts.append(f"Applying to {qa:02d}{query_op}{qb:02d}:")
        reasoning_parts.append(f"  {FUNCTION_NAMES.get(qfname, qfname)}({qa:02d}, {qb:02d}) = {answer}")
        reasoning_parts.append(f"  Result: {answer}")
        examples_list.append(make_message(prompt, answer, "\n".join(reasoning_parts)))

    return examples_list


_AUGMENT_SYMBOLS = list('!"#$%&\'()*+-./:;<>?@[\\]^`{|}')


def _box_individual(chars: list[str]) -> str:
    return "".join(f"\u3010{c}\u3011" for c in chars)


def _box_merged(chars: list[str]) -> str:
    return f"\u3010{''.join(chars)}\u3011"


def gen_concatenation(n: int = 1500) -> list[dict]:
    """Merge individually-bracketed symbols into one bracket.
    Trains the model on symbol merging/concatenation fundamentals.
    """
    rng = random.Random(99)
    examples = []
    lines_per = 100
    demo_lines = 3

    for i in range(n):
        demo_chars = [[rng.choice(_AUGMENT_SYMBOLS) for _ in range(rng.randint(2, 8))]
                      for _ in range(demo_lines)]
        demo_pairs = [(_box_individual(c), _box_merged(c)) for c in demo_chars]

        sample_in = [f"{j:02d} {inp}" for j, (inp, _) in enumerate(demo_pairs)]
        sample_out = [f"{j:02d} {inp} -> {out}" for j, (inp, out) in enumerate(demo_pairs)]

        test_inputs = []
        test_answers = []
        for row in range(lines_per):
            chars = [rng.choice(_AUGMENT_SYMBOLS) for _ in range(rng.randint(2, 8))]
            inp, out = _box_individual(chars), _box_merged(chars)
            test_inputs.append(f"{row:02d} {inp}")
            test_answers.append(f"{row:02d} {inp} -> {out}")

        prompt = (
            "In Alice's Wonderland, secret processing rules are used on text.\n\n"
            "This is a sample input.\n" + "\n".join(sample_in) +
            "\n\nThis is a sample output.\n" + "\n".join(sample_out) +
            "\n\nThis is your input.\n" + "\n".join(test_inputs)
        )
        answer = "\n".join(test_answers)

        reasoning = (
            "The rule merges individually-bracketed symbols into a single bracket.\n"
            f"For example: {demo_pairs[0][0]} -> {demo_pairs[0][1]}\n"
            "Apply this to each row."
        )
        examples.append(make_message(prompt, answer, reasoning))

    return examples


def gen_splitting(n: int = 1500) -> list[dict]:
    """Split a single bracket into individually-bracketed symbols.
    Reverse of concatenation — trains symbol-level awareness.
    """
    rng = random.Random(77)
    examples = []
    lines_per = 100
    demo_lines = 3

    for i in range(n):
        demo_chars = [[rng.choice(_AUGMENT_SYMBOLS) for _ in range(rng.randint(2, 8))]
                      for _ in range(demo_lines)]
        demo_pairs = [(_box_merged(c), _box_individual(c)) for c in demo_chars]

        sample_in = [f"{j:02d} {inp}" for j, (inp, _) in enumerate(demo_pairs)]
        sample_out = [f"{j:02d} {inp} -> {out}" for j, (inp, out) in enumerate(demo_pairs)]

        test_inputs = []
        test_answers = []
        for row in range(lines_per):
            chars = [rng.choice(_AUGMENT_SYMBOLS) for _ in range(rng.randint(2, 8))]
            inp, out = _box_merged(chars), _box_individual(chars)
            test_inputs.append(f"{row:02d} {inp}")
            test_answers.append(f"{row:02d} {inp} -> {out}")

        prompt = (
            "In Alice's Wonderland, secret processing rules are used on text.\n\n"
            "This is a sample input.\n" + "\n".join(sample_in) +
            "\n\nThis is a sample output.\n" + "\n".join(sample_out) +
            "\n\nThis is your input.\n" + "\n".join(test_inputs)
        )
        answer = "\n".join(test_answers)

        reasoning = (
            "The rule splits a merged bracket into individual character brackets.\n"
            f"For example: {demo_pairs[0][0]} -> {demo_pairs[0][1]}\n"
            "Apply this to each row."
        )
        examples.append(make_message(prompt, answer, reasoning))

    return examples


def gen_lstrip(n: int = 300) -> list[dict]:
    """Strip leading space from a bracketed symbol string.
    Trains precise symbol boundary handling.
    """
    rng = random.Random(91)
    examples = []
    lines_per = 100
    demo_lines = 3

    def _entry():
        length = 5 if rng.random() < 0.5 else rng.randint(1, 10)
        symbols = "".join(rng.choice(_AUGMENT_SYMBOLS) for _ in range(length))
        return f"\u3010 {symbols}\u3011", f"\u3010{symbols}\u3011"

    for i in range(n):
        demo_pairs = [_entry() for _ in range(demo_lines)]
        sample_in = [f"{j:02d} {inp}" for j, (inp, _) in enumerate(demo_pairs)]
        sample_out = [f"{j:02d} {inp} -> {out}" for j, (inp, out) in enumerate(demo_pairs)]

        test_inputs = []
        test_answers = []
        for row in range(lines_per):
            inp, out = _entry()
            test_inputs.append(f"{row:02d} {inp}")
            test_answers.append(f"{row:02d} {inp} -> {out}")

        prompt = (
            "In Alice's Wonderland, secret processing rules are used on text.\n\n"
            "This is a sample input.\n" + "\n".join(sample_in) +
            "\n\nThis is a sample output.\n" + "\n".join(sample_out) +
            "\n\nThis is your input.\n" + "\n".join(test_inputs)
        )
        answer = "\n".join(test_answers)

        reasoning = (
            "The rule strips the leading space from the bracketed text.\n"
            f"For example: {demo_pairs[0][0]} -> {demo_pairs[0][1]}\n"
            "Apply this to each row."
        )
        examples.append(make_message(prompt, answer, reasoning))

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Nemotron SFT JSONL from solvers + synthetics")
    parser.add_argument(
        "--audit-csv-only",
        action="store_true",
        help="Phase 0: print solver/trusted-CoT stats on train.csv and exit (no JSONL written).",
    )
    parser.add_argument(
        "--train-csv-max-rows",
        type=int,
        default=None,
        help="Optional: limit rows read from train.csv (local dev only; omit for full competition CSV).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction per puzzle type held out for val")
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to train.csv (default: competition-data path; used with --audit-csv-only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.audit_csv_only:
        report = audit_train_csv(args.train_csv_max_rows, csv_path=args.csv)
        print_audit_train_csv(report)
        return

    random.seed(args.seed)
    csv_limit = args.train_csv_max_rows

    print("Processing train.csv...")
    train_examples = process_train_csv(max_rows=csv_limit, csv_path=args.csv)
    print(f"  {len(train_examples)} examples from train.csv (max_rows={csv_limit!r})")

    print("Generating synthetic examples...")
    synthetic: list[dict] = []
    generators = [
        ("gravity", gen_gravity),
        ("unit_conversion", gen_unit_conversion),
        ("roman_numeral", gen_roman_numeral),
        ("cipher", gen_cipher),
        ("bit_manipulation", gen_bit_manipulation),
        ("bit_manipulation_perbit", gen_bit_manipulation_perbit),
        ("symbol_equation", gen_symbol_equation),
        ("symbol_equation_digit", gen_symbol_equation_digit),
        ("sym_concatenation", gen_concatenation),
        ("sym_splitting", gen_splitting),
        ("sym_lstrip", gen_lstrip),
    ]
    gen_counts = {
        "bit_manipulation_perbit": 750,
        "symbol_equation_digit": 750,
        "sym_concatenation": 1500,
        "sym_splitting": 1500,
        "sym_lstrip": 300,
    }
    default_n = 500
    for name, gen in generators:
        count = gen_counts.get(name, default_n)
        data = gen(count)
        print(f"  {name}: {len(data)} synthetic examples")
        synthetic.extend(data)

    all_examples = train_examples + synthetic
    random.shuffle(all_examples)

    train_set, val_set = stratified_train_val_split(
        all_examples, val_ratio=args.val_ratio, seed=args.seed
    )

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
    train_path = os.path.join(os.path.dirname(__file__), "..", "data", "sft_train.jsonl")
    val_path = os.path.join(os.path.dirname(__file__), "..", "data", "sft_val.jsonl")

    with open(train_path, "w") as f:
        for ex in train_set:
            f.write(json.dumps(ex) + "\n")

    with open(val_path, "w") as f:
        for ex in val_set:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(train_set)} train and {len(val_set)} val examples (stratified split)")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")

    print_sft_jsonl_audit(train_path, val_path)

    type_counts = Counter()
    for ex in all_examples:
        prompt = ex["messages"][1]["content"]
        type_counts[classify_puzzle(prompt)] += 1
    print("\nExamples per type:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")

    total_tokens = sum(
        len(msg["content"]) // 4
        for ex in all_examples
        for msg in ex["messages"]
    )
    print(f"\nEstimated total tokens: ~{total_tokens:,}")


if __name__ == "__main__":
    main()
