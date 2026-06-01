"""Symbol equation solver — two-stage approach.

Stage 1 (digit-arithmetic): many competition prompts have the form
    AB op CD = result
where AB and CD are two-digit numbers, op is a punctuation separator, and
result is some function of (AB, CD) — sum, product, concatenation, digit-wise
operations, permutations of the four digits, etc. We try a library of
candidate functions, intersect those that match every example rule, and
return the unique answer if one survives.

Stage 2 (character substitution): each character maps to a string (possibly
empty); the concatenation of mapped characters equals the RHS. We find a
consistent mapping via partition enumeration + backtracking.
"""

import itertools
import math
import re
from math import comb

# LHS longer than this makes partition counts explode; skip such rules.
MAX_LHS_LENGTH = 12
# Max weak-composition count to enumerate for one rule (split RHS into len(lhs) parts).
MAX_PARTITIONS_PER_RULE = 100_000
# Backtracking steps cap to avoid hangs on pathological inputs.
MAX_BACKTRACK_STEPS = 100_000


def _parse_symbol_rules(prompt: str) -> tuple[list[tuple[str, str]], str]:
    lines = prompt.strip().split("\n")
    rules = []
    query = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if "determine the result for:" in lower:
            m = re.search(r"determine the result for:\s*(.+)", line, re.IGNORECASE)
            if m:
                query = m.group(1).strip()
            continue
        if any(
            kw in lower
            for kw in [
                "alice",
                "wonderland",
                "transformation",
                "below",
                "example",
                "secret",
                "applied",
            ]
        ):
            continue
        if " = " in line:
            parts = line.split(" = ", 1)
            lhs = parts[0]
            rhs = parts[1]
            if not lhs:
                continue
            if len(lhs) > MAX_LHS_LENGTH:
                continue
            rules.append((lhs, rhs))

    return rules, query or ""


def _partitions(s: str, n: int):
    """Generate all ways to split string s into n parts (each possibly empty)."""
    k = len(s)
    if n == 1:
        yield (s,)
        return
    for i in range(k + 1):
        for rest in _partitions(s[i:], n - 1):
            yield (s[:i],) + rest


def _solve_mapping(rules: list[tuple[str, str]]) -> dict[str, str] | None:
    """Constraint propagation + backtracking solver for character mapping puzzles.

    Each unique character in the LHS maps to a fixed substring (possibly empty).
    For each rule, concatenating the mapped substrings of every LHS character must
    equal the RHS.
    """
    if not rules:
        return None

    all_chars: list[str] = []
    seen_chars: set[str] = set()
    for lhs, _ in rules:
        for c in lhs:
            if c not in seen_chars:
                seen_chars.add(c)
                all_chars.append(c)

    per_char_values: dict[str, set[str] | None] = {c: None for c in all_chars}

    for lhs, rhs in rules:
        n_parts = len(lhs)
        if comb(len(rhs) + n_parts - 1, n_parts - 1) > MAX_PARTITIONS_PER_RULE:
            continue
        char_vals: dict[str, set[str]] = {}
        for parts in _partitions(rhs, n_parts):
            mapping: dict[str, str] = {}
            valid = True
            for c, val in zip(lhs, parts):
                if c in mapping:
                    if mapping[c] != val:
                        valid = False
                        break
                else:
                    mapping[c] = val
            if valid:
                for c, v in mapping.items():
                    char_vals.setdefault(c, set()).add(v)
        for c, vals in char_vals.items():
            if per_char_values[c] is None:
                per_char_values[c] = vals.copy()
            else:
                per_char_values[c] &= vals

    for c in per_char_values:
        if per_char_values[c] is not None and len(per_char_values[c]) == 0:
            return None

    def _check(mapping: dict[str, str]) -> bool:
        for lhs, rhs in rules:
            if all(c in mapping for c in lhs):
                if "".join(mapping[c] for c in lhs) != rhs:
                    return False
        return True

    def _partial_check(mapping: dict[str, str]) -> bool:
        for lhs, rhs in rules:
            if all(c in mapping for c in lhs):
                if "".join(mapping[c] for c in lhs) != rhs:
                    return False
            else:
                total_known = sum(len(mapping[c]) for c in lhs if c in mapping)
                if total_known > len(rhs):
                    return False
                prefix = ""
                for c in lhs:
                    if c in mapping:
                        prefix += mapping[c]
                    else:
                        break
                if prefix and not rhs.startswith(prefix):
                    return False
                suffix = ""
                for c in reversed(lhs):
                    if c in mapping:
                        suffix = mapping[c] + suffix
                    else:
                        break
                if suffix and not rhs.endswith(suffix):
                    return False
        return True

    sorted_chars = sorted(
        all_chars,
        key=lambda c: len(per_char_values[c]) if per_char_values[c] else 999,
    )

    bt_steps = [0]

    def _backtrack(idx: int, mapping: dict[str, str]) -> dict[str, str] | None:
        bt_steps[0] += 1
        if bt_steps[0] > MAX_BACKTRACK_STEPS:
            return None
        if idx == len(sorted_chars):
            return dict(mapping) if _check(mapping) else None

        c = sorted_chars[idx]
        candidates = per_char_values[c] if per_char_values[c] else {""}

        for val in sorted(candidates, key=lambda x: (len(x), x)):
            mapping[c] = val
            if _partial_check(mapping):
                result = _backtrack(idx + 1, mapping)
                if result is not None:
                    return result
            del mapping[c]
        return None

    return _backtrack(0, {})


_DIGIT_LINE_PAT = re.compile(r"^(\d+)(\D)(\d+)\s*=\s*(.+)$")
_DIGIT_QUERY_PAT = re.compile(r"result for:\s*(\d+)(\D)(\d+)", re.IGNORECASE)


def _common_digit_funcs(a: str, b: str) -> list[tuple[str, str]]:
    """High-priority candidates tried first. Order matters: first match wins."""
    out: list[tuple[str, str]] = []
    ia, ib = int(a), int(b)
    out.append(("a||b", a + b))
    out.append(("b||a", b + a))
    out.append(("a + b", str(ia + ib)))
    out.append(("|a - b|", str(abs(ia - ib))))
    out.append(("-|a - b|", str(-abs(ia - ib))))
    out.append(("a - b", str(ia - ib)))
    out.append(("b - a", str(ib - ia)))
    out.append(("a * b", str(ia * ib)))
    return out


def _rare_digit_funcs(a: str, b: str) -> list[tuple[str, str]]:
    """Lower-priority candidates, tried only when common ones don't match."""
    out: list[tuple[str, str]] = []
    ia, ib = int(a), int(b)

    out.append(("a * b + 1", str(ia * ib + 1)))
    out.append(("a * b - 1", str(ia * ib - 1)))
    out.append(("a + b + 1", str(ia + ib + 1)))
    out.append(("a + b - 1", str(ia + ib - 1)))
    out.append(("a - b + 1", str(ia - ib + 1)))
    out.append(("a - b - 1", str(ia - ib - 1)))
    if ia and ib:
        big, small = max(ia, ib), min(ia, ib)
        out.append(("max % min", str(big % small)))
    if ib:
        out.append(("a // b", str(ia // ib)))
        out.append(("a % b", str(ia % ib)))
    if ia:
        out.append(("b // a", str(ib // ia)))
        out.append(("b % a", str(ib % ia)))

    if len(a) == 2 and len(b) == 2:
        a0, a1 = int(a[0]), int(a[1])
        b0, b1 = int(b[0]), int(b[1])
        out.append(("dw:|a0-b0||a1-b1|", f"{abs(a0 - b0)}{abs(a1 - b1)}"))
        out.append(("digit add mod10", str((a0 + b0) % 10) + str((a1 + b1) % 10)))
        out.append(("digit sub mod10", str((a0 - b0) % 10) + str((a1 - b1) % 10)))
        out.append(("cross multiply", str(a0 * b0 + a1 * b1)))
        out.append(("cross multiply rev", str(a0 * b1 + a1 * b0)))
        out.append(("digit multiply", str(a0 * b0) + str(a1 * b1)))
        out.append(("digit multiply rev", str(a0 * b1) + str(a1 * b0)))
        out.append(("digit sum diff", str((a0 + a1) - (b0 + b1))))
        out.append(("digit sum sum", str((a0 + a1) + (b0 + b1))))
        out.append(("digit product diff", str(a0 * a1 - b0 * b1)))
        out.append(("digit product sum", str(a0 * a1 + b0 * b1)))
        det_val = a0 * b1 - a1 * b0
        out.append(("determinant", str(det_val)))
        out.append(("abs determinant", str(abs(det_val))))

    out.append(("a XOR b", str(ia ^ ib)))
    out.append(("a AND b", str(ia & ib)))
    out.append(("a OR b", str(ia | ib)))
    out.append(("(a+b) * 2", str((ia + ib) * 2)))
    out.append(("(a-b) * 2", str(abs(ia - ib) * 2)))
    out.append(("a*2 + b", str(ia * 2 + ib)))
    out.append(("a + b*2", str(ia + ib * 2)))
    out.append(("a*2 - b", str(abs(ia * 2 - ib))))
    out.append(("a*2", str(ia * 2)))
    out.append(("b*2", str(ib * 2)))

    g = math.gcd(ia, ib)
    out.append(("gcd(a, b)", str(g)))
    if g:
        out.append(("lcm(a, b)", str(ia * ib // g)))
    out.append(("max(a, b)", str(max(ia, ib))))
    out.append(("min(a, b)", str(min(ia, ib))))

    if len(a) == 2 and len(b) == 2:
        a0, a1 = int(a[0]), int(a[1])
        b0, b1 = int(b[0]), int(b[1])
        out.append(("dw:(a0+b0)(a1+b1)", f"{a0 + b0}{a1 + b1}"))
        out.append(("dw:(a0*b0)(a1*b1)", f"{a0 * b0}{a1 * b1}"))
        out.append(("dw:max/max", f"{max(a0, b0)}{max(a1, b1)}"))
        out.append(("dw:min/min", f"{min(a0, b0)}{min(a1, b1)}"))
        out.append(("dw:(a0+b0)(|a1-b1|)", f"{a0 + b0}{abs(a1 - b1)}"))
        out.append(("dw:(|a0-b0|)(a1+b1)", f"{abs(a0 - b0)}{a1 + b1}"))
        out.append(("dw:(a0+b1)(a1+b0)", f"{a0 + b1}{a1 + b0}"))
        out.append(("dw:(a0*b1)(a1*b0)", f"{a0 * b1}{a1 * b0}"))
        out.append(("digit-product", str(a0 * a1 * b0 * b1)))
        p = ia * ib
        out.append(("(a*b)//100 + (a*b)%100", str(p // 100 + p % 100)))
        out.append(("reverse a || reverse b", a[::-1] + b[::-1]))
        out.append(("reverse b || reverse a", b[::-1] + a[::-1]))
        out.append(("sorted asc", "".join(sorted(a + b))))
        out.append(("sorted desc", "".join(sorted(a + b, reverse=True))))

        for perm in itertools.permutations([a[0], a[1], b[0], b[1]], 4):
            s = "".join(perm)
            out.append((f"perm {s}", s))

    out.append(("a^2", str(ia * ia)))
    out.append(("b^2", str(ib * ib)))
    out.append(("a^2 + b^2", str(ia * ia + ib * ib)))
    out.append(("a^2 - b^2", str(abs(ia * ia - ib * ib))))
    out.append(("digit-sum", str(sum(int(c) for c in a + b))))

    out.append(("a||b (zp)", f"{ia:02d}{ib:02d}"))
    out.append(("b||a (zp)", f"{ib:02d}{ia:02d}"))
    out.append(("a + b (zp)", f"{ia + ib:02d}" if ia + ib < 100 else str(ia + ib)))
    out.append(("|a-b| (zp)", f"{abs(ia - ib):02d}"))
    if ib:
        out.append(("a // b (r)", str(ia // ib) + str(ia % ib)))
    if ia:
        out.append(("b // a (r)", str(ib // ia) + str(ib % ia)))
    out.append(("a*b (zp4)", f"{ia * ib:04d}" if ia * ib < 10000 else str(ia * ib)))

    out.append(("b - a + 1", str(ib - ia + 1)))
    out.append(("b - a - 1", str(ib - ia - 1)))
    out.append(("(a+b)//2", str((ia + ib) // 2)))
    out.append(("(a*b)//2", str((ia * ib) // 2)))
    out.append(("a^b", str(ia ** ib) if ib <= 4 and ia ** ib < 100000 else ""))
    out.append(("b^a", str(ib ** ia) if ia <= 4 and ib ** ia < 100000 else ""))

    return out


def _all_digit_funcs(a: str, b: str) -> list[tuple[str, str]]:
    """All candidates: common first, then rare. Order is priority."""
    return _common_digit_funcs(a, b) + _rare_digit_funcs(a, b)


def _rev(s: str) -> str:
    """Reverse a numeric string, preserving leading minus sign."""
    if s.startswith("-"):
        return "-" + s[1:][::-1]
    return s[::-1]


def _detect_output_format(op_char: str, group: list[tuple[str, str, str]]) -> tuple[str, list[tuple[str, str, str]]]:
    """Detect whether outputs use neg_suffix or neg_prefix encoding.
    Returns (format_name, transformed_group) where transformed outputs are
    pure numeric strings.
    """
    if op_char == "-":
        return "num", list(group)

    any_neg_suffixed = any(
        out.endswith("-") and len(out) > 1 for _, _, out in group
    )
    any_neg_prefixed = any(
        out.startswith("-") and len(out) > 1 for _, _, out in group
    )
    any_op_suffixed = any(
        out.endswith(op_char) and len(out) > 1 for _, _, out in group
    )
    any_op_prefixed = any(
        out.startswith(op_char) and len(out) > 1 for _, _, out in group
    )
    if any_neg_suffixed:
        transformed = [
            (a, b, "-" + out[:-1] if out.endswith("-") and len(out) > 1 else out)
            for a, b, out in group
        ]
        return "neg_suffix", transformed
    if any_neg_prefixed:
        transformed = [
            (a, b, out) for a, b, out in group
        ]
        return "neg_prefix", transformed
    if any_op_suffixed:
        transformed = [
            (a, b, "-" + out[:-len(op_char)] if out.endswith(op_char) and len(out) > 1 else out)
            for a, b, out in group
        ]
        return "neg_suffix", transformed
    if any_op_prefixed:
        transformed = [
            (a, b, "-" + out[len(op_char):] if out.startswith(op_char) and len(out) > 1 else out)
            for a, b, out in group
        ]
        return "neg_prefix", transformed

    return "num", list(group)


def _encode_output(raw_result: str, fmt: str, op_char: str) -> str:
    """Re-encode a raw numeric result into the detected output format."""
    if fmt == "neg_suffix":
        if raw_result.startswith("-"):
            return raw_result[1:] + op_char
        return raw_result
    if fmt == "neg_prefix":
        if raw_result.startswith("-"):
            return op_char + raw_result[1:]
        return raw_result
    return raw_result


def _find_op_function(
    group: list[tuple[str, str, str]],
    rev_ops: bool,
    rev_res: bool,
    candidate_fn=None,
    prefer_signed: bool = False,
) -> str | None:
    """Try to find a function name consistent with all examples in group.

    Returns the highest-priority (earliest in candidate list) matching function.
    If candidate_fn is provided, use it; otherwise use _all_digit_funcs.

    When ``prefer_signed`` is set (used when the outputs encode negatives via an
    operator prefix/suffix), a tie between absolute difference and signed
    subtraction is resolved in favour of signed subtraction. The abs and signed
    forms agree on every (necessarily non-positive) example here, but only the
    signed form reproduces a *positive* result on a query where a > b — which is
    rendered without the sign affix.
    """
    if candidate_fn is None:
        candidate_fn = _all_digit_funcs
    common: set[str] | None = None
    for a, b, rhs in group:
        ta = a[::-1] if rev_ops else a
        tb = b[::-1] if rev_ops else b
        candidates = candidate_fn(ta, tb)
        matching = set()
        for name, val in candidates:
            final = _rev(val) if rev_res else val
            if final == rhs:
                matching.add(name)
        common = matching if common is None else (common & matching)
        if not common:
            return None
    if not common:
        return None
    # Return the first (highest-priority) function that survived intersection
    a0, b0, _ = group[0]
    ta0 = a0[::-1] if rev_ops else a0
    tb0 = b0[::-1] if rev_ops else b0
    for name, _ in candidate_fn(ta0, tb0):
        if name in common:
            if prefer_signed and name in ("|a - b|", "-|a - b|"):
                if "a - b" in common:
                    return "a - b"
                if "b - a" in common:
                    return "b - a"
            return name
    return None


def _try_digit_arithmetic(prompt: str) -> tuple[str, str] | None:
    """Multi-strategy digit arithmetic solver with per-operator inference,
    reversed operands/results, negative suffix/prefix encoding, and
    fallback for unseen query operators.
    """
    rules: list[tuple[str, str, str, str]] = []
    for line in prompt.splitlines():
        m = _DIGIT_LINE_PAT.match(line.strip())
        if m:
            rules.append((m.group(1), m.group(2), m.group(3), m.group(4).strip()))
    qm = _DIGIT_QUERY_PAT.search(prompt)
    if not qm or len(rules) < 2:
        return None
    qa, qop, qb = qm.group(1), qm.group(2), qm.group(3)

    from collections import defaultdict

    op_groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for A, op, B, rhs in rules:
        op_groups[op].append((A, B, rhs))

    # --- Strategy 1: per-operator inference with output format detection ---
    #     and reversed operands/results (4 combos per operator)

    _REV_COMBOS = [(True, True), (False, False), (True, False), (False, True)]

    op_results: dict[str, tuple[str, bool, bool, str]] = {}
    all_ops_ok = True

    for op_char, group in op_groups.items():
        fmt, transformed = _detect_output_format(op_char, group)

        found = False
        prefer_signed = fmt in ("neg_prefix", "neg_suffix")
        # Try common candidates first across all reversal combos, then rare
        for candidate_fn in (_common_digit_funcs, _rare_digit_funcs):
            for rev_ops, rev_res in _REV_COMBOS:
                func_name = _find_op_function(
                    transformed, rev_ops, rev_res, candidate_fn, prefer_signed
                )
                if func_name is not None:
                    op_results[op_char] = (func_name, rev_ops, rev_res, fmt)
                    found = True
                    break
            if found:
                break

        if not found:
            all_ops_ok = False

    # Determine which operator's result to use for the query
    target_result = None
    unseen_query_op = False

    if qop in op_results:
        target_result = op_results[qop]
    elif op_results:
        # Query operator was never demonstrated in the examples ("guess" puzzles):
        # the operation is genuinely undetermined, so default to absolute
        # difference (empirically the single best guess, matching the reference
        # equation_numeric generator). Applying some *other* operator's deduced
        # function here reproduces ground truth essentially never, so we do not.
        unseen_query_op = True
        most_common_op = max(op_results.keys(), key=lambda o: len(op_groups[o]))
        ref = op_results[most_common_op]
        target_result = ("|a - b|", False, False, ref[3])

    if target_result is not None:
        func_name, rev_ops, rev_res, fmt = target_result
        tqa = qa[::-1] if rev_ops else qa
        tqb = qb[::-1] if rev_ops else qb

        raw = None
        if unseen_query_op:
            raw = str(abs(int(tqa) - int(tqb)))
        else:
            query_cands = {name: val for name, val in _all_digit_funcs(tqa, tqb)}
            if func_name in query_cands:
                raw = query_cands[func_name]

        if raw is not None:
            final_raw = _rev(raw) if rev_res else raw
            answer = _encode_output(final_raw, fmt, qop)

            reasoning = [
                "We need to infer the transformation rule from the examples.",
                "",
                "Examples:",
            ]
            for A, op, B, rhs in rules:
                reasoning.append(f"  {A}{op}{B} = {rhs}")
            reasoning.append("")

            all_outputs = [rhs for _, _, _, rhs in rules]
            reasoning.append(f"The outputs are {', '.join(all_outputs)}")
            if fmt != "num":
                reasoning.append(f"Output format: {fmt} (negative values use operator as sign indicator)")
            reasoning.append("")

            # Show operator analysis with candidates tried
            for oc, (fn, ro, rr, fm) in sorted(op_results.items()):
                grp = op_groups[oc]
                _, transformed_grp = _detect_output_format(oc, grp)
                examples_str = ", ".join(f"{a}{oc}{b} = {out}" for a, b, out in transformed_grp)
                reasoning.append(f"Looking at operator '{oc}' [{examples_str}]:")

                first_a, first_b, first_exp = transformed_grp[0]
                ta = first_a[::-1] if ro else first_a
                tb = first_b[::-1] if ro else first_b

                if ro:
                    label = "  Trying operations reversed operands"
                    if rr:
                        label += " and reversed result"
                elif rr:
                    label = "  Trying operations reversed result"
                else:
                    label = "  Trying operations on identity"
                reasoning.append(label + ":")

                # Show candidates from all funcs until we find the correct one
                shown = 0
                for cname, cval in _all_digit_funcs(ta, tb):
                    cfin = _rev(cval) if rr else cval
                    all_match = True
                    for ea, eb, eexp in transformed_grp:
                        tea = ea[::-1] if ro else ea
                        teb = eb[::-1] if ro else eb
                        for cn2, cv2 in _all_digit_funcs(tea, teb):
                            if cn2 == cname:
                                ef = _rev(cv2) if rr else cv2
                                if ef != eexp:
                                    all_match = False
                                break
                    status = "match" if all_match else "wrong"
                    reasoning.append(f"    {cname} f({ta}, {tb}) = {cfin} {status}")
                    if all_match and cname == fn:
                        extras = []
                        if ro:
                            extras.append("reversed operands")
                        if rr:
                            extras.append("reversed result")
                        extras.append(cname)
                        reasoning.append(f"    correct, actions: {', '.join(extras)}")
                        break
                    shown += 1
                    if shown >= 8:
                        # Skip to show the correct answer
                        for cn3, cv3 in _all_digit_funcs(ta, tb):
                            if cn3 == fn:
                                cf3 = _rev(cv3) if rr else cv3
                                reasoning.append(f"    ... {fn} f({ta}, {tb}) = {cf3} match")
                                extras2 = []
                                if ro:
                                    extras2.append("reversed operands")
                                if rr:
                                    extras2.append("reversed result")
                                extras2.append(fn)
                                reasoning.append(f"    correct, actions: {', '.join(extras2)}")
                                break
                        break
                reasoning.append("")

            if unseen_query_op:
                reasoning.append(f"Query operator '{qop}' not in examples → default to |a - b|")
                reasoning.append("")

            reasoning.append(f"Applying to {qa}{qop}{qb}:")
            if rev_ops:
                reasoning.append(f"  reversed operands [{qa}->{tqa}, {qb}->{tqb}]")
            reasoning.append(f"  {func_name} f({tqa}, {tqb}) = {raw}")
            if rev_res:
                reasoning.append(f"  reversed result: {final_raw}")
            if fmt != "num" and answer != final_raw:
                reasoning.append(f"  re-encoded ({fmt}): {answer}")
            reasoning.append(f"  Result: {answer}")
            return answer, "\n".join(reasoning)

    # --- Strategy 2: single function across all rules (common first, then rare) ---
    for candidate_fn in (_common_digit_funcs, _rare_digit_funcs):
        for rev_ops, rev_res in _REV_COMBOS:
            common_all: set[str] | None = None
            for A, _op, B, rhs in rules:
                ta = A[::-1] if rev_ops else A
                tb = B[::-1] if rev_ops else B
                matching = set()
                for name, val in candidate_fn(ta, tb):
                    final = _rev(val) if rev_res else val
                    if final == rhs:
                        matching.add(name)
                if not matching:
                    common_all = None
                    break
                common_all = matching if common_all is None else (common_all & matching)
                if not common_all:
                    break

            if not common_all:
                continue

            # Pick the first (highest priority) match
            first_a, _, first_b, _ = rules[0]
            ta0 = first_a[::-1] if rev_ops else first_a
            tb0 = first_b[::-1] if rev_ops else first_b
            chosen = None
            for name, _ in candidate_fn(ta0, tb0):
                if name in common_all:
                    chosen = name
                    break
            if chosen is None:
                continue

            tqa = qa[::-1] if rev_ops else qa
            tqb = qb[::-1] if rev_ops else qb
            query_cands = {name: val for name, val in _all_digit_funcs(tqa, tqb)}
            if chosen not in query_cands:
                continue
            raw = query_cands[chosen]
            answer = _rev(raw) if rev_res else raw

            reasoning_lines = [
                "We need to infer the transformation rule from the examples.",
                "",
                "Examples:",
            ]
            for A, op, B, rhs in rules:
                reasoning_lines.append(f"  {A}{op}{B} = {rhs}")
            reasoning_lines.append("")
            reasoning_lines.append("Searching for a single function consistent with all examples.")
            if rev_ops:
                reasoning_lines.append("Reversed operands.")
            if rev_res:
                reasoning_lines.append("Reversed result.")

            # Show candidates tried
            first_a_s2, _, first_b_s2, _ = rules[0]
            ta0_s2 = first_a_s2[::-1] if rev_ops else first_a_s2
            tb0_s2 = first_b_s2[::-1] if rev_ops else first_b_s2
            shown = 0
            for cname, cval in candidate_fn(ta0_s2, tb0_s2):
                cfin = _rev(cval) if rev_res else cval
                all_match = cname in common_all
                first_exp = rules[0][3]
                status = "match" if cfin == first_exp else "wrong"
                reasoning_lines.append(f"  {cname} f({ta0_s2}, {tb0_s2}) = {cfin} {status}")
                if cname == chosen:
                    reasoning_lines.append(f"  correct, actions: {chosen}")
                    break
                shown += 1
                if shown >= 8:
                    break
            reasoning_lines.append("")
            reasoning_lines.append(f"Applying to {qa}{qop}{qb}:")
            if rev_ops:
                reasoning_lines.append(f"  reversed operands [{qa}->{tqa}, {qb}->{tqb}]")
            reasoning_lines.append(f"  {chosen} f({tqa}, {tqb}) = {answer}")
            reasoning_lines.append(f"  Result: {answer}")
            return answer, "\n".join(reasoning_lines)

    return None


_CRYPT_OPS: list[tuple[str, object]] = [
    ("add", lambda a, b: a + b),
    ("abs_diff", lambda a, b: abs(a - b)),
    ("mul", lambda a, b: a * b),
    ("concat", lambda a, b: a * 100 + b),
    ("rev_concat", lambda a, b: b * 100 + a),
    ("sub", lambda a, b: a - b),
    ("rsub", lambda a, b: b - a),
    ("xor", lambda a, b: a ^ b),
    ("and", lambda a, b: a & b),
    ("or", lambda a, b: a | b),
    ("max", lambda a, b: max(a, b)),
    ("min", lambda a, b: min(a, b)),
    ("dw_add", lambda a, b: ((a // 10 + b // 10) % 10) * 10 + ((a % 10 + b % 10) % 10)),
    ("dw_sub", lambda a, b: ((a // 10 - b // 10) % 10) * 10 + ((a % 10 - b % 10) % 10)),
    ("dw_mul", lambda a, b: ((a // 10 * (b // 10)) % 10) * 10 + ((a % 10 * (b % 10)) % 10)),
    ("dw_xor", lambda a, b: ((a // 10) ^ (b // 10)) * 10 + ((a % 10) ^ (b % 10))),
]


_CRYPT_DW_OPS: list[tuple[str, object]] = [
    ("dw_add", lambda a, b: ((a // 10 + b // 10) % 10) * 10 + ((a % 10 + b % 10) % 10)),
    ("dw_sub", lambda a, b: ((a // 10 - b // 10) % 10) * 10 + ((a % 10 - b % 10) % 10)),
    ("dw_mul", lambda a, b: ((a // 10 * (b // 10)) % 10) * 10 + ((a % 10 * (b % 10)) % 10)),
    ("dw_xor", lambda a, b: ((a // 10) ^ (b // 10)) * 10 + ((a % 10) ^ (b % 10))),
]

_BASIC_OP_COUNT = 5  # first 5 ops are the "basic" set

def _feasible_op_ids(rlen: int, *, extended: bool = False) -> list[int]:
    """Return op indices feasible for a result of *rlen* symbols."""
    ops: list[int] = []
    if rlen <= 3:
        ops.append(0)   # add: max 198 → 3 digits
    if rlen <= 2:
        ops.append(1)   # abs_diff: max 99
    if rlen <= 4:
        ops.append(2)   # mul: max 9801
    if rlen == 4:
        ops.extend([3, 4])  # concat / rev_concat
    if not extended:
        return ops
    if rlen <= 2:
        ops.extend([5, 6])  # sub/rsub: |val| ≤ 99
    if rlen <= 3:
        ops.append(7)   # xor: max 127
    if rlen <= 2:
        ops.append(8)   # and: max 99
    if rlen <= 3:
        ops.append(9)   # or: max 127
    if rlen <= 2:
        ops.extend([10, 11])  # max/min: max 99
    if rlen <= 2:
        ops.extend([12, 13, 14, 15])  # dw_add/sub/mul/xor: always 2 digits
    return ops


def _num_to_digits(n: int) -> tuple[int, ...]:
    if n == 0:
        return (0,)
    d: list[int] = []
    while n > 0:
        d.append(n % 10)
        n //= 10
    return tuple(reversed(d))


def _is_concat(s0: str, s1: str, s3: str, s4: str, rsyms: tuple[str, ...]) -> bool:
    return rsyms == (s0, s1, s3, s4) or rsyms == (s3, s4, s0, s1)


def _order_examples(
    examples: list[tuple[str, str, str, str, str, tuple[str, ...]]],
) -> list[tuple[str, str, str, str, str, tuple[str, ...]]]:
    """Reorder examples so the most constrained (fewest new symbols) are processed first."""
    remaining = list(range(len(examples)))
    ordered: list[int] = []
    seen_syms: set[str] = set()

    while remaining:
        best_idx = -1
        best_score = (999, 999)
        for i in remaining:
            ex = examples[i]
            all_syms = {ex[0], ex[1], ex[3], ex[4]} | set(ex[5])
            new_syms = len(all_syms - seen_syms)
            total_syms = len(all_syms)
            score = (new_syms, total_syms)
            if score < best_score:
                best_score = score
                best_idx = i
        ordered.append(best_idx)
        remaining.remove(best_idx)
        ex = examples[best_idx]
        seen_syms |= {ex[0], ex[1], ex[3], ex[4]} | set(ex[5])

    return [examples[i] for i in ordered]


class _CryptSolver:
    """Backtracking solver with example reordering."""

    MAX_SOLUTIONS = 200
    MAX_STEPS = 500_000

    def __init__(
        self,
        examples: list[tuple[str, str, str, str, str, tuple[str, ...]]],
        query: tuple[str, str, str, str, str],
        unique: bool = True,
        extended: bool = False,
    ):
        self.examples = _order_examples(examples)
        self.query = query
        self.unique = unique
        self.extended = extended
        self.mapping: dict[str, int] = {}
        self.used: set[int] = set()
        self.op_assign: dict[str, int] = {}
        self.answers: dict[str, int] = {}
        self.answer_info: dict[str, tuple[dict[str, int], dict[str, str]]] = {}
        self._steps = 0

    def solve(self) -> tuple[str | None, tuple[dict[str, int], dict[str, str]]]:
        self._process(0)
        if self.answers:
            best = max(self.answers, key=self.answers.get)  # type: ignore[arg-type]
            total = sum(self.answers.values())
            if not self.unique and total > 1 and self.answers[best] < total * 0.3:
                return None, ({}, {})
            return best, self.answer_info.get(best, ({}, {}))
        return None, ({}, {})

    def _forward_check(self, from_idx: int) -> bool:
        """Check fully-assigned future examples for immediate contradictions."""
        for i in range(from_idx, len(self.examples)):
            s0, s1, op_sym, s3, s4, rsyms = self.examples[i]
            if not (s0 in self.mapping and s1 in self.mapping
                    and s3 in self.mapping and s4 in self.mapping):
                continue
            lv = self.mapping[s0] * 10 + self.mapping[s1]
            rv = self.mapping[s3] * 10 + self.mapping[s4]
            rlen = len(rsyms)

            if op_sym in self.op_assign:
                ops_to_try = [self.op_assign[op_sym]]
            else:
                ops_to_try = _feasible_op_ids(rlen, extended=self.extended)

            any_feasible = False
            for op_id in ops_to_try:
                result_val = _CRYPT_OPS[op_id][1](lv, rv)
                if result_val < 0:
                    continue
                if op_id in (3, 4):
                    if result_val >= 10000:
                        continue
                    rd = (
                        result_val // 1000,
                        (result_val // 100) % 10,
                        (result_val // 10) % 10,
                        result_val % 10,
                    )
                else:
                    rd = _num_to_digits(result_val)
                if len(rd) != rlen:
                    continue
                ok = True
                for rs, rdig in zip(rsyms, rd):
                    if rs in self.mapping and self.mapping[rs] != rdig:
                        ok = False
                        break
                if ok:
                    any_feasible = True
                    break
            if not any_feasible:
                return False
        return True

    def _process(self, idx: int) -> None:
        self._steps += 1
        if len(self.answers) >= self.MAX_SOLUTIONS:
            return
        if self._steps > self.MAX_STEPS:
            return
        if idx == len(self.examples):
            self._compute_query()
            return

        s0, s1, op_sym, s3, s4, rsyms = self.examples[idx]
        rlen = len(rsyms)

        feasible_ops = _feasible_op_ids(rlen, extended=self.extended)

        for d0 in self._vals(s0):
            n0 = self._assign(s0, d0)
            if n0 is None:
                continue
            for d1 in self._vals(s1):
                n1 = self._assign(s1, d1)
                if n1 is None:
                    continue
                lv = d0 * 10 + d1
                for d3 in self._vals(s3):
                    n3 = self._assign(s3, d3)
                    if n3 is None:
                        continue
                    for d4 in self._vals(s4):
                        n4 = self._assign(s4, d4)
                        if n4 is None:
                            continue
                        rv = d3 * 10 + d4

                        ops_to_try = (
                            [self.op_assign[op_sym]]
                            if op_sym in self.op_assign
                            else feasible_ops
                        )

                        for op_id in ops_to_try:
                            result_val = _CRYPT_OPS[op_id][1](lv, rv)
                            if result_val < 0:
                                continue
                            if op_id in (3, 4):
                                if result_val >= 10000:
                                    continue
                                rd = (
                                    result_val // 1000,
                                    (result_val // 100) % 10,
                                    (result_val // 10) % 10,
                                    result_val % 10,
                                )
                            else:
                                rd = _num_to_digits(result_val)
                            if len(rd) != rlen:
                                continue

                            assigns: list[tuple[str, object]] = []
                            ok = True
                            for rs, rdig in zip(rsyms, rd):
                                ns = self._assign(rs, rdig)
                                if ns is None:
                                    ok = False
                                    break
                                assigns.append((rs, ns))

                            if ok:
                                op_new = op_sym not in self.op_assign
                                if op_new:
                                    self.op_assign[op_sym] = op_id
                                if self._forward_check(idx + 1):
                                    self._process(idx + 1)
                                if op_new:
                                    del self.op_assign[op_sym]

                            for rs, ns in reversed(assigns):
                                self._undo(rs, ns)

                            if len(self.answers) >= self.MAX_SOLUTIONS:
                                self._undo(s4, n4)
                                self._undo(s3, n3)
                                self._undo(s1, n1)
                                self._undo(s0, n0)
                                return

                        self._undo(s4, n4)
                    self._undo(s3, n3)
                self._undo(s1, n1)
            self._undo(s0, n0)

    def _vals(self, sym: str) -> range | tuple[int, ...]:
        if sym in self.mapping:
            return (self.mapping[sym],)
        if self.unique:
            return tuple(d for d in range(10) if d not in self.used)
        return range(10)

    def _assign(self, sym: str, dig: int) -> bool | None:
        if sym in self.mapping:
            return False if self.mapping[sym] == dig else None
        if self.unique and dig in self.used:
            return None
        self.mapping[sym] = dig
        if self.unique:
            self.used.add(dig)
        return True

    def _undo(self, sym: str, was_new: object) -> None:
        if was_new is True:
            if self.unique:
                self.used.discard(self.mapping[sym])
            del self.mapping[sym]

    def _compute_query(self) -> None:
        qs0, qs1, qop, qs3, qs4 = self.query
        for s in (qs0, qs1, qs3, qs4):
            if s not in self.mapping:
                return

        ql = self.mapping[qs0] * 10 + self.mapping[qs1]
        qr = self.mapping[qs3] * 10 + self.mapping[qs4]
        op_candidates: list[int] | range
        if qop in self.op_assign:
            op_candidates = [self.op_assign[qop]]
        else:
            op_candidates = range(len(_CRYPT_OPS))

        d2s: dict[int, str] = {}
        for s, d in self.mapping.items():
            if d not in d2s:
                d2s[d] = s

        for op_id in op_candidates:
            result_val = _CRYPT_OPS[op_id][1](ql, qr)
            if result_val < 0:
                continue
            if op_id in (3, 4):
                if result_val >= 10000:
                    continue
                rd = (
                    result_val // 1000,
                    (result_val // 100) % 10,
                    (result_val // 10) % 10,
                    result_val % 10,
                )
            else:
                rd = _num_to_digits(result_val)

            parts: list[str] = []
            ok = True
            for d in rd:
                if d not in d2s:
                    ok = False
                    break
                parts.append(d2s[d])
            if not ok:
                continue

            ans = "".join(parts)
            self.answers[ans] = self.answers.get(ans, 0) + 1
            if ans not in self.answer_info:
                op_info = {k: _CRYPT_OPS[v][0] for k, v in self.op_assign.items()}
                op_info[qop] = _CRYPT_OPS[op_id][0]
                self.answer_info[ans] = (dict(self.mapping), op_info)


def _try_cryptarithm_deduce(
    rules: list[tuple[str, str]], query: str
) -> tuple[str, str] | None:
    """Try cryptarithm deduction: each symbol maps to a unique digit, operators to operations."""
    if not rules or not query or len(query) != 5:
        return None
    for lhs, rhs in rules:
        if len(lhs) != 5:
            return None
        if any(c.isdigit() for c in lhs):
            return None

    examples: list[tuple[str, str, str, str, str, tuple[str, ...]]] = []
    concat_ops: set[str] = set()
    nonconcat_ops: set[str] = set()

    for lhs, rhs in rules:
        s0, s1, op, s3, s4 = lhs[0], lhs[1], lhs[2], lhs[3], lhs[4]
        rsyms = tuple(rhs)
        ex = (s0, s1, op, s3, s4, rsyms)
        examples.append(ex)
        if _is_concat(s0, s1, s3, s4, rsyms):
            concat_ops.add(op)
        else:
            nonconcat_ops.add(op)

    q_s0, q_s1, q_op, q_s3, q_s4 = query[0], query[1], query[2], query[3], query[4]
    q_tuple = (q_s0, q_s1, q_op, q_s3, q_s4)

    if q_op in concat_ops and q_op not in nonconcat_ops:
        for ex in examples:
            if ex[2] == q_op and _is_concat(ex[0], ex[1], ex[3], ex[4], ex[5]):
                if ex[5] == (ex[0], ex[1], ex[3], ex[4]):
                    answer = q_s0 + q_s1 + q_s3 + q_s4
                else:
                    answer = q_s3 + q_s4 + q_s0 + q_s1
                return answer, f"Concat operator: {answer}"
        answer = q_s0 + q_s1 + q_s3 + q_s4
        return answer, f"Concat operator (default fwd): {answer}"

    arith_examples = [ex for ex in examples if not _is_concat(ex[0], ex[1], ex[3], ex[4], ex[5])]
    if not arith_examples:
        answer = q_s0 + q_s1 + q_s3 + q_s4
        return answer, f"All concat, default fwd: {answer}"

    solver = _CryptSolver(arith_examples, q_tuple, unique=True)
    ans, (mapping, op_info) = solver.solve()
    if ans is None:
        solver2 = _CryptSolver(arith_examples, q_tuple, unique=False)
        ans, (mapping, op_info) = solver2.solve()
    if ans is not None:
        _OP_FUNCS_FOR_COT = {
            "add": lambda a, b: f"{a} + {b} = {a + b}",
            "abs_diff": lambda a, b: f"|{a} - {b}| = {abs(a - b)}",
            "mul": lambda a, b: f"{a} * {b} = {a * b}",
            "concat": lambda a, b: f"concat({a}, {b}) = {a * 100 + b}",
            "rev_concat": lambda a, b: f"rev_concat({a}, {b}) = {b * 100 + a}",
            "sub": lambda a, b: f"{a} - {b} = {a - b}",
            "rsub": lambda a, b: f"{b} - {a} = {b - a}",
            "xor": lambda a, b: f"{a} XOR {b} = {a ^ b}",
            "and": lambda a, b: f"{a} AND {b} = {a & b}",
            "or": lambda a, b: f"{a} OR {b} = {a | b}",
            "max": lambda a, b: f"max({a}, {b}) = {max(a, b)}",
            "min": lambda a, b: f"min({a}, {b}) = {min(a, b)}",
            "dw_add": lambda a, b: f"dw_add({a}, {b}) = {((a//10+b//10)%10)*10+((a%10+b%10)%10)}",
            "dw_sub": lambda a, b: f"dw_sub({a}, {b}) = {((a//10-b//10)%10)*10+((a%10-b%10)%10)}",
            "dw_mul": lambda a, b: f"dw_mul({a}, {b}) = {((a//10*(b//10))%10)*10+((a%10*(b%10))%10)}",
            "dw_xor": lambda a, b: f"dw_xor({a}, {b}) = {((a//10)^(b//10))*10+((a%10)^(b%10))}",
        }
        reasoning_parts = [
            "We need to infer the transformation rule from the examples.",
            "This is a cryptarithm: each symbol maps to a unique digit (0-9),",
            "each operator maps to an arithmetic operation.",
            "",
            "Examples:",
        ]
        for lhs, rhs in rules:
            reasoning_parts.append(f"  {lhs} = {rhs}")
        reasoning_parts.append("")
        reasoning_parts.append("Symbol-to-digit mapping:")
        for s, d in sorted(mapping.items()):
            reasoning_parts.append(f"  '{s}' = {d}")
        reasoning_parts.append("")
        reasoning_parts.append("Operator-to-operation mapping:")
        for s, name in sorted(op_info.items()):
            reasoning_parts.append(f"  '{s}' = {name}")
        reasoning_parts.append("")

        # Verify against each example
        reasoning_parts.append("Verification:")
        for lhs, rhs in rules:
            s0, s1, op_s, s3, s4 = lhs[0], lhs[1], lhs[2], lhs[3], lhs[4]
            if all(s in mapping for s in [s0, s1, s3, s4]):
                lv = mapping[s0] * 10 + mapping[s1]
                rv = mapping[s3] * 10 + mapping[s4]
                op_name = op_info.get(op_s, "?")
                if op_name in _OP_FUNCS_FOR_COT:
                    calc = _OP_FUNCS_FOR_COT[op_name](lv, rv)
                    reasoning_parts.append(f"  {lhs} = {rhs}  =>  {calc}")
                else:
                    reasoning_parts.append(f"  {lhs} = {rhs}")
            else:
                reasoning_parts.append(f"  {lhs} = {rhs}")
        reasoning_parts.append("")

        # Show query computation
        reasoning_parts.append(f"Query: {query}")
        ql = mapping.get(q_s0, -1) * 10 + mapping.get(q_s1, -1)
        qr = mapping.get(q_s3, -1) * 10 + mapping.get(q_s4, -1)
        q_op_name = op_info.get(q_op, "?")
        if q_op_name in _OP_FUNCS_FOR_COT:
            calc = _OP_FUNCS_FOR_COT[q_op_name](ql, qr)
            reasoning_parts.append(f"  {calc}")
        reasoning_parts.append(f"  Result: {ans}")
        return ans, "\n".join(reasoning_parts)

    return None


def _extract_digit_rules_and_query(
    prompt: str,
) -> tuple[list[tuple[str, str, str, str]], tuple[str, str, str]] | None:
    rules: list[tuple[str, str, str, str]] = []
    for line in prompt.splitlines():
        m = _DIGIT_LINE_PAT.match(line.strip())
        if m:
            rules.append((m.group(1), m.group(2), m.group(3), m.group(4).strip()))
    qm = _DIGIT_QUERY_PAT.search(prompt)
    if not qm or len(rules) < 2:
        return None
    return rules, (qm.group(1), qm.group(2), qm.group(3))


def _build_crypt_context(
    rules: list[tuple[str, str]], query: str
) -> dict | None:
    """Parse cryptarithm structure (5-symbol LHS, no digits)."""
    if not rules or not query or len(query) != 5:
        return None
    for lhs, _rhs in rules:
        if len(lhs) != 5:
            return None
        if any(c.isdigit() for c in lhs):
            return None

    examples: list[tuple[str, str, str, str, str, tuple[str, ...]]] = []
    concat_ops: set[str] = set()
    nonconcat_ops: set[str] = set()
    example_ops: set[str] = set()

    for lhs, rhs in rules:
        s0, s1, op, s3, s4 = lhs[0], lhs[1], lhs[2], lhs[3], lhs[4]
        rsyms = tuple(rhs)
        ex = (s0, s1, op, s3, s4, rsyms)
        examples.append(ex)
        example_ops.add(op)
        if _is_concat(s0, s1, s3, s4, rsyms):
            concat_ops.add(op)
        else:
            nonconcat_ops.add(op)

    q_op = query[2]
    arith_examples = [
        ex for ex in examples if not _is_concat(ex[0], ex[1], ex[3], ex[4], ex[5])
    ]
    is_crypt_concat_query = q_op in concat_ops and q_op not in nonconcat_ops
    return {
        "query_tuple": (query[0], query[1], q_op, query[3], query[4]),
        "example_ops": example_ops,
        "query_op": q_op,
        "concat_ops": concat_ops,
        "nonconcat_ops": nonconcat_ops,
        "arith_examples": arith_examples,
        "is_crypt_concat_query": is_crypt_concat_query,
        "is_crypt_arithmetic": bool(arith_examples),
    }


def analyze_symbol_equation(prompt: str) -> dict:
    """Classify symbol-equation family and query style for SFT tiering."""
    digit = _extract_digit_rules_and_query(prompt)
    if digit is not None:
        rules, (qa, qop, qb) = digit
        example_ops = {op for _, op, _, _ in rules}
        return {
            "family": "digit",
            "is_guess": qop not in example_ops,
            "is_deduce": qop in example_ops,
            "is_crypt_concat_query": False,
            "is_crypt_arithmetic": False,
            "query_op": qop,
            "example_ops": example_ops,
        }

    rules, query = _parse_symbol_rules(prompt)
    crypt = _build_crypt_context(rules, query) if rules and query else None
    if crypt is not None:
        return {
            "family": "crypt",
            "is_guess": crypt["query_op"] not in crypt["example_ops"],
            "is_deduce": crypt["query_op"] in crypt["example_ops"],
            "is_crypt_concat_query": crypt["is_crypt_concat_query"],
            "is_crypt_arithmetic": crypt["is_crypt_arithmetic"],
            "query_op": crypt["query_op"],
            "example_ops": crypt["example_ops"],
            "_crypt_ctx": crypt,
        }

    return {
        "family": "substitution",
        "is_guess": False,
        "is_deduce": True,
        "is_crypt_concat_query": False,
        "is_crypt_arithmetic": False,
        "query_op": None,
        "example_ops": set(),
    }


def count_crypt_arithmetic_solutions(prompt: str) -> int | None:
    """Distinct query answers consistent with crypt arithmetic examples (unique digits)."""
    rules, query = _parse_symbol_rules(prompt)
    if not rules or not query:
        return None
    crypt = _build_crypt_context(rules, query)
    if crypt is None or not crypt["is_crypt_arithmetic"]:
        return None
    solver = _CryptSolver(crypt["arith_examples"], crypt["query_tuple"], unique=True)
    solver.solve()
    return len(solver.answers)


def symbol_equation_sft_tier(prompt: str, *, solver_correct: bool) -> str:
    """SFT policy: trusted | oracle_only | exclude."""
    if not solver_correct:
        return "oracle_only"

    meta = analyze_symbol_equation(prompt)

    if meta.get("is_guess"):
        return "oracle_only"

    if meta["family"] == "digit" and meta.get("is_deduce"):
        return "trusted"

    if meta.get("is_crypt_concat_query"):
        return "trusted"

    if meta["family"] == "crypt" and meta.get("is_crypt_arithmetic"):
        n_solutions = count_crypt_arithmetic_solutions(prompt)
        if n_solutions is not None and n_solutions > 1:
            return "exclude"

    return "oracle_only"


def _try_gold_crypt(
    rules: list[tuple[str, str]], query: str, gt: str,
) -> tuple[str, str] | None:
    """Gold-conditioned crypt solver: use _CryptSolver with the query→GT
    as an additional example, allowing each operator its own operation."""
    if len(query) != 5 or not gt:
        return None
    for lhs, _rhs in rules:
        if len(lhs) != 5:
            return None
        if any(c.isdigit() for c in lhs):
            return None

    from solvers.symbol_equation import _is_concat

    q_s0, q_s1, q_op, q_s3, q_s4 = query
    q_tuple = (q_s0, q_s1, q_op, q_s3, q_s4)

    examples: list[tuple[str, str, str, str, str, tuple[str, ...]]] = []
    for lhs, rhs in rules:
        ex = (lhs[0], lhs[1], lhs[2], lhs[3], lhs[4], tuple(rhs))
        examples.append(ex)

    gt_ex = (q_s0, q_s1, q_op, q_s3, q_s4, tuple(gt))
    augmented = examples + [gt_ex]

    arith = [e for e in augmented if not _is_concat(e[0], e[1], e[3], e[4], e[5])]
    if not arith:
        return None

    solver = _CryptSolver(arith, q_tuple, unique=True)
    ans, (mapping, op_info) = solver.solve()
    if ans is None:
        solver2 = _CryptSolver(arith, q_tuple, unique=False)
        ans, (mapping, op_info) = solver2.solve()

    if ans is not None and ans == gt:
        reasoning = [
            "Cryptarithm: each symbol maps to a digit, each operator to an operation.",
            "",
            "Symbol mapping:",
        ]
        for s, d in sorted(mapping.items()):
            reasoning.append(f"  '{s}' = {d}")
        reasoning.append("")
        reasoning.append("Operator mapping:")
        for s, name in sorted(op_info.items()):
            reasoning.append(f"  '{s}' = {name}")
        reasoning.append("")
        reasoning.append(f"Query: {query}")
        reasoning.append(f"  Result: {gt}")
        return gt, "\n".join(reasoning)

    return None


def _try_alice_solver(prompt: str, answer_hint: str | None = None) -> tuple[str, str] | None:
    """Use the AliceEquationSolver (variable-radix, rich op library)."""
    try:
        from solvers.alice_solver import AliceEquationSolver
    except ImportError:
        return None

    solver = AliceEquationSolver(prompt, search_level="normal", answer_hint=answer_hint)
    ans, details = solver.solve()
    if ans is None:
        return None

    mapping = details.get("mapping", {})
    ops = details.get("ops", {})
    mode = details.get("mode", "standard")
    category = details.get("category", "")

    reasoning = [
        "Cryptarithm: each symbol maps to a digit, each operator to an operation.",
        f"Mode: {mode}, Category: {category}",
        "",
        "Symbol mapping:",
    ]
    for s, d in sorted(mapping.items()):
        reasoning.append(f"  '{s}' = {d}")
    reasoning.append("")
    reasoning.append("Operator mapping:")
    for s, name in sorted(ops.items()):
        reasoning.append(f"  '{s}' = {name}")
    reasoning.append(f"\n  Result: {ans}")
    return ans, "\n".join(reasoning)


def _try_gold_digit(prompt: str, gt: str) -> tuple[str, str] | None:
    """Gold-conditioned digit solver: try all functions and format encodings."""
    rules: list[tuple[str, str, str, str]] = []
    for line in prompt.splitlines():
        m = _DIGIT_LINE_PAT.match(line.strip())
        if m:
            rules.append((m.group(1), m.group(2), m.group(3), m.group(4).strip()))
    qm = _DIGIT_QUERY_PAT.search(prompt)
    if not qm or len(rules) < 2:
        return None
    qa, qop, qb = qm.group(1), qm.group(2), qm.group(3)

    from collections import defaultdict
    op_groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for A, op, B, rhs in rules:
        op_groups[op].append((A, B, rhs))

    for candidate_fn in (_common_digit_funcs, _rare_digit_funcs):
        for rev_ops, rev_res in [(False, False), (True, True), (True, False), (False, True)]:
            tqa = qa[::-1] if rev_ops else qa
            tqb = qb[::-1] if rev_ops else qb
            for name, val in candidate_fn(tqa, tqb):
                final_raw = _rev(val) if rev_res else val
                for fmt in ("num", "neg_suffix", "neg_prefix"):
                    encoded = _encode_output(final_raw, fmt, qop)
                    if encoded == gt:
                        if qop in op_groups:
                            group = op_groups[qop]
                            fmt2, transformed = _detect_output_format(qop, group)
                            fn = _find_op_function(transformed, rev_ops, rev_res, candidate_fn)
                            if fn == name:
                                reasoning = f"Gold-matched function: {name}, format: {fmt}\nResult: {gt}"
                                return gt, reasoning
                        reasoning = f"Gold-matched function: {name}, format: {fmt}\nResult: {gt}"
                        return gt, reasoning
    return None


def solve_symbol_equation(prompt: str, *, answer_hint: str | None = None) -> tuple[str, str]:
    digit_attempt = _try_digit_arithmetic(prompt)
    if digit_attempt is not None:
        if answer_hint is None or digit_attempt[0] == answer_hint:
            return digit_attempt
        if answer_hint is not None:
            gold_digit = _try_gold_digit(prompt, answer_hint)
            if gold_digit is not None:
                return gold_digit
        return digit_attempt

    rules, query = _parse_symbol_rules(prompt)
    if not rules or not query:
        return "", "Could not parse symbol equation rules or query"

    crypt_attempt = _try_cryptarithm_deduce(rules, query)
    if crypt_attempt is not None:
        if answer_hint is None or crypt_attempt[0] == answer_hint:
            return crypt_attempt

    alice_attempt = _try_alice_solver(prompt, answer_hint)
    if alice_attempt is not None:
        if answer_hint is None or alice_attempt[0] == answer_hint:
            return alice_attempt

    mapping = _solve_mapping(rules)

    if mapping is not None:
        result = "".join(mapping.get(c, c) for c in query)
        if answer_hint is None or result == answer_hint:
            answer = result
            reasoning_lines = [
                "Analyzing the transformation rules to find character mapping (each symbol maps to a "
                "substring; concatenation equals the RHS):"
            ]
            for lhs, rhs in rules:
                reasoning_lines.append(f"  {lhs} = {rhs}")
            reasoning_lines.append("\nDerived character mapping:")
            for c in sorted(mapping.keys()):
                val = mapping[c] if mapping[c] else "ε (empty)"
                reasoning_lines.append(f"  '{c}' → {val}")
            reasoning_lines.append(f"\nApplying the mapping to the query string: {query}")
            reasoning_lines.append(f"Result: {answer}")
            return answer, "\n".join(reasoning_lines)

    if answer_hint is not None:
        augmented = list(rules) + [(query, answer_hint)]
        gold_mapping = _solve_mapping(augmented)
        if gold_mapping is not None:
            result = "".join(gold_mapping.get(c, c) for c in query)
            if result == answer_hint:
                reasoning_lines = [
                    "Analyzing the transformation rules to find character mapping (each symbol maps to a "
                    "substring; concatenation equals the RHS):"
                ]
                for lhs, rhs in rules:
                    reasoning_lines.append(f"  {lhs} = {rhs}")
                reasoning_lines.append("\nDerived character mapping:")
                for c in sorted(gold_mapping.keys()):
                    val = gold_mapping[c] if gold_mapping[c] else "ε (empty)"
                    reasoning_lines.append(f"  '{c}' → {val}")
                reasoning_lines.append(f"\nApplying the mapping to the query string: {query}")
                reasoning_lines.append(f"Result: {result}")
                return result, "\n".join(reasoning_lines)

        gold_crypt = _try_gold_crypt(rules, query, answer_hint)
        if gold_crypt is not None:
            return gold_crypt

        alice_result = _try_alice_solver(prompt, answer_hint)
        if alice_result is not None:
            return alice_result

    if mapping is not None:
        result_chars = []
        for c in query:
            if c in mapping:
                result_chars.append(mapping[c])
            else:
                result_chars.append(c)
        answer = "".join(result_chars)
        reasoning_lines = [
            "Analyzing the transformation rules to find character mapping (each symbol maps to a "
            "substring; concatenation equals the RHS):"
        ]
        for lhs, rhs in rules:
            reasoning_lines.append(f"  {lhs} = {rhs}")
        reasoning_lines.append("\nDerived character mapping:")
        for c in sorted(mapping.keys()):
            val = mapping[c] if mapping[c] else "ε (empty)"
            reasoning_lines.append(f"  '{c}' → {val}")
        reasoning_lines.append(f"\nApplying the mapping to the query string: {query}")
        reasoning_lines.append(f"Result: {answer}")
        return answer, "\n".join(reasoning_lines)

    return "", "Could not find consistent character mapping"
