"""Bit manipulation solver — reverse-engineer 8-bit binary transformations.

Uses a multi-layer approach:
1. Try simple single operations (XOR, NOT, shift, rotate, AND, OR with constants)
2. Try compositions of two single operations (op2(op1(x)))
3. Per-bit boolean function analysis (handles complex multi-bit dependencies)
"""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

MASK = 0xFF
# Cap inner iterations for two-op search to avoid hangs when many ops match the first pair.
MAX_TWO_OP_ITERATIONS = 500_000

N_BITS = 8
SYM_FAMILIES: tuple[str, ...] = ("XOR", "OR", "AND")
ASYM_FAMILIES: tuple[str, ...] = ("AND-NOT", "XOR-NOT", "OR-NOT")
PAIR_FAMILIES: tuple[str, ...] = SYM_FAMILIES + ASYM_FAMILIES
UNARY_FAMILIES: tuple[str, ...] = ("I", "NOT")
CONSTANT_FAMILIES: tuple[str, ...] = ("0", "1")
DEFAULT_FAMILY = "DEFAULT"
SECTION_ORDER: tuple[str, ...] = (
    "Identity",
    "NOT",
    "Constant",
    "AND",
    "OR",
    "XOR",
    "AND-NOT",
    "OR-NOT",
    "XOR-NOT",
)


@dataclass(frozen=True)
class _RuleCandidate:
    family: str
    primary: int | None
    secondary: int | None
    expr: str

    @property
    def is_default(self) -> bool:
        return self.family == DEFAULT_FAMILY


def _out_col_word(out: list[int], n_ex: int) -> int:
    """Hash a column: all-0, all-1, or 2+ encoded bitstring."""
    ones = sum(out)
    if ones == 0:
        return 0
    if ones == n_ex:
        return 1
    w = 2
    for e in range(n_ex):
        w = (w << 1) | out[e]
    return w


def _apply_sym_pair(
    a_col: list[int], b_col: list[int], fam: str, n_ex: int
) -> int:
    out = [0] * n_ex
    for e in range(n_ex):
        x, y = a_col[e], b_col[e]
        if fam == "AND":
            out[e] = x & y
        elif fam == "OR":
            out[e] = 1 if (x or y) else 0
        else:  # XOR
            out[e] = x ^ y
    return _out_col_word(out, n_ex)


def _apply_asym_pair(
    a_col: list[int], b_col: list[int], fam: str, n_ex: int
) -> int:
    out = [0] * n_ex
    for e in range(n_ex):
        x, y0 = a_col[e], b_col[e]
        y = 1 - y0
        if fam == "AND-NOT":
            out[e] = x & y
        elif fam == "OR-NOT":
            out[e] = 1 if (x or y) else 0
        else:  # XOR-NOT
            out[e] = x ^ y
    return _out_col_word(out, n_ex)


def _build_all_matches(
    in_cols: list[list[int]], out_cols: list[list[int]], n_ex: int
) -> dict[str, list[list[_RuleCandidate]]]:
    all_matches: dict[str, list[list[_RuleCandidate]]] = {
        name: [[] for _ in range(N_BITS)] for name in SECTION_ORDER
    }
    for out_idx in range(N_BITS):
        ocol = out_cols[out_idx]
        oword = _out_col_word(ocol, n_ex)
        for i_col, icol in enumerate(in_cols):
            if _out_col_word(icol, n_ex) == oword:
                all_matches["Identity"][out_idx].append(
                    _RuleCandidate("I", i_col, None, f"I{i_col}")
                )
        for i_col, icol in enumerate(in_cols):
            inv = [1 - t for t in icol]
            if _out_col_word(inv, n_ex) == oword:
                all_matches["NOT"][out_idx].append(
                    _RuleCandidate("NOT", i_col, None, f"NOT{i_col}")
                )
        if oword == 0:
            all_matches["Constant"][out_idx].append(
                _RuleCandidate("0", None, None, "C0")
            )
        if oword == 1:
            all_matches["Constant"][out_idx].append(
                _RuleCandidate("1", None, None, "C1")
            )

    for fam in SYM_FAMILIES:
        for circ_diff in range(1, N_BITS // 2 + 1):
            n_pairs = N_BITS // 2 if circ_diff == N_BITS // 2 else N_BITS
            for a in range(n_pairs):
                b = (a + circ_diff) % N_BITS
                col_word = _apply_sym_pair(
                    in_cols[a], in_cols[b], fam, n_ex
                )
                for out_idx in range(N_BITS):
                    if _out_col_word(out_cols[out_idx], n_ex) == col_word:
                        all_matches[fam][out_idx].append(
                            _RuleCandidate(fam, a, b, f"{fam}{a}{b}")
                        )
                        all_matches[fam][out_idx].append(
                            _RuleCandidate(fam, b, a, f"{fam}{b}{a}")
                        )

    for fam in ASYM_FAMILIES:
        for diff in range(1, N_BITS):
            for a in range(N_BITS):
                b = (a + diff) % N_BITS
                col_word = _apply_asym_pair(
                    in_cols[a], in_cols[b], fam, n_ex
                )
                for out_idx in range(N_BITS):
                    if _out_col_word(out_cols[out_idx], n_ex) == col_word:
                        all_matches[fam][out_idx].append(
                            _RuleCandidate(fam, a, b, f"{fam}{a}{b}")
                        )
    return all_matches


def _merge_per_bit_cands(
    all_matches: dict[str, list[list[_RuleCandidate]]]
) -> list[list[_RuleCandidate]]:
    merged: list[list[_RuleCandidate]] = [[] for _ in range(N_BITS)]
    for name in SECTION_ORDER:
        for b in range(N_BITS):
            merged[b].extend(all_matches[name][b])
    return merged


def _find_match(
    cands: list[_RuleCandidate],
    fam: str,
    ep: int | None,
    es: int | None,
) -> _RuleCandidate | None:
    for c in cands:
        if c.family != fam:
            continue
        if c.primary == ep and (fam not in PAIR_FAMILIES or c.secondary == es):
            return c
    return None


def _exists_anywhere(
    all_merged: list[list[_RuleCandidate]], fam: str, ep: int | None, es: int | None
) -> bool:
    for bit_cands in all_merged:
        if _find_match(bit_cands, fam, ep, es) is not None:
            return True
    return False


def _fail_suffix(
    all_merged: list[list[_RuleCandidate]], fam: str, ep: int | None, es: int | None
) -> str:
    if _exists_anywhere(all_merged, fam, ep, es):
        return "y"
    return "x"


def _find_all_left_runs(
    all_matches: list[list[_RuleCandidate]],
) -> list[tuple[list[_RuleCandidate], str | None]]:
    if not all_matches or not all_matches[0]:
        return []
    runs: list[tuple[list[_RuleCandidate], str | None]] = []
    p_step, s_step = 1, 1
    for start_cand in all_matches[0]:
        fam = start_cand.family
        chain = [start_cand]
        cur_p, cur_s = start_cand.primary, start_cand.secondary
        failed_next: str | None = None
        for b in range(1, len(all_matches)):
            ep = (cur_p + p_step) % N_BITS if cur_p is not None else None
            es = (cur_s + s_step) % N_BITS if cur_s is not None else None
            found = _find_match(all_matches[b], fam, ep, es)
            if found is None:
                if ep is not None and es is not None:
                    failed_next = f"{ep}{es}{_fail_suffix(all_matches, fam, ep, es)}"
                elif ep is not None:
                    failed_next = f"{ep}{_fail_suffix(all_matches, fam, ep, es)}"
                break
            chain.append(found)
            cur_p, cur_s = ep, es
        runs.append((chain, failed_next))
    return runs


def _find_all_right_runs(
    all_matches: list[list[_RuleCandidate]],
) -> list[tuple[list[_RuleCandidate], str | None]]:
    n = len(all_matches)
    if not all_matches or not all_matches[-1]:
        return []
    runs: list[tuple[list[_RuleCandidate], str | None]] = []
    p_step, s_step = 1, 1
    for end_cand in all_matches[-1]:
        fam = end_cand.family
        chain = [end_cand]
        cur_p, cur_s = end_cand.primary, end_cand.secondary
        failed_next: str | None = None
        for k in range(1, n):
            b = n - 1 - k
            pp = (cur_p - p_step) % N_BITS if cur_p is not None else None
            ps = (cur_s - s_step) % N_BITS if cur_s is not None else None
            found = _find_match(all_matches[b], fam, pp, ps)
            if found is None:
                if pp is not None and ps is not None:
                    failed_next = f"{pp}{ps}{_fail_suffix(all_matches, fam, pp, ps)}"
                elif pp is not None:
                    failed_next = f"{pp}{_fail_suffix(all_matches, fam, pp, ps)}"
                break
            chain.insert(0, found)
            cur_p, cur_s = pp, ps
        runs.append((chain, failed_next))
    return runs


def _chain_sort_key(t: tuple[list[_RuleCandidate], str | None]) -> tuple:
    ch = t[0]
    if not ch:
        return (0, ())
    return (len(ch), tuple(c.expr for c in ch))


def _extrap_from(
    run: list[_RuleCandidate],
    bit: int,
    run_start_bit: int,
    side: str = "left",
) -> str | None:
    if not run:
        return None
    r0 = run[0]
    p, s = r0.primary, r0.secondary
    if p is not None:
        p_off = (p - run_start_bit) % N_BITS
        ep = (p_off + bit) % N_BITS
    else:
        ep = None
    if s is not None:
        s_off = (s - run_start_bit) % N_BITS
        es = (s_off + bit) % N_BITS
    else:
        es = None
    if ep is not None and es is not None:
        return f"?{ep}{es}"
    if ep is not None:
        if side == "left":
            return f"?{ep}?"
        return f"??{ep}"
    return None


def _eval_rule_on_query(q: int, rule: _RuleCandidate) -> int:
    if rule.family in CONSTANT_FAMILIES:
        return 1 if rule.family == "1" else 0
    if rule.family == "I" and rule.primary is not None:
        return (q >> rule.primary) & 1
    if rule.family == "NOT" and rule.primary is not None:
        return 1 - ((q >> rule.primary) & 1)
    if (
        rule.family in PAIR_FAMILIES
        and rule.primary is not None
        and rule.secondary is not None
    ):
        a = (q >> rule.primary) & 1
        b0 = (q >> rule.secondary) & 1
        b = 1 - b0 if ("-NOT" in rule.family) else b0
        if rule.family in ("AND", "AND-NOT"):
            return a & b
        if rule.family in ("OR", "OR-NOT"):
            return 1 if (a or b) else 0
        return a ^ b
    if rule.is_default:
        return 1
    return 0


def _vector_matches_examples(rules: list[_RuleCandidate], examples: list[tuple[int, int]]) -> bool:
    for inp, out in examples:
        v = 0
        for i in range(N_BITS):
            v |= _eval_rule_on_query(inp, rules[i]) << i
        if v != (out & MASK):
            return False
    return True


def _parse_bit_examples(prompt: str) -> tuple[list[tuple[int, int]], int]:
    pairs = re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    examples = [(int(a, 2), int(b, 2)) for a, b in pairs]

    query_match = re.search(
        r"(?:determine|find|compute|calculate).*?:\s*([01]{8})", prompt
    )
    if not query_match:
        all_bins = re.findall(r"[01]{8}", prompt)
        query_val = int(all_bins[-1], 2) if all_bins else 0
    else:
        query_val = int(query_match.group(1), 2)

    return examples, query_val


def _rotate_left(x, n, bits=8):
    n %= bits
    return ((x << n) | (x >> (bits - n))) & ((1 << bits) - 1)


def _rotate_right(x, n, bits=8):
    n %= bits
    return ((x >> n) | (x << (bits - n))) & ((1 << bits) - 1)


def _reverse_bits(x, bits=8):
    result = 0
    for _ in range(bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result


def _single_ops():
    ops = []
    ops.append(("NOT", lambda x: (~x) & MASK))
    ops.append(("reverse_bits", lambda x: _reverse_bits(x)))
    ops.append(("swap_nibbles", lambda x: ((x >> 4) | ((x << 4) & MASK))))

    for n in range(1, 8):
        ops.append((f"rotl({n})", lambda x, n=n: _rotate_left(x, n)))
        ops.append((f"rotr({n})", lambda x, n=n: _rotate_right(x, n)))
        ops.append((f"shl({n})", lambda x, n=n: (x << n) & MASK))
        ops.append((f"shr({n})", lambda x, n=n: (x >> n) & MASK))

    for c in range(256):
        ops.append((f"XOR(0x{c:02x})", lambda x, c=c: x ^ c))
        ops.append((f"AND(0x{c:02x})", lambda x, c=c: x & c))
        ops.append((f"OR(0x{c:02x})", lambda x, c=c: x | c))
        ops.append((f"ADD(0x{c:02x})", lambda x, c=c: (x + c) & MASK))
        ops.append((f"SUB(0x{c:02x})", lambda x, c=c: (x - c) & MASK))

    ops.append(("identity", lambda x: x))
    return ops


def _test_op(op_func, examples):
    return all(op_func(inp) == out for inp, out in examples)


def _try_two_op_composition(
    examples: list[tuple[int, int]],
    query_input: int,
    single_ops: list[tuple[str, Callable[[int], int]]],
) -> tuple[str, str] | None:
    """Find f(x)=op2(op1(x)) where op1, op2 are from the single-op library.

    Speed strategy: enumerate every op1, compute its intermediates `mids`, then
    restrict op2 to the (typically small) set that maps mids[0] to out0 before
    checking the remaining examples. Filtering op2 by the first *intermediate*
    (not op1 by the first input) is what makes the search both correct and
    fast — the previous implementation pruned op1 by f(inp0)==out0, which
    silently dropped most valid compositions because it implicitly required
    op2(out0)==out0.
    """
    if not examples:
        return None
    inp0, out0 = examples[0]
    n_ex = len(examples)
    iterations = 0

    for name1, func1 in single_ops:
        mids = [func1(inp) for inp, _ in examples]
        mid0 = mids[0]
        for name2, func2 in single_ops:
            iterations += 1
            if iterations > MAX_TWO_OP_ITERATIONS:
                return None
            if func2(mid0) != out0:
                continue
            ok = True
            for i in range(1, n_ex):
                if func2(mids[i]) != examples[i][1]:
                    ok = False
                    break
            if not ok:
                continue
            q_mid = func1(query_input)
            result = func2(q_mid)
            answer = f"{result:08b}"
            reasoning_parts = [
                "Two-operation composition (applied as second(first(input))):",
                f"  First operation: {name1}",
                f"  Second operation (maps intermediates to all outputs): {name2}",
                "Step-by-step on the query:",
                f"  x = {query_input:08b}",
                f"  after {name1}: {q_mid:08b}",
                f"  after {name2}: {answer}",
            ]
            return answer, "\n".join(reasoning_parts)
    return None


def _enumerate_bit_functions(in_bits_arr, out_vals, query, n, out_bit):
    """Enumerate ALL valid boolean functions for one output bit (legacy)."""
    preds: dict[int, list[tuple[str, int]]] = {0: [], 1: []}
    if all(v == 0 for v in out_vals):
        preds[0].append(("const_0", -1))
    if all(v == 1 for v in out_vals):
        preds[1].append(("const_1", -1))
    for b in range(8):
        if all(in_bits_arr[b][i] == out_vals[i] for i in range(n)):
            val = (query >> b) & 1
            preds[val].append((f"bit{b}", b))
        if all((1 - in_bits_arr[b][i]) == out_vals[i] for i in range(n)):
            val = 1 - ((query >> b) & 1)
            preds[val].append((f"NOT(bit{b})", b))
    for b1 in range(8):
        for b2 in range(b1 + 1, 8):
            for fn_id in range(16):
                ok = True
                for i in range(n):
                    idx = in_bits_arr[b1][i] * 2 + in_bits_arr[b2][i]
                    if ((fn_id >> idx) & 1) != out_vals[i]:
                        ok = False
                        break
                if ok:
                    idx_q = ((query >> b1) & 1) * 2 + ((query >> b2) & 1)
                    val = (fn_id >> idx_q) & 1
                    preds[val].append((f"f{fn_id}(b{b1},b{b2})", b1))
    return preds


def _solve_per_bit_legacy(
    examples: list[tuple[int, int]], query: int
) -> tuple[str, str] | None:
    """Heuristic + 3-input fallback (previous solver)."""
    n = len(examples)
    in_bits_arr = [[(inp >> b) & 1 for inp, _ in examples] for b in range(8)]
    bit_results: list = [None] * 8
    bit_names = [""] * 8
    bit_confidence = [0] * 8
    bit_preds: list = [None] * 8
    for out_bit in range(8):
        out_vals = [(out >> out_bit) & 1 for _, out in examples]
        preds = _enumerate_bit_functions(in_bits_arr, out_vals, query, n, out_bit)
        bit_preds[out_bit] = preds
        has_0, has_1 = len(preds[0]) > 0, len(preds[1]) > 0
        if has_0 and not has_1:
            bit_results[out_bit] = 0
            bit_confidence[out_bit] = 2
            bit_names[out_bit] = preds[0][0][0]
        elif has_1 and not has_0:
            bit_results[out_bit] = 1
            bit_confidence[out_bit] = 2
            bit_names[out_bit] = preds[1][0][0]
    offset_counts: Counter = Counter()
    for out_bit in range(8):
        if bit_confidence[out_bit] != 2 or bit_preds[out_bit] is None:
            continue
        val = bit_results[out_bit]
        for name, primary_bit in bit_preds[out_bit][val]:
            if primary_bit >= 0:
                offset_counts[(primary_bit - out_bit) % 8] += 1
    best_offset = offset_counts.most_common(1)[0][0] if offset_counts else 0
    fn_id_counts: Counter = Counter()
    offset_pair_counts: Counter = Counter()
    for out_bit in range(8):
        if bit_confidence[out_bit] != 2 or bit_preds[out_bit] is None:
            continue
        val = bit_results[out_bit]
        for name, _ in bit_preds[out_bit][val]:
            if name.startswith("f") and "(" in name:
                fn_part = name.split("(")[0]
                try:
                    fn_id_counts[int(fn_part[1:])] += 1
                except ValueError:
                    pass
                try:
                    bits_part = name.split("(")[1].rstrip(")")
                    b_parts = bits_part.split(",")
                    if len(b_parts) == 2:
                        b1 = int(b_parts[0].strip()[1:])
                        b2 = int(b_parts[1].strip()[1:])
                        offset_pair_counts[
                            ((b1 - out_bit) % 8, (b2 - out_bit) % 8)
                        ] += 1
                except (ValueError, IndexError):
                    pass
    best_fn_ids = {fid for fid, _ in fn_id_counts.most_common(3)}
    best_offset_pair = (
        offset_pair_counts.most_common(1)[0][0] if offset_pair_counts else None
    )
    # Additional heuristic: count dominant families among confident bits
    family_counts: Counter = Counter()
    for out_bit in range(8):
        if bit_confidence[out_bit] != 2 or bit_preds[out_bit] is None:
            continue
        val = bit_results[out_bit]
        for name, _ in bit_preds[out_bit][val]:
            if name.startswith("bit") and not name.startswith("NOT"):
                family_counts["I"] += 1
            elif name.startswith("NOT"):
                family_counts["NOT"] += 1
            elif name.startswith("f") and "(" in name:
                try:
                    fid = int(name.split("(")[0][1:])
                    family_counts[fid] += 1
                except ValueError:
                    pass
            break  # Only count the first (chosen) pred

    for out_bit in range(8):
        if bit_confidence[out_bit] == 2:
            continue
        preds = bit_preds[out_bit]
        if preds is None:
            continue
        has_0, has_1 = len(preds[0]) > 0, len(preds[1]) > 0
        if not (has_0 and has_1):
            continue
        expected_input = (out_bit + best_offset) % 8
        exp_b1 = (out_bit + best_offset_pair[0]) % 8 if best_offset_pair else -1
        exp_b2 = (out_bit + best_offset_pair[1]) % 8 if best_offset_pair else -1
        combined = {0: 0, 1: 0}
        for val in (0, 1):
            for name, primary_bit in preds[val]:
                if primary_bit == expected_input:
                    combined[val] += 3
                # Simplicity bonus: prefer Identity/NOT over complex functions
                if name.startswith("bit") and primary_bit == expected_input:
                    combined[val] += 1
                elif name.startswith("NOT") and primary_bit == expected_input:
                    combined[val] += 1
                if name.startswith("f") and "(" in name:
                    try:
                        fid = int(name.split("(")[0][1:])
                        if fid in best_fn_ids:
                            combined[val] += 2
                        # Bonus for matching dominant family
                        if fid in family_counts and family_counts[fid] >= 2:
                            combined[val] += 2
                    except ValueError:
                        pass
                    if best_offset_pair is not None:
                        try:
                            bits_part = name.split("(")[1].rstrip(")")
                            b_parts = bits_part.split(",")
                            if len(b_parts) == 2:
                                pb1 = int(b_parts[0].strip()[1:])
                                pb2 = int(b_parts[1].strip()[1:])
                                if pb1 == exp_b1 and pb2 == exp_b2:
                                    combined[val] += 5
                                # Also try swapped operand pair
                                elif pb1 == exp_b2 and pb2 == exp_b1:
                                    combined[val] += 3
                        except (ValueError, IndexError):
                            pass
        if combined[0] != combined[1]:
            winner = 0 if combined[0] > combined[1] else 1
            bit_results[out_bit] = winner
            bit_names[out_bit] = preds[winner][0][0]
        elif len(preds[1]) >= len(preds[0]):
            bit_results[out_bit] = 1
            bit_names[out_bit] = preds[1][0][0]
        else:
            bit_results[out_bit] = 0
            bit_names[out_bit] = preds[0][0][0]
        bit_confidence[out_bit] = 1
    for out_bit in range(8):
        if bit_results[out_bit] is not None:
            continue
        out_vals = [(out >> out_bit) & 1 for _, out in examples]
        found_3 = False
        preds_3: dict[int, list[str]] = {0: [], 1: []}
        for b1 in range(8):
            for b2 in range(b1 + 1, 8):
                for b3 in range(b2 + 1, 8):
                    for fn_id in range(256):
                        ok = True
                        for i in range(n):
                            idx = (
                                in_bits_arr[b1][i] * 4
                                + in_bits_arr[b2][i] * 2
                                + in_bits_arr[b3][i]
                            )
                            if ((fn_id >> idx) & 1) != out_vals[i]:
                                ok = False
                                break
                        if ok:
                            idx_q = (
                                ((query >> b1) & 1) * 4
                                + ((query >> b2) & 1) * 2
                                + ((query >> b3) & 1)
                            )
                            val = (fn_id >> idx_q) & 1
                            preds_3[val].append(
                                f"f3_{fn_id}(b{b1},b{b2},b{b3})"
                            )
                            found_3 = True
        if not found_3:
            return None
        has_0 = len(preds_3[0]) > 0
        has_1 = len(preds_3[1]) > 0
        if has_0 and not has_1:
            bit_results[out_bit] = 0
            bit_names[out_bit] = preds_3[0][0]
        elif has_1 and not has_0:
            bit_results[out_bit] = 1
            bit_names[out_bit] = preds_3[1][0]
        elif has_0 and has_1:
            bit_results[out_bit] = (
                1 if len(preds_3[1]) >= len(preds_3[0]) else 0
            )
            bit_names[out_bit] = (
                preds_3[1] if bit_results[out_bit] else preds_3[0]
            )[0]
        else:
            return None
    if any(b is None for b in bit_results):
        return None
    result = sum(b << i for i, b in enumerate(bit_results))  # type: ignore
    parts = ["Per-bit analysis (legacy heuristic + 3-input if needed):"]
    for i in range(8):
        parts.append(
            f"  Output bit {i} = {bit_names[i]} → {bit_results[i]}"
        )
    parts.append(f"Result: {result:08b}")
    return f"{result:08b}", "\n".join(parts)


def _rule_complexity(rule: _RuleCandidate) -> int:
    """Lower is simpler (preferred when multiple vectors match examples)."""
    if rule.family in CONSTANT_FAMILIES:
        return 0
    if rule.family in UNARY_FAMILIES:
        return 1 + (rule.primary or 0)
    if rule.family in PAIR_FAMILIES:
        return 10 + (rule.primary or 0) + (rule.secondary or 0)
    if rule.is_default:
        return 100
    return 50


def _vector_score(rules: list[_RuleCandidate]) -> tuple[int, int, int]:
    """Sort key for choosing among valid rule vectors (min is best)."""
    complexity = sum(_rule_complexity(r) for r in rules)
    stride_breaks = 0
    families = 0
    prev: _RuleCandidate | None = None
    for r in rules:
        if prev is None or r.family != prev.family:
            families += 1
        elif r.family in UNARY_FAMILIES + PAIR_FAMILIES:
            if (
                r.primary is not None
                and prev.primary is not None
                and (r.primary - prev.primary) % N_BITS != 1
            ):
                stride_breaks += 1
            if (
                r.family in PAIR_FAMILIES
                and r.secondary is not None
                and prev.secondary is not None
                and (r.secondary - prev.secondary) % N_BITS != 1
            ):
                stride_breaks += 1
        prev = r
    return (stride_breaks, families, complexity)


def _best_section_run(
    all_matches: dict[str, list[list[_RuleCandidate]]], direction: str
) -> tuple[str, list[_RuleCandidate]]:
    """Longest stride-consistent run within one operation family."""
    best_name = ""
    best_run: list[_RuleCandidate] = []
    finder = (
        _find_all_left_runs if direction == "left" else _find_all_right_runs
    )
    for name in SECTION_ORDER:
        per_bit = all_matches[name]
        runs = finder(per_bit)
        if not runs:
            continue
        run = max(runs, key=_chain_sort_key)[0]
        if len(run) > len(best_run):
            best_run = run
            best_name = name
    return best_name, best_run


def _enumerate_valid_vectors(
    merged: list[list[_RuleCandidate]],
    examples: list[tuple[int, int]],
    max_solutions: int = 64,
) -> list[list[_RuleCandidate]]:
    """Backtracking search for rule vectors consistent with all examples."""
    solutions: list[list[_RuleCandidate]] = []

    def backtrack(bit_idx: int, chosen: list[_RuleCandidate]) -> None:
        if len(solutions) >= max_solutions:
            return
        if bit_idx == N_BITS:
            if _vector_matches_examples(chosen, examples):
                solutions.append(list(chosen))
            return
        for cand in merged[bit_idx]:
            chosen.append(cand)
            backtrack(bit_idx + 1, chosen)
            chosen.pop()

    backtrack(0, [])
    return solutions


def _solve_per_bit_search(
    examples: list[tuple[int, int]], query: int
) -> tuple[str, str] | None:
    """Find the simplest rule vector matching all examples, apply to query."""
    n_ex = len(examples)
    in_cols = [[(inp >> i) & 1 for inp, _ in examples] for i in range(N_BITS)]
    out_cols = [[(out >> i) & 1 for _, out in examples] for i in range(N_BITS)]
    all_matches = _build_all_matches(in_cols, out_cols, n_ex)
    merged = _merge_per_bit_cands(all_matches)
    if any(not merged[b] for b in range(N_BITS)):
        return None

    solutions = _enumerate_valid_vectors(merged, examples)
    if not solutions:
        return None

    best = min(solutions, key=_vector_score)
    result = 0
    for i in range(N_BITS):
        result |= _eval_rule_on_query(query, best[i]) << i
    answer = f"{result:08b}"
    lines = ["Per-bit search (validated rule vector):"]
    for bit_idx in range(N_BITS):
        lines.append(
            f"  Output bit {bit_idx} = {best[bit_idx].expr} ({best[bit_idx].family})"
        )
    lines.append(f"Result: {answer}")
    return answer, "\n".join(lines)


def _solve_per_bit_stride(examples: list[tuple[int, int]], query: int) -> tuple[str, str] | None:
    """Stride-consistent run detection, stride extrapolation, and perfect-match fallback."""
    n_ex = len(examples)
    in_cols = [[(inp >> i) & 1 for inp, _ in examples] for i in range(N_BITS)]
    out_cols = [[(out >> i) & 1 for _, out in examples] for i in range(N_BITS)]

    all_matches = _build_all_matches(in_cols, out_cols, n_ex)
    merged = _merge_per_bit_cands(all_matches)
    if any(not merged[b] for b in range(N_BITS)):
        return None

    left_winner_name, left_run = _best_section_run(all_matches, "left")
    right_winner_name, right_run = _best_section_run(all_matches, "right")
    left_winner_count = len(left_run)
    right_winner_count = len(right_run)
    if left_winner_count == 0 and right_winner_count == 0:
        return None

    left_len_final = left_winner_count
    right_len_final = right_winner_count
    if left_len_final + right_len_final > N_BITS:
        if right_len_final > left_len_final:
            left_len_final = N_BITS - right_len_final
            left_run = left_run[:left_len_final]
        else:
            right_len_final = N_BITS - left_len_final
            right_run = right_run[-right_len_final:] if right_len_final else []
    right_start_final = N_BITS - right_len_final

    left_fam = left_run[0].family if left_run else None
    right_fam = right_run[0].family if right_run else None
    left_is_binary = left_fam in PAIR_FAMILIES if left_fam else False
    right_is_binary = right_fam in PAIR_FAMILIES if right_fam else False
    left_is_unary = left_fam in UNARY_FAMILIES if left_fam else False
    right_is_unary = right_fam in UNARY_FAMILIES if right_fam else False

    default_cand = _RuleCandidate(DEFAULT_FAMILY, None, None, "default_1")
    preferred: list[str] = [""] * N_BITS
    if right_winner_count > left_winner_count:
        for i in range(N_BITS):
            if i >= right_start_final and right_run:
                preferred[i] = right_run[i - right_start_final].expr
            elif i < left_len_final and left_run:
                preferred[i] = left_run[i].expr
            elif right_is_binary or right_is_unary:
                preferred[i] = (
                    _extrap_from(
                        right_run, i, right_start_final, "right"
                    )
                    or "pending"
                )
            else:
                preferred[i] = "pending"
        for i in range(N_BITS):
            if preferred[i] == "pending":
                if left_is_binary or left_is_unary:
                    preferred[i] = (
                        _extrap_from(left_run, i, 0, "left") or "?"
                    )
                else:
                    preferred[i] = "?"
            elif len(preferred[i]) == 3 and "?" in preferred[i][1:] and left_is_unary:
                el = _extrap_from(left_run, i, 0, "left")
                if el:
                    merged_ch = list(preferred[i])
                    el_chars = list(el)
                    for j in range(1, min(len(merged_ch), len(el_chars))):
                        if merged_ch[j] == "?" and el_chars[j] != "?":
                            merged_ch[j] = el_chars[j]
                    preferred[i] = "".join(merged_ch)
    else:
        for i in range(N_BITS):
            if i < left_len_final and left_run:
                preferred[i] = left_run[i].expr
            elif i >= right_start_final and right_run:
                preferred[i] = right_run[i - right_start_final].expr
            elif left_is_binary or left_is_unary:
                preferred[i] = (
                    _extrap_from(left_run, i, 0, "left")
                    or "pending"
                )
            else:
                preferred[i] = "pending"
        for i in range(N_BITS):
            if preferred[i] == "pending":
                if right_is_binary or right_is_unary:
                    preferred[i] = (
                        _extrap_from(
                            right_run, i, right_start_final, "right"
                        )
                        or "?"
                    )
                else:
                    preferred[i] = "?"
            elif len(preferred[i]) == 3 and "?" in preferred[i][1:] and right_is_unary:
                er = _extrap_from(
                    right_run, i, right_start_final, "right"
                )
                if er:
                    mc = list(preferred[i])
                    ec = list(er)
                    for j in range(1, min(len(mc), len(ec))):
                        if mc[j] == "?" and ec[j] != "?":
                            mc[j] = ec[j]
                    preferred[i] = "".join(mc)

    best: list[_RuleCandidate] = [default_cand] * N_BITS
    for i, rc in enumerate(left_run):
        if i < N_BITS:
            best[i] = rc
    for j, rc in enumerate(right_run):
        idx = right_start_final + j
        if 0 <= idx < N_BITS:
            best[idx] = rc

    pending_indices: list[int] = []
    per_bit_cat: dict[str, dict[int, list[_RuleCandidate]]] = {
        name: {} for name in SECTION_ORDER
    }

    for i in range(N_BITS):
        pref = preferred[i]
        if not pref.startswith("?") or pref == "?":
            continue
        pending_indices.append(i)
        if len(pref) < 2:
            continue
        pref_digits = [int(d) for d in pref[1:] if d.isdigit()]

        for section_name in SECTION_ORDER:
            cands = all_matches[section_name][i]
            if section_name in ("Identity", "NOT"):
                found = [c for c in cands if c.primary in pref_digits]
                if found:
                    per_bit_cat[section_name][i] = found
            elif section_name == "Constant":
                if cands:
                    per_bit_cat["Constant"][i] = list(cands)
            else:
                found_c: _RuleCandidate | None = None
                want_p = int(pref[1]) if len(pref) > 1 and pref[1] != "?" else None
                want_s = int(pref[2]) if len(pref) > 2 and pref[2] != "?" else None
                orderings: list[tuple[int | None, int | None]] = [(want_p, want_s)]
                if want_p is not None and want_s is not None and want_p != want_s:
                    orderings.append((want_s, want_p))
                for wp, ws in orderings:
                    for c in cands:
                        if (wp is None or c.primary == wp) and (
                            ws is None or c.secondary == ws
                        ):
                            found_c = c
                            break
                    if found_c is not None:
                        break
                if found_c is not None:
                    per_bit_cat[section_name][i] = [found_c]

    chosen_cat: str | None = None
    for cat in SECTION_ORDER:
        is_perfect = bool(pending_indices) and all(
            i in per_bit_cat[cat] for i in pending_indices
        )
        if is_perfect and chosen_cat is None:
            chosen_cat = cat

    pending_set = set(pending_indices)
    for i in range(N_BITS):
        if i not in pending_set:
            continue
        if chosen_cat and i in per_bit_cat[chosen_cat]:
            best[i] = per_bit_cat[chosen_cat][i][0]
        else:
            all_cands: list[_RuleCandidate] = []
            for name in SECTION_ORDER:
                if i in per_bit_cat[name]:
                    all_cands.extend(per_bit_cat[name][i])
            if all_cands:
                best[i] = min(all_cands, key=_rule_complexity)
            elif merged[i]:
                best[i] = min(merged[i], key=_rule_complexity)
            else:
                return None

    if all(r.is_default for r in best):
        return None
    if not _vector_matches_examples(list(best), examples):
        return None

    result = 0
    for i in range(N_BITS):
        result |= _eval_rule_on_query(query, best[i]) << i
    answer = f"{result:08b}"
    lines = [
        "Per-bit analysis (stride-consistent runs, extrapolation, perfect-match):",
    ]
    if left_winner_name:
        lines.append(f"  Left section: {left_winner_name} ({left_winner_count} bits)")
    if right_winner_name:
        lines.append(f"  Right section: {right_winner_name} ({right_winner_count} bits)")
    for bit_idx in range(N_BITS):
        lines.append(
            f"  Output bit {bit_idx} = {best[bit_idx].expr} ({best[bit_idx].family})"
        )
    lines.append(f"Result: {answer}")
    return answer, "\n".join(lines)


def _solve_per_bit(
    examples: list[tuple[int, int]], query: int
) -> tuple[str, str] | None:
    """Stride heuristic first; validated search if stride fails; legacy last."""
    stride = _solve_per_bit_stride(examples, query)
    if stride is not None:
        return stride
    search = _solve_per_bit_search(examples, query)
    if search is not None:
        return search
    return _solve_per_bit_legacy(examples, query)


def solve_bit_manipulation(prompt: str) -> tuple[str, str]:
    examples, query_input = _parse_bit_examples(prompt)
    if not examples:
        return "", "Could not parse bit manipulation examples"

    single_ops = _single_ops()

    for name, func in single_ops:
        if _test_op(func, examples):
            result = func(query_input)
            answer = f"{result:08b}"
            reasoning = f"Found single operation: {name}\n"
            reasoning += f"Apply to query: {name}({query_input:08b}) = {answer}"
            return answer, reasoning

    two_op = _try_two_op_composition(examples, query_input, single_ops)
    if two_op:
        return two_op

    per_bit_result = _solve_per_bit(examples, query_input)
    if per_bit_result:
        return per_bit_result

    return "", f"Could not find matching operation for {len(examples)} examples"
