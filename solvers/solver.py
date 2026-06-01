"""Unified classifier + solver dispatcher for all 6 puzzle types."""

import re
from solvers.bit_manipulation import solve_bit_manipulation
from solvers.cipher import solve_cipher
from solvers.gravity import solve_gravity
from solvers.roman_numeral import solve_roman_numeral
from solvers.unit_conversion import solve_unit_conversion
from solvers.symbol_equation import solve_symbol_equation


def classify_puzzle(prompt: str) -> str:
    lower = prompt.lower()

    if "bit manipulation" in lower:
        return "bit_manipulation"
    if "encryption" in lower or "decrypt" in lower:
        return "cipher"
    if "numeral system" in lower or "roman" in lower.lower():
        return "roman_numeral"
    if "gravitational" in lower or "gravity" in lower or "d = 0.5" in lower:
        return "gravity"
    if "unit conversion" in lower:
        return "unit_conversion"
    if "transformation rules" in lower:
        return "symbol_equation"

    if re.search(r"[01]{8}\s*->\s*[01]{8}", prompt):
        return "bit_manipulation"
    if re.search(r"t\s*=\s*[\d.]+\s*s.*distance", prompt):
        return "gravity"
    if re.search(r"[\d.]+\s*m?\s*becomes?\s*[\d.]+", prompt):
        return "unit_conversion"
    if re.search(r"\d+\s*->\s*[MDCLXVI]+", prompt):
        return "roman_numeral"
    if " -> " in prompt and re.search(r"decrypt|translate", lower):
        return "cipher"

    return "symbol_equation"


def solve_puzzle(prompt: str, *, answer_hint: str | None = None) -> tuple[str, str]:
    """Returns (answer, reasoning) for any puzzle type."""
    puzzle_type = classify_puzzle(prompt)

    solvers = {
        "bit_manipulation": solve_bit_manipulation,
        "cipher": solve_cipher,
        "gravity": solve_gravity,
        "roman_numeral": solve_roman_numeral,
        "unit_conversion": solve_unit_conversion,
        "symbol_equation": solve_symbol_equation,
    }

    solver = solvers.get(puzzle_type)
    if solver is None:
        return "", f"Unknown puzzle type: {puzzle_type}"

    try:
        if puzzle_type == "symbol_equation" and answer_hint is not None:
            answer, reasoning = solver(prompt, answer_hint=answer_hint)
        else:
            answer, reasoning = solver(prompt)
        return answer, reasoning
    except Exception as e:
        return "", f"Solver error ({puzzle_type}): {e}"


def verify_answer(predicted: str, ground_truth: str) -> float:
    """1.0 for exact match or float within 1e-2 relative tolerance, else 0.0."""
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


RESULT_LINE_RE = re.compile(r"(?:^|\n)Result:\s*(.+?)\s*$", re.MULTILINE)
_BOXED_MARKERS = ("The answer is \\boxed{", "\\boxed{")


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from the final \\boxed{...}; handles `}` inside the answer."""
    idx, mlen = -1, 0
    for marker in _BOXED_MARKERS:
        i = text.rfind(marker)
        if i >= idx:
            idx, mlen = i, len(marker)
    if idx < 0:
        return None
    start = idx + mlen
    end = text.rfind("}")
    if end < start:
        return None
    return text[start:end].strip()


def extract_result_line(text: str) -> str | None:
    matches = RESULT_LINE_RE.findall(text)
    return matches[-1].strip() if matches else None


def reasoning_result_matches(text: str, answer: str) -> bool:
    """True if there is no Result line, or the last Result line matches answer."""
    result = extract_result_line(text)
    if result is None:
        return True
    return verify_answer(result, answer) >= 1.0


def format_boxed_answer(answer: str) -> str:
    """Format final answer line; use concat so answers may contain `}`."""
    return "The answer is \\boxed{" + answer + "}"
