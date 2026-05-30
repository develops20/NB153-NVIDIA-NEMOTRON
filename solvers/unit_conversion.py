"""Unit conversion solver — detect hidden linear scale factor."""

import re


def solve_unit_conversion(prompt: str) -> tuple[str, str]:
    pairs = re.findall(
        r"([\d.]+)\s*(?:m|kg|L|units?|cm)?\s*becomes?\s*([\d.]+)", prompt
    )
    if not pairs:
        return "", "Could not parse conversion pairs"

    in_vals = [float(x) for x, _ in pairs]
    out_vals = [float(y) for _, y in pairs]

    ratios = []
    for inv, outv in zip(in_vals, out_vals):
        if inv > 0:
            ratios.append(outv / inv)
    if not ratios:
        return "", "All input values are zero"

    ratio_avg = sum(ratios) / len(ratios)

    query_match = re.search(
        r"convert.*?([\d.]+)\s*(?:m|kg|L|units?|cm)?", prompt
    )
    if not query_match:
        return "", "Could not parse query value"

    q_val = float(query_match.group(1))
    result = q_val * ratio_avg
    answer = f"{result:.2f}"

    reasoning_lines = ["Computing conversion ratio from each example pair:"]
    for i, (inv, outv, r) in enumerate(zip(in_vals, out_vals, ratios)):
        reasoning_lines.append(
            f"  Example {i+1}: {inv} → {outv}, ratio = {outv}/{inv} = {r:.6f}"
        )
    reasoning_lines.append(f"\nAverage ratio = {ratio_avg:.6f}")
    reasoning_lines.append(
        f"\nFor query value {q_val}:"
        f"\n  {q_val} × {ratio_avg:.6f} = {result:.2f}"
    )

    return answer, "\n".join(reasoning_lines)
