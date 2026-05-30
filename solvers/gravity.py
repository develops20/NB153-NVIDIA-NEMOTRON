"""Gravity solver — fit d = 0.5 * g * t^2 and predict distance."""

import re


def solve_gravity(prompt: str) -> tuple[str, str]:
    pairs = re.findall(
        r"t\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m", prompt
    )
    if not pairs:
        return "", "Could not parse (t, d) pairs from prompt"

    t_vals = [float(t) for t, _ in pairs]
    d_vals = [float(d) for _, d in pairs]

    g_estimates = []
    for t, d in zip(t_vals, d_vals):
        if t > 0:
            g_estimates.append(2 * d / (t * t))
    if not g_estimates:
        return "", "All t values are zero"

    g_avg = sum(g_estimates) / len(g_estimates)

    query_match = re.search(
        r"(?:determine|predict|calculate|find).*?t\s*=\s*([\d.]+)\s*s", prompt
    )
    if not query_match:
        query_match = re.search(r"t\s*=\s*([\d.]+)\s*s[^=]*$", prompt)
    if not query_match:
        return "", "Could not parse query time"

    t_query = float(query_match.group(1))
    d_predicted = 0.5 * g_avg * t_query * t_query
    answer = f"{d_predicted:.2f}"

    reasoning_lines = ["Computing gravitational constant g from each example pair:"]
    for i, (t, d, g) in enumerate(zip(t_vals, d_vals, g_estimates)):
        reasoning_lines.append(
            f"  Example {i+1}: t={t}, d={d} → g = 2×{d}/{t}² = {g:.4f}"
        )
    reasoning_lines.append(f"\nAverage g = {g_avg:.4f}")
    reasoning_lines.append(
        f"\nFor t = {t_query}s:"
        f"\n  d = 0.5 × {g_avg:.4f} × {t_query}² = {d_predicted:.2f}"
    )

    return answer, "\n".join(reasoning_lines)
