"""Roman numeral solver — decimal integer to Roman numeral string."""

import re


ROMAN_MAP = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def int_to_roman(n: int) -> str:
    result = []
    for value, numeral in ROMAN_MAP:
        while n >= value:
            result.append(numeral)
            n -= value
    return "".join(result)


def solve_roman_numeral(prompt: str) -> tuple[str, str]:
    m = re.search(r"write the number (\d+)", prompt, re.IGNORECASE)
    if not m:
        m = re.search(r"convert.*?(\d+)\s*(?:in|to)", prompt, re.IGNORECASE)
    if not m:
        nums = re.findall(r"\b(\d+)\b", prompt)
        if nums:
            m_val = int(nums[-1])
        else:
            return "", "Could not parse number from prompt"
    else:
        m_val = int(m.group(1))

    query_num = m_val
    answer = int_to_roman(query_num)

    examples = re.findall(r"(\d+)\s*->\s*([MDCLXVI]+)", prompt)
    reasoning_parts = []
    if examples:
        reasoning_parts.append("I can verify the mapping with the given examples:")
        for dec_str, rom_str in examples:
            computed = int_to_roman(int(dec_str))
            reasoning_parts.append(f"  {dec_str} → {computed} (expected {rom_str}, {'match' if computed == rom_str else 'MISMATCH'})")

    reasoning_parts.append(f"\nNow converting {query_num} to Roman numerals:")
    remainder = query_num
    steps = []
    for value, numeral in ROMAN_MAP:
        while remainder >= value:
            steps.append(f"{remainder} >= {value} → {numeral}, remainder {remainder - value}")
            remainder -= value
    for s in steps:
        reasoning_parts.append(f"  {s}")
    reasoning_parts.append(f"\nResult: {answer}")

    return answer, "\n".join(reasoning_parts)
