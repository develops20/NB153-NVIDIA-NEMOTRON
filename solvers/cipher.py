"""Cipher solver — character-level substitution cipher decoder.

Each puzzle uses a single-alphabet substitution cipher on letters.
Build char→char mapping from all example plaintext↔ciphertext pairs,
then apply to query. Use bijection + wordlist to infer unmapped chars.
"""

import re

WORDLIST = {
    "a", "above", "alice", "ancient", "around", "beyond", "bird", "book",
    "bright", "castle", "cat", "cave", "chases", "clever", "colorful",
    "creates", "crystal", "curious", "dark", "discovers", "door",
    "dragon", "draws", "dreams", "explores", "follows", "forest",
    "found", "garden", "golden", "hatter", "hidden", "imagines", "in",
    "inside", "island", "key", "king", "knight", "library", "magical",
    "map", "message", "mirror", "mountain", "mouse", "mysterious",
    "near", "ocean", "palace", "potion", "princess", "puzzle", "queen",
    "rabbit", "reads", "school", "secret", "sees", "silver", "story",
    "strange", "student", "studies", "teacher", "the", "through",
    "tower", "treasure", "turtle", "under", "valley", "village",
    "watches", "wise", "wizard", "wonderland", "writes",
}


def solve_cipher(prompt: str) -> tuple[str, str]:
    lines = prompt.strip().split("\n")

    char_map: dict[str, str] = {}
    example_pairs: list[tuple[list[str], list[str]]] = []

    for line in lines:
        line = line.strip()
        if " -> " not in line:
            continue
        if "input" in line.lower() and "output" in line.lower():
            continue
        parts = line.split(" -> ", 1)
        if len(parts) != 2:
            continue
        cipher_words = parts[0].strip().split()
        plain_words = parts[1].strip().split()
        if len(cipher_words) != len(plain_words):
            continue

        example_pairs.append((cipher_words, plain_words))
        for cw, pw in zip(cipher_words, plain_words):
            if len(cw) != len(pw):
                continue
            for cc, pc in zip(cw, pw):
                if cc.isalpha() and pc.isalpha():
                    char_map[cc.lower()] = pc.lower()

    all_plain_words = set()
    for _, pwords in example_pairs:
        for w in pwords:
            all_plain_words.add(w.lower())

    query_match = re.search(
        r"(?:decrypt|translate|decode|convert).*?:\s*(.+)",
        prompt, re.IGNORECASE,
    )
    if not query_match:
        for line in reversed(lines):
            line = line.strip()
            if line and " -> " not in line:
                lower = line.lower()
                if "decrypt" in lower or "translate" in lower:
                    query_match = re.search(r":\s*(.+)", line)
                    break

    if not query_match:
        return "", "Could not parse cipher query"

    query_text = query_match.group(1).strip()
    query_words = query_text.split()

    all_letters = set("abcdefghijklmnopqrstuvwxyz")
    mapped_from = set(char_map.keys())
    mapped_to = set(char_map.values())
    unmapped_from = sorted(all_letters - mapped_from)
    unmapped_to = sorted(all_letters - mapped_to)

    query_unmapped = set()
    for w in query_words:
        for c in w:
            if c.isalpha() and c.lower() not in char_map:
                query_unmapped.add(c.lower())

    if query_unmapped and len(unmapped_from) == len(unmapped_to):
        best_map = _infer_by_wordlist(
            char_map, unmapped_from, unmapped_to,
            query_words, all_plain_words,
        )
        if best_map:
            char_map.update(best_map)

    decoded_words = []
    for word in query_words:
        dec = "".join(
            char_map.get(c.lower(), c) if c.isalpha() else c for c in word
        )
        decoded_words.append(dec)
    answer = " ".join(decoded_words)

    reasoning_lines = ["Building character substitution map from examples:"]
    map_items = sorted(char_map.items())
    reasoning_lines.append("  " + ", ".join(f"{k}→{v}" for k, v in map_items))
    reasoning_lines.append(f"\nDecrypting: {query_text}")
    for cw, dw in zip(query_words, decoded_words):
        reasoning_lines.append(f"  {cw} → {dw}")
    reasoning_lines.append(f"\nResult: {answer}")

    return answer, "\n".join(reasoning_lines)


def _infer_by_wordlist(
    base_map: dict[str, str],
    unmapped_from: list[str],
    unmapped_to: list[str],
    query_words: list[str],
    known_words: set[str],
) -> dict[str, str] | None:
    """Infer unmapped cipher→plain assignments using wordlist constraints.

    For each query word with unknown letters, pattern-match against the
    wordlist to narrow candidate plaintext letters.  Then backtrack over
    only the relevant (query-appearing) cipher letters with pruned
    candidate sets — fast even when many global letters are unmapped.
    """
    candidate_wordlist = known_words | WORDLIST
    unmapped_to_set = set(unmapped_to)

    relevant = []
    for c in unmapped_from:
        if any(c in w.lower() for w in query_words):
            relevant.append(c)
    if not relevant:
        return None

    relevant_set = set(relevant)

    # Narrow candidates per cipher letter via wordlist pattern matching
    candidates: dict[str, set[str]] = {c: set() for c in relevant}

    for w in query_words:
        wl = w.lower()
        if not any(ch in relevant_set for ch in wl):
            continue
        for dictword in candidate_wordlist:
            if len(dictword) != len(wl):
                continue
            local: dict[str, str] = {}
            ok = True
            for cc, dc in zip(wl, dictword):
                if not cc.isalpha():
                    if dc != cc:
                        ok = False
                        break
                    continue
                if cc in base_map:
                    if dc != base_map[cc]:
                        ok = False
                        break
                elif cc in relevant_set:
                    if dc not in unmapped_to_set:
                        ok = False
                        break
                    if cc in local and local[cc] != dc:
                        ok = False
                        break
                    local[cc] = dc
            if ok and local and len(set(local.values())) == len(local):
                for cc, dc in local.items():
                    candidates[cc].add(dc)

    # Build search list; fall back to all unmapped_to if no matches found
    search = [
        (c, sorted(candidates[c]) if candidates[c] else list(unmapped_to))
        for c in relevant
    ]

    best_score = -1
    best_map: dict[str, str] | None = None
    used: set[str] = set()
    assignment: dict[str, str] = {}

    def backtrack(idx: int) -> None:
        nonlocal best_score, best_map
        if idx == len(search):
            trial = dict(base_map)
            trial.update(assignment)
            score = sum(
                1
                for w in query_words
                if "".join(trial.get(c, c) for c in w.lower()) in candidate_wordlist
            )
            if score > best_score:
                best_score = score
                best_map = dict(assignment)
            return
        cipher_c, options = search[idx]
        for p in options:
            if p not in used:
                assignment[cipher_c] = p
                used.add(p)
                backtrack(idx + 1)
                used.discard(p)
                del assignment[cipher_c]

    backtrack(0)
    return best_map
