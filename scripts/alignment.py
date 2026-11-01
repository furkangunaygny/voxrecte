"""Align a reference phoneme sequence with the model's phoneme sequence.

Weighted Levenshtein alignment over common phonemes (see phoneme_map.py). Differences that
match an acceptable-variant rule get a small cost and are tagged with the rule id, so they
are aligned like matches but can still be reported. A vowel is never substituted for a
consonant or the other way round (that costs more than a deletion plus an insertion).
"""

from phoneme_map import UNKNOWN, VOWELS, deletion_rule, insertion_rule, substitution_rule

ACCEPTED_COST = 0.1
CROSS_CLASS_COST = 2.5


def align(ref, hyp, use_rules=True):
    """ref: [(phoneme, stress)] or [(phoneme, stress, alternatives)] (see phoneme_map.build_reference),
    hyp: [phoneme]. Returns a list of operations:
    (op, ref_index, hyp_index, rule) with op in match/accepted/substitution/deletion/insertion;
    ref_index or hyp_index is None for insertions/deletions."""
    n, m = len(ref), len(hyp)

    def sub(i, j):
        r, s, *alts = ref[i]
        if r == hyp[j]:
            return 0.0, "match", None
        rule = substitution_rule(r, s, hyp[j], alts[0] if alts else ()) if use_rules else None
        if rule:
            return ACCEPTED_COST, "accepted", rule
        cross = hyp[j] != UNKNOWN and (r in VOWELS) != (hyp[j] in VOWELS)
        return (CROSS_CLASS_COST if cross else 1.0), "substitution", None

    def dele(i):
        prev_ref = ref[i - 1][0] if i > 0 else None
        next_ref = ref[i + 1][0] if i + 1 < n else None
        rule = deletion_rule(ref[i][0], ref[i][1], prev_ref, next_ref) if use_rules else None
        return (ACCEPTED_COST, "accepted", rule) if rule else (1.0, "deletion", None)

    def ins(j):
        rule = insertion_rule(hyp[j], hyp[j - 1] if j > 0 else None) if use_rules else None
        return (ACCEPTED_COST, "accepted", rule) if rule else (1.0, "insertion", None)

    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        c, op, rule = dele(i - 1)
        cost[i][0], back[i][0] = cost[i - 1][0] + c, ("del", op, rule)
    for j in range(1, m + 1):
        c, op, rule = ins(j - 1)
        cost[0][j], back[0][j] = cost[0][j - 1] + c, ("ins", op, rule)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cs, ops, rs = sub(i - 1, j - 1)
            cd, opd, rd = dele(i - 1)
            ci, opi, ri = ins(j - 1)
            cost[i][j], back[i][j] = min(
                (cost[i - 1][j - 1] + cs, ("sub", ops, rs)),
                (cost[i - 1][j] + cd, ("del", opd, rd)),
                (cost[i][j - 1] + ci, ("ins", opi, ri)),
                key=lambda t: t[0],
            )

    ops, i, j = [], n, m
    while i > 0 or j > 0:
        move, op, rule = back[i][j]
        if move == "sub":
            i, j = i - 1, j - 1
            ops.append((op, i, j, rule))
        elif move == "del":
            i -= 1
            ops.append((op, i, None, rule))
        else:
            j -= 1
            ops.append((op, None, j, rule))
    return ops[::-1]
