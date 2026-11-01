"""Alignment of the reference phoneme sequence with the model's phoneme sequence.

Weighted Levenshtein alignment over common phonemes. Differences that match an acceptable
pronunciation rule (rules.py) get a small cost and are tagged with the rule id, so they align
like matches but can still be reported. A vowel is never substituted for a consonant or the
other way round: that costs more than a deletion plus an insertion.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from voxrecte.phonemes import UNKNOWN, VOWELS
from voxrecte.rules import ReferencePhone, deletion_rule, insertion_rule, substitution_rule

MATCH, ACCEPTED, SUBSTITUTION, DELETION, INSERTION = "match", "accepted", "substitution", "deletion", "insertion"
ERROR_KINDS = (SUBSTITUTION, DELETION, INSERTION)

ACCEPTED_COST = 0.1
SUBSTITUTION_COST = 1.0
GAP_COST = 1.0  # deletion or insertion
CROSS_CLASS_COST = 2.5  # vowel <-> consonant substitution; > 2 * GAP_COST, so never chosen


@dataclass(frozen=True)
class AlignedPair:
    """One step of the alignment.

    kind:        MATCH, ACCEPTED, SUBSTITUTION, DELETION or INSERTION
    ref_index:   index in the reference, None for an insertion
    hyp_index:   index in the model sequence, None for a deletion
    rule:        rule id for ACCEPTED steps
    """
    kind: str
    ref_index: int | None
    hyp_index: int | None
    rule: str | None = None


class _Costs:
    """Cost and label of each edit operation for one reference / hypothesis pair."""

    def __init__(self, ref: Sequence[ReferencePhone], hyp: Sequence[str], use_rules: bool):
        """Keep the sequences being aligned and whether rules apply."""
        self.ref, self.hyp, self.use_rules = ref, hyp, use_rules

    def substitute(self, i: int, j: int) -> tuple[float, str, str | None]:
        """Cost, kind and rule for aligning reference phoneme i with model phoneme j."""
        ref, heard = self.ref[i], self.hyp[j]
        if heard == ref.phoneme:
            return 0.0, MATCH, None
        rule = substitution_rule(ref, heard) if self.use_rules else None
        if rule:
            return ACCEPTED_COST, ACCEPTED, rule
        cross_class = heard != UNKNOWN and (ref.phoneme in VOWELS) != (heard in VOWELS)
        return (CROSS_CLASS_COST if cross_class else SUBSTITUTION_COST), SUBSTITUTION, None

    def delete(self, i: int) -> tuple[float, str, str | None]:
        """Cost, kind and rule for a reference phoneme without a model counterpart."""
        previous = self.ref[i - 1].phoneme if i > 0 else None
        following = self.ref[i + 1].phoneme if i + 1 < len(self.ref) else None
        rule = deletion_rule(self.ref[i], previous, following) if self.use_rules else None
        return (ACCEPTED_COST, ACCEPTED, rule) if rule else (GAP_COST, DELETION, None)

    def insert(self, j: int) -> tuple[float, str, str | None]:
        """Cost, kind and rule for a model phoneme without a reference counterpart."""
        previous = self.hyp[j - 1] if j > 0 else None
        rule = insertion_rule(self.hyp[j], previous) if self.use_rules else None
        return (ACCEPTED_COST, ACCEPTED, rule) if rule else (GAP_COST, INSERTION, None)


def align(ref: Sequence[ReferencePhone], hyp: Sequence[str], use_rules: bool = True) -> list[AlignedPair]:
    """Minimum-cost alignment of reference phonemes and model phonemes, in sequence order.

    With use_rules=False every difference is an error (used to measure what the rules change)."""
    costs = _Costs(ref, hyp, use_rules)
    n, m = len(ref), len(hyp)
    total = [[0.0] * (m + 1) for _ in range(n + 1)]
    step = [[None] * (m + 1) for _ in range(n + 1)]  # (move, kind, rule) that reached each cell

    for i in range(1, n + 1):
        cost, kind, rule = costs.delete(i - 1)
        total[i][0], step[i][0] = total[i - 1][0] + cost, ("del", kind, rule)
    for j in range(1, m + 1):
        cost, kind, rule = costs.insert(j - 1)
        total[0][j], step[0][j] = total[0][j - 1] + cost, ("ins", kind, rule)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost, sub_kind, sub_rule = costs.substitute(i - 1, j - 1)
            del_cost, del_kind, del_rule = costs.delete(i - 1)
            ins_cost, ins_kind, ins_rule = costs.insert(j - 1)
            # On equal cost the first option wins: substitution, then deletion, then insertion.
            total[i][j], step[i][j] = min(
                (total[i - 1][j - 1] + sub_cost, ("sub", sub_kind, sub_rule)),
                (total[i - 1][j] + del_cost, ("del", del_kind, del_rule)),
                (total[i][j - 1] + ins_cost, ("ins", ins_kind, ins_rule)),
                key=lambda option: option[0],
            )
    return _backtrace(step, n, m)


def _backtrace(step, i: int, j: int) -> list[AlignedPair]:
    """Follow the stored moves back from the last cell to recover the alignment."""
    pairs = []
    while i > 0 or j > 0:
        move, kind, rule = step[i][j]
        if move == "sub":
            i, j = i - 1, j - 1
            pairs.append(AlignedPair(kind, i, j, rule))
        elif move == "del":
            i -= 1
            pairs.append(AlignedPair(kind, i, None, rule))
        else:
            j -= 1
            pairs.append(AlignedPair(kind, None, j, rule))
    return pairs[::-1]


def count_errors(pairs: Sequence[AlignedPair]) -> int:
    """Number of substitutions, deletions and insertions (accepted differences excluded)."""
    return sum(pair.kind in ERROR_KINDS for pair in pairs)
