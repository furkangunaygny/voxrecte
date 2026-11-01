"""Reference / model phoneme alignment (voxrecte/alignment.py)."""

import unittest

from voxrecte.alignment import (
    ACCEPTED, DELETION, INSERTION, MATCH, SUBSTITUTION, align, count_errors,
)
from voxrecte.rules import build_reference


def reference(word, phones):
    """Reference phonemes of one word."""
    return build_reference([{"text": word, "phones": phones.split()}])


def kinds(ref, hyp, use_rules=True):
    """(kind, rule) of each aligned pair."""
    return [(pair.kind, pair.rule) for pair in align(ref, hyp.split(), use_rules)]


class Alignment(unittest.TestCase):
    """Alignment kinds, indices and the vowel / consonant constraint."""

    def test_exact_match(self):
        self.assertEqual(kinds(reference("CAT", "K AE1 T"), "K AE T"), [(MATCH, None)] * 3)

    def test_substitution_deletion_insertion(self):
        ref = reference("THINK", "TH IH1 NG K")
        self.assertEqual(kinds(ref, "S IH NG K")[0], (SUBSTITUTION, None))  # think -> sink
        self.assertIn((DELETION, None), kinds(ref, "TH IH NG"))             # final K missing
        self.assertIn((INSERTION, None), kinds(ref, "TH IH NG K AH"))       # extra vowel at the end

    def test_indices(self):
        pairs = align(reference("THINK", "TH IH1 NG K"), ["TH", "IH", "NG"])
        self.assertEqual([(p.ref_index, p.hyp_index) for p in pairs], [(0, 0), (1, 1), (2, 2), (3, None)])

    def test_accepted_difference_is_tagged(self):
        self.assertEqual(kinds(reference("MATTER", "M AE1 T ER0"), "M AE DX ER")[2], (ACCEPTED, "R3"))

    def test_rules_can_be_switched_off(self):
        ref = reference("MATTER", "M AE1 T ER0")
        self.assertEqual(kinds(ref, "M AE DX ER", use_rules=False)[2], (SUBSTITUTION, None))

    def test_no_vowel_consonant_substitution(self):
        # "WILL" heard as "V EH": W -> V and IH -> EH, L deleted; never IH -> V or L -> EH
        pairs = align(reference("WILL", "W IH1 L"), ["V", "EH"])
        self.assertEqual([p.kind for p in pairs], [SUBSTITUTION, SUBSTITUTION, DELETION])

    def test_count_errors_ignores_accepted(self):
        pairs = align(reference("MATTER", "M AE1 T ER0"), ["M", "EH", "DX", "ER"])
        self.assertEqual(count_errors(pairs), 1)  # AE -> EH; T -> DX is accepted


if __name__ == "__main__":
    unittest.main()
