"""Acceptable pronunciation differences (voxrecte/rules.py), checked on example words.

Reference-side examples use ARPAbet as written in speechocean762.
"""

import unittest

from voxrecte.rules import (
    ReferencePhone, build_reference, deletion_rule, insertion_rule, reference_variants, substitution_rule,
)


def phone(phoneme, stress=None, alternatives=()):
    """A reference phoneme with optional stress and R8 alternatives."""
    return ReferencePhone(phoneme, stress, frozenset(alternatives))


class ReferenceBuilding(unittest.TestCase):
    """Reference sequence built from word entries."""

    def test_one_syllable_words_have_unknown_stress(self):
        reference = build_reference([{"text": "NO", "phones": ["N", "OW0"]},
                                     {"text": "VERY", "phones": ["V", "EH1", "R", "IY0"]}])
        self.assertEqual([p.stress for p in reference], [None, None, None, 1, None, 0])
        self.assertEqual([p.word_index for p in reference], [0, 0, 1, 1, 1, 1])
        self.assertIsNone(substitution_rule(reference[1], "IH"))  # "NO" -> "nih" is not reduction

    def test_reference_variants(self):  # R8: speechocean762 writes NOT as N AH T and N AA T
        words = [{"text": "NOT", "phones": ["N", "AH0", "T"]}, {"text": "NOT", "phones": ["N", "AA1", "T"]}]
        reference = build_reference(words[:1], reference_variants(words))
        self.assertEqual(reference[1], ReferencePhone("AH", None, frozenset({"AA"}), 0))
        self.assertEqual(substitution_rule(reference[1], "AA"), "R8")
        self.assertIsNone(substitution_rule(phone("AH", 1), "AA"))  # "cup" -> "cop" without a variant


class AcceptableDifferences(unittest.TestCase):
    """Each acceptable-difference rule, and real errors it must not accept."""

    def test_flap_for_t_and_d(self):  # R3: "matter", "ladder"
        self.assertEqual(substitution_rule(phone("T"), "DX"), "R3")
        self.assertEqual(substitution_rule(phone("D"), "DX"), "R3")

    def test_glottal_stop_for_t(self):  # R4: "button"
        self.assertEqual(substitution_rule(phone("T"), "Q"), "R4")
        self.assertIsNone(substitution_rule(phone("D"), "Q"))

    def test_unstressed_vowel_reduction(self):  # R2: "classes" K L AE1 S AH0 Z vs ... ᵻ
        self.assertEqual(substitution_rule(phone("AH", 0), "IX"), "R2")
        self.assertEqual(substitution_rule(phone("IH", 0), "AH"), "R2")
        self.assertIsNone(substitution_rule(phone("AE", 1), "AH"))  # stressed vowel: possible error

    def test_rhotic_and_non_rhotic(self):
        # speechocean762 "FOR" = F AO (British) while the model writes f ɔːɹ -> extra R is accepted
        self.assertEqual(insertion_rule("R", "AO"), "R5")
        # reference F AO R, speaker drops the R
        self.assertEqual(deletion_rule(phone("R"), "AO", None), "R5")
        # "MOTHER" M AH1 DH AH0 vs model m ʌ ð ɚ
        self.assertEqual(substitution_rule(phone("AH", 0), "ER"), "R6")
        # stressed "TURN" (T ER N) read as "ton" is not an accent variant
        self.assertIsNone(substitution_rule(phone("ER"), "AH"))
        # R before a vowel is not optional ("RED")
        self.assertIsNone(deletion_rule(phone("R"), None, "EH"))
        self.assertIsNone(insertion_rule("R", "S"))

    def test_syllabic_consonant(self):  # R7: "little" L IH1 T AH0 L, model may give ... l
        self.assertEqual(deletion_rule(phone("AH", 0), "T", "L"), "R7")
        self.assertIsNone(deletion_rule(phone("AH", 1), "T", "L"))

    def test_real_substitutions_are_not_accepted(self):
        self.assertIsNone(substitution_rule(phone("TH"), "S"))     # think -> sink
        self.assertIsNone(substitution_rule(phone("R"), "L"))
        self.assertIsNone(substitution_rule(phone("V"), "W"))
        self.assertIsNone(substitution_rule(phone("IY", 1), "IH"))  # sheep -> ship


if __name__ == "__main__":
    unittest.main()
