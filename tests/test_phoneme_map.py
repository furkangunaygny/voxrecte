"""Example-word checks for scripts/phoneme_map.py.

Run:  python -m unittest discover tests

Model-side examples use the eSpeak IPA tokens the model produces (several are taken from the
LibriSpeech outputs in outputs/phoneme_baseline.md); reference-side examples use ARPAbet as
written in speechocean762.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from phoneme_map import (  # noqa: E402
    UNKNOWN, build_reference, deletion_rule, insertion_rule, ipa_to_common, parse_arpabet,
    reference_variants, substitution_rule,
)

# word, model IPA, expected common phonemes
IPA_WORDS = [
    ("matter", "m æ ɾ ɚ", "M AE DX ER"),
    ("little", "l ɪ ɾ əl", "L IH DX AH L"),
    ("classes", "k l æ s ᵻ z", "K L AE S IX Z"),
    ("for", "f ɔːɹ", "F AO R"),
    ("car", "k ɑːɹ", "K AA R"),
    ("here", "h ɪɹ", "HH IH R"),
    ("fire", "f aɪɚ", "F AY ER"),
    ("measure", "m ɛ ʒ ɚ", "M EH ZH ER"),
    ("boy", "b ɔɪ", "B OY"),
    ("church", "tʃ ɜː tʃ", "CH ER CH"),
    ("judge", "dʒ ʌ dʒ", "JH AH JH"),
    ("think", "θ ɪ ŋ k", "TH IH NG K"),
    ("the", "ð ə", "DH AH"),
    ("go", "ɡ oʊ", "G OW"),
    ("house", "h aʊ s", "HH AW S"),
    ("button", "b ʌ ʔ n̩", "B AH Q AH N"),
    ("yes", "j ɛ s", "Y EH S"),
    ("book", "b ʊ k", "B UH K"),
    ("food", "f uː d", "F UW D"),
    ("cat", "k æ t", "K AE T"),
]


class IpaConversion(unittest.TestCase):
    def test_example_words(self):
        for word, ipa, expected in IPA_WORDS:
            with self.subTest(word=word):
                self.assertEqual(ipa_to_common(ipa), expected.split())

    def test_non_english_tokens_are_reduced_or_marked(self):
        self.assertEqual(ipa_to_common("tʃʰ"), ["CH"])   # aspiration removed
        self.assertEqual(ipa_to_common("dː"), ["D"])     # length removed
        self.assertEqual(ipa_to_common("a1"), ["AE"])    # tone digit removed
        self.assertEqual(ipa_to_common("ø"), [UNKNOWN])  # no English counterpart
        self.assertEqual(ipa_to_common("ʐ"), [UNKNOWN])

    def test_mandarin_symbols(self):
        self.assertEqual(ipa_to_common("th iː"), ["T", "IY"])      # aspirated t ("tea")
        self.assertEqual(ipa_to_common("x aʊ s"), ["HH", "AW", "S"])  # velar fricative for h ("house")
        self.assertEqual(ipa_to_common("ai5"), ["AY"])             # tone mark removed
        self.assertEqual(ipa_to_common("iouɜ"), ["Y", "OW"])      # tone-3 mark removed
        self.assertEqual(ipa_to_common("ts"), [UNKNOWN])           # no single English counterpart


class ArpabetParsing(unittest.TestCase):
    def test_stress_is_split_off(self):
        self.assertEqual(parse_arpabet("K AO0 L"), [("K", None), ("AO", 0), ("L", None)])
        self.assertEqual(parse_arpabet(["W", "IY1"]), [("W", None), ("IY", 1)])
        self.assertEqual(parse_arpabet("IH"), [("IH", None)])  # some references have no digit

    def test_one_syllable_words_have_unknown_stress(self):
        ref, _ = build_reference([{"text": "NO", "phones": ["N", "OW0"]},
                                  {"text": "VERY", "phones": ["V", "EH1", "R", "IY0"]}])
        self.assertEqual([s for _, s, _ in ref], [None, None, None, 1, None, 0])
        self.assertIsNone(substitution_rule("OW", ref[1][1], "IH"))  # "NO" -> "nih" is not reduction

    def test_unknown_symbol_raises(self):
        with self.assertRaises(ValueError):
            parse_arpabet("K AX L")


class AcceptableDifferences(unittest.TestCase):
    def test_flap_for_t_and_d(self):  # R3: "matter", "ladder"
        self.assertEqual(substitution_rule("T", None, "DX"), "R3")
        self.assertEqual(substitution_rule("D", None, "DX"), "R3")

    def test_glottal_stop_for_t(self):  # R4: "button"
        self.assertEqual(substitution_rule("T", None, "Q"), "R4")
        self.assertIsNone(substitution_rule("D", None, "Q"))

    def test_unstressed_vowel_reduction(self):  # R2: "classes" K L AE1 S AH0 Z vs ... ᵻ
        self.assertEqual(substitution_rule("AH", 0, "IX"), "R2")
        self.assertEqual(substitution_rule("IH", 0, "AH"), "R2")
        self.assertIsNone(substitution_rule("AE", 1, "AH"))  # stressed vowel: possible error

    def test_rhotic_and_non_rhotic(self):
        # speechocean762 "FOR" = F AO (British) while the model writes f ɔːɹ -> extra R is accepted
        self.assertEqual(insertion_rule("R", "AO"), "R5")
        # reference F AO R, speaker drops the R
        self.assertEqual(deletion_rule("R", None, "AO", None), "R5")
        # "MOTHER" M AH1 DH AH0 vs model m ʌ ð ɚ
        self.assertEqual(substitution_rule("AH", 0, "ER"), "R6")
        # stressed "TURN" (T ER N) read as "ton" is not an accent variant
        self.assertIsNone(substitution_rule("ER", None, "AH"))
        # R before a vowel is not optional ("RED")
        self.assertIsNone(deletion_rule("R", None, None, "EH"))
        self.assertIsNone(insertion_rule("R", "S"))

    def test_syllabic_consonant(self):  # R7: "little" L IH1 T AH0 L, model may give l ɪ ɾ l̩ / ... l
        self.assertEqual(deletion_rule("AH", 0, "T", "L"), "R7")
        self.assertIsNone(deletion_rule("AH", 1, "T", "L"))

    def test_reference_variants(self):  # R8: speechocean762 writes NOT as N AH T and N AA T
        words = [{"text": "NOT", "phones": ["N", "AH0", "T"]}, {"text": "NOT", "phones": ["N", "AA1", "T"]}]
        ref, word_index = build_reference(words[:1], reference_variants(words))
        self.assertEqual(ref[1], ("AH", None, frozenset({"AA"})))  # one-syllable word: stress unknown
        self.assertEqual(word_index, [0, 0, 0])
        self.assertEqual(substitution_rule("AH", 1, "AA", ref[1][2]), "R8")
        self.assertIsNone(substitution_rule("AH", 1, "AA"))  # "cup" -> "cop" without a variant

    def test_real_substitutions_are_not_accepted(self):
        self.assertIsNone(substitution_rule("TH", None, "S"))  # think -> sink
        self.assertIsNone(substitution_rule("R", None, "L"))
        self.assertIsNone(substitution_rule("V", None, "W"))
        self.assertIsNone(substitution_rule("IY", 1, "IH"))   # sheep -> ship


if __name__ == "__main__":
    unittest.main()
