"""Example-word checks for the IPA / ARPAbet conversion (voxrecte/phonemes.py).

Model-side examples use the eSpeak IPA tokens the model produces (several are taken from the
LibriSpeech outputs in outputs/phoneme_baseline.json).
"""

import unittest

from voxrecte.phonemes import (
    FROM_REDUCED, FROM_TABLE, NOT_MAPPED, UNKNOWN, ipa_to_common, mapping_source, parse_arpabet, strip_stress,
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
    """Model IPA tokens to common phonemes."""

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
        self.assertEqual(ipa_to_common("th iː"), ["T", "IY"])         # aspirated t ("tea")
        self.assertEqual(ipa_to_common("x aʊ s"), ["HH", "AW", "S"])  # velar fricative for h ("house")
        self.assertEqual(ipa_to_common("ai5"), ["AY"])                # tone mark removed
        self.assertEqual(ipa_to_common("iouɜ"), ["Y", "OW"])          # tone-3 mark removed
        self.assertEqual(ipa_to_common("ts"), [UNKNOWN])              # no single English counterpart

    def test_mapping_source(self):
        self.assertEqual(mapping_source("æ"), FROM_TABLE)
        self.assertEqual(mapping_source("ai5"), FROM_REDUCED)
        self.assertEqual(mapping_source("ø"), NOT_MAPPED)


class ArpabetParsing(unittest.TestCase):
    """Reference ARPAbet parsing."""

    def test_stress_is_split_off(self):
        self.assertEqual(parse_arpabet("K AO0 L"), [("K", None), ("AO", 0), ("L", None)])
        self.assertEqual(parse_arpabet(["W", "IY1"]), [("W", None), ("IY", 1)])
        self.assertEqual(parse_arpabet("IH"), [("IH", None)])  # some references have no digit

    def test_unknown_symbol_raises(self):
        with self.assertRaises(ValueError):
            parse_arpabet("K AX L")

    def test_strip_stress(self):
        self.assertEqual(strip_stress(["W", "IY0"]), ("W", "IY"))


if __name__ == "__main__":
    unittest.main()
