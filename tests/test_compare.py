"""Checks for scripts/alignment.py and scripts/compare_pronunciation.py (no model needed).

Run:  python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from alignment import align  # noqa: E402
from compare_pronunciation import compare, normalize_text, target_words  # noqa: E402
from phoneme_map import build_reference  # noqa: E402


def ref_of(*words):
    return build_reference([{"text": t, "phones": p.split()} for t, p in words])[0]


def ops(ref, hyp):
    return [(op, rule) for op, _, _, rule in align(ref, hyp.split())]


class Alignment(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(ops(ref_of(("CAT", "K AE1 T")), "K AE T"), [("match", None)] * 3)

    def test_substitution_deletion_insertion(self):
        ref = ref_of(("THINK", "TH IH1 NG K"))
        self.assertEqual(ops(ref, "S IH NG K")[0], ("substitution", None))       # think -> sink
        self.assertIn(("deletion", None), ops(ref, "TH IH NG"))                  # final K missing
        self.assertIn(("insertion", None), ops(ref, "TH IH NG K AH"))            # extra vowel at the end

    def test_accepted_difference_is_tagged(self):
        self.assertEqual(ops(ref_of(("MATTER", "M AE1 T ER0")), "M AE DX ER")[2], ("accepted", "R3"))

    def test_rules_can_be_switched_off(self):
        ref = ref_of(("MATTER", "M AE1 T ER0"))
        self.assertEqual([op for op, *_ in align(ref, "M AE DX ER".split(), use_rules=False)][2], "substitution")

    def test_no_vowel_consonant_substitution(self):
        # "WILL" heard as "V EH": W -> V and IH -> EH, L deleted; never IH -> V or L -> EH
        result = align(ref_of(("WILL", "W IH1 L")), ["V", "EH"])
        self.assertEqual([op for op, *_ in result], ["substitution", "substitution", "deletion"])


class Compare(unittest.TestCase):
    LEXICON = {
        "NOT": ["N AH0 T", "N AA0 T"],
        "HERE": ["HH IH1 R"],
        "THINK": ["TH IH1 NG K"],
    }

    def test_text_normalisation(self):
        self.assertEqual(normalize_text("Not here, think!"), ["NOT", "HERE", "THINK"])

    def test_unknown_word(self):
        with self.assertRaises(KeyError):
            target_words("NOT THERE", self.LEXICON)

    def test_word_statuses(self):
        words, variants = target_words("NOT HERE THINK", self.LEXICON)
        # model: "n ɑː t h ɪɹ s ɪ ŋ k" -> NOT via variant (R8), HERE correct, THINK with TH -> S
        result = compare("n ɑː t h ɪɹ s ɪ ŋ k", words, variants)
        status = {w["word"]: w["status"] for w in result["words"]}
        self.assertEqual(status, {"NOT": "kabul edilebilir fark", "HERE": "doğru", "THINK": "olası hata"})
        error = result["words"][2]["possible_errors"][0]
        self.assertEqual((error["type"], error["expected"], error["heard"], error["heard_ipa"]),
                         ("değiştirme", "TH", "S", "s"))
        self.assertEqual(result["summary"]["possible_errors"], 1)
        self.assertEqual(result["summary"]["accepted_differences"], 1)

    def test_missing_word(self):
        words, variants = target_words("NOT HERE", self.LEXICON)
        result = compare("n ɑː t", words, variants)
        here = result["words"][1]
        self.assertEqual([(d["type"], d["expected"]) for d in here["possible_errors"]], [("silme", "HH"), ("silme", "IH")])
        self.assertEqual([d["rule"] for d in here["accepted"]], ["R5"])  # R after a vowel is optional


if __name__ == "__main__":
    unittest.main()
