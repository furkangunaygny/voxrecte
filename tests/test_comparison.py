"""Recording vs. target sentence comparison (voxrecte/comparison.py, voxrecte/lexicon.py).

No model is needed: the model output is given as an IPA string.
"""

import unittest

from voxrecte.comparison import (
    STATUS_ACCEPTED, STATUS_CORRECT, STATUS_POSSIBLE_ERROR, compare, describe_difference, format_result,
)
from voxrecte.lexicon import UnknownWordsError, target_words
from voxrecte.text import normalize_sentence, words_of

LEXICON = {
    "NOT": ["N AH0 T", "N AA0 T"],
    "HERE": ["HH IH1 R"],
    "THINK": ["TH IH1 NG K"],
}


class Text(unittest.TestCase):
    """Sentence normalisation."""

    def test_normalisation(self):
        self.assertEqual(words_of("Not here, think!"), ["NOT", "HERE", "THINK"])
        self.assertEqual(normalize_sentence("It's  here."), "IT'S HERE")


class Lexicon(unittest.TestCase):
    """Target pronunciation lookup."""

    def test_target_uses_most_frequent_pronunciation(self):
        words, variants = target_words("not", LEXICON)
        self.assertEqual(words, [{"text": "NOT", "phones": ["N", "AH0", "T"]}])
        self.assertEqual(variants["NOT"], {("N", "AH", "T"), ("N", "AA", "T")})

    def test_unknown_word(self):
        with self.assertRaises(UnknownWordsError) as raised:
            target_words("NOT THERE", LEXICON)
        self.assertEqual(raised.exception.words, ["THERE"])


class Compare(unittest.TestCase):
    """Per-word comparison results and the text output."""

    def test_word_statuses(self):
        words, variants = target_words("NOT HERE THINK", LEXICON)
        # model: NOT via variant (R8), HERE correct, THINK with TH -> S
        result = compare("n ɑː t h ɪɹ s ɪ ŋ k", words, variants)
        status = {w["word"]: w["status"] for w in result["words"]}
        self.assertEqual(status, {"NOT": STATUS_ACCEPTED, "HERE": STATUS_CORRECT, "THINK": STATUS_POSSIBLE_ERROR})
        error = result["words"][2]["possible_errors"][0]
        self.assertEqual((error["type"], error["expected"], error["heard"], error["heard_ipa"]),
                         ("değiştirme", "TH", "S", "s"))
        self.assertEqual(describe_difference(error), "TH yerine S (model: s)")
        self.assertEqual(result["summary"]["possible_errors"], 1)
        self.assertEqual(result["summary"]["accepted_differences"], 1)

    def test_missing_word(self):
        words, variants = target_words("NOT HERE", LEXICON)
        here = compare("n ɑː t", words, variants)["words"][1]
        self.assertEqual([(d["type"], d["expected"]) for d in here["possible_errors"]],
                         [("silme", "HH"), ("silme", "IH")])
        self.assertEqual([d["rule"] for d in here["accepted"]], ["R5"])  # R after a vowel is optional

    def test_format_result_mentions_every_word(self):
        words, variants = target_words("NOT HERE THINK", LEXICON)
        text = format_result("not here think", compare("n ɑː t h ɪɹ s ɪ ŋ k", words, variants))
        for word in ("NOT", "HERE", "THINK", "R8"):
            self.assertIn(word, text)


if __name__ == "__main__":
    unittest.main()
