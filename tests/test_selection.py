"""Greedy phoneme-coverage sentence selection (voxrecte/selection.py)."""

import unittest

from voxrecte.selection import Sentence, rarity_weights, select_greedy, tally


def sentence(text, *transcriptions):
    """A Sentence with the given stress-free transcriptions."""
    return Sentence(text, transcriptions=list(transcriptions))


class SentencePhonemes(unittest.TestCase):
    """Phoneme counting per sentence."""

    def test_phoneme_counted_only_if_in_every_transcription(self):
        s = sentence("FOR ME", "F AO R | M IY", "F AO | M IY")  # rhotic and non-rhotic readings
        self.assertEqual(s.phonemes, {"F": 1, "AO": 1, "M": 1, "IY": 1})
        self.assertEqual(s.num_words, 2)

    def test_tally(self):
        tokens, containing = tally([sentence("A", "S IY S"), sentence("B", "S AH")])
        self.assertEqual(tokens["S"], 3)
        self.assertEqual(containing["S"], 2)


class GreedySelection(unittest.TestCase):
    """Greedy selection order and determinism."""

    def setUp(self):
        """Three small candidate sentences and their rarity weights."""
        self.candidates = [
            sentence("SEE", "S IY"),
            sentence("SEA SHE", "S IY | SH IY"),
            sentence("ZOO", "Z UW"),
        ]
        self.weights = rarity_weights(self.candidates)

    def test_most_new_phonemes_first(self):
        picked = select_greedy(self.candidates, 2, 1, self.weights)
        self.assertEqual([p.sentence.text for p in picked], ["SEA SHE", "ZOO"])
        self.assertEqual(picked[1].new_phonemes, ("UW", "Z"))

    def test_continues_from_already_selected(self):
        picked = select_greedy(self.candidates, 1, 1, self.weights, already_selected=[self.candidates[1]])
        self.assertEqual(picked[0].sentence.text, "ZOO")

    def test_deterministic(self):
        first = select_greedy(self.candidates, 3, 2, self.weights)
        second = select_greedy(list(reversed(self.candidates)), 3, 2, self.weights)
        self.assertEqual([p.sentence.text for p in first], [p.sentence.text for p in second])


if __name__ == "__main__":
    unittest.main()
