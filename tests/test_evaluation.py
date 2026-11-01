"""Expert label parsing and detection scores (voxrecte/evaluation.py)."""

import math
import unittest

from voxrecte.evaluation import (
    ACCENTED, CORRECT, WRONG, PhoneOutcome, detection_scores, evaluate_utterance, expert_word,
    parse_expert_annotation,
)

# The README example of scores-detail.json: BEAR scored by five experts.
BEAR_DETAIL = {"text": "BEAR", "ref-phones": "B EH0 R",
               "phones": ["B (EH0) (R)", "B {EH0} {R}", "B EH0 R", "B (EH0) (R)", "B EH0 [L] R"]}


def outcome(wrong_votes, system):
    """A phoneme outcome with the given number of 'incorrect' votes."""
    votes = tuple([WRONG] * wrong_votes + [CORRECT] * (5 - wrong_votes))
    return PhoneOutcome("u", 0, "W", "T", votes, system, None, None)


class ExpertLabels(unittest.TestCase):
    """Parsing of the experts' marks in scores-detail.json."""

    def test_annotation_marks(self):
        self.assertEqual(parse_expert_annotation("B (EH0) {R} [L]"), ([CORRECT, WRONG, ACCENTED], ["L"]))
        self.assertEqual(parse_expert_annotation("K AO0 L"), ([CORRECT] * 3, []))

    def test_votes_per_phoneme(self):
        word = expert_word(BEAR_DETAIL)
        self.assertEqual(word.votes[0], [CORRECT] * 5)                              # B
        self.assertEqual(word.votes[1], [WRONG, ACCENTED, CORRECT, WRONG, CORRECT])  # EH0, mean 1.0 as in scores.json
        self.assertEqual(word.insertions, [0, 0, 0, 0, 1])

    def test_annotation_must_match_reference(self):
        with self.assertRaises(ValueError):
            expert_word({**BEAR_DETAIL, "phones": ["B EH0"] * 5})


class Utterance(unittest.TestCase):
    """Pairing of reference phonemes with the expert votes."""

    def test_phonemes_paired_with_votes(self):
        words = [{"text": "BEAR", "phones": ["B", "EH0", "R"]}]
        result = evaluate_utterance("u", "b iː ɹ", words, [BEAR_DETAIL])
        self.assertEqual([(p.phoneme, p.system, p.wrong_votes) for p in result.phones],
                         [("B", "match", 0), ("EH", "substitution", 2), ("R", "match", 2)])
        self.assertTrue(result.phones[1].flagged)
        self.assertAlmostEqual(result.phones[1].mean_score, 1.0)


class Detection(unittest.TestCase):
    """Precision / recall bookkeeping."""

    def test_counts_and_rates(self):
        phones = [outcome(5, "substitution"), outcome(4, "match"), outcome(0, "deletion"), outcome(0, "match"),
                  outcome(0, "accepted")]
        scores = detection_scores(phones, lambda p: p.wrong_votes >= 3)
        self.assertEqual((scores.true_positives, scores.false_negatives, scores.false_positives,
                          scores.true_negatives), (1, 1, 1, 2))
        self.assertAlmostEqual(scores.precision, 0.5)
        self.assertAlmostEqual(scores.recall, 0.5)
        self.assertAlmostEqual(scores.false_alarm_rate, 1 / 3)

    def test_ambiguous_phonemes_left_out(self):
        phones = [outcome(2, "substitution"), outcome(5, "substitution")]
        scores = detection_scores(phones, lambda p: None if p.wrong_votes == 2 else p.wrong_votes >= 3)
        self.assertEqual(scores.true_positives + scores.false_positives, 1)

    def test_custom_detector(self):
        phones = [outcome(5, "match")]
        scores = detection_scores(phones, lambda p: True, flagged=lambda p: p.votes[0] == WRONG)
        self.assertEqual(scores.true_positives, 1)

    def test_no_positives(self):
        self.assertTrue(math.isnan(detection_scores([outcome(0, "match")], lambda p: False).recall))


if __name__ == "__main__":
    unittest.main()
