"""GOP features: phoneme classes, CTC forced alignment, per-phoneme evidence (voxrecte/gop.py)."""

import unittest

import numpy as np

from voxrecte.gop import (
    BLANK, CLASS_INDEX, CLASSES, OTHER, accepted_classes, class_log_scores, class_membership, ctc_force_align,
    phone_evidence,
)
from voxrecte.rules import build_reference

SPECIAL = {"blank": 0, "unk": 3, "bos": 1, "eos": 2}
TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "k", "æ", "t", "ɾ", "ɑːɹ", "ts"]


def reference(word, phones):
    """Reference phonemes of one word."""
    return build_reference([{"text": word, "phones": phones.split()}])


def frames(*classes, peak=0.9):
    """Class log scores where each frame puts `peak` probability on one class."""
    scores = np.full((len(classes), len(CLASSES)), (1 - peak) / (len(CLASSES) - 1))
    for t, name in enumerate(classes):
        scores[t, CLASS_INDEX[name]] = peak
    return np.log(scores)


class Classes(unittest.TestCase):
    """Pooling of model symbols into phoneme classes."""

    def setUp(self):
        self.membership = class_membership(TOKENS, SPECIAL)

    def test_blank_special_and_unknown_symbols(self):
        self.assertEqual(self.membership[0].nonzero()[0].tolist(), [BLANK])
        for token_id in (1, 2, 3, 9):  # <s>, </s>, <unk>, a Mandarin symbol
            self.assertEqual(self.membership[token_id].nonzero()[0].tolist(), [OTHER])

    def test_multi_phoneme_symbol_counts_for_each(self):
        self.assertEqual(set(self.membership[8].nonzero()[0]), {CLASS_INDEX["AA"], CLASS_INDEX["R"]})

    def test_scores_pool_probabilities(self):
        log_posteriors = np.log(np.array([[0.1, 0, 0, 0, 0.2, 0.3, 0.1, 0.1, 0.2, 0]]) + 1e-12)
        scores = np.exp(class_log_scores(log_posteriors, self.membership))[0]
        self.assertAlmostEqual(scores[CLASS_INDEX["AE"]], 0.3, places=5)
        self.assertAlmostEqual(scores[CLASS_INDEX["R"]], 0.2, places=5)


class AcceptedClasses(unittest.TestCase):
    """Rule-covered variants count as the reference phoneme."""

    def test_flap_accepted_for_t(self):
        t = reference("WATER", "W AO1 T ER0")[2]
        self.assertTrue(set(accepted_classes(t)) >= {CLASS_INDEX["T"], CLASS_INDEX["DX"]})

    def test_plain_consonant_only_itself(self):
        k = reference("CAT", "K AE1 T")[0]
        self.assertEqual(accepted_classes(k), [CLASS_INDEX["K"]])


class ForcedAlignment(unittest.TestCase):
    """CTC Viterbi alignment."""

    def test_spans_follow_the_frames(self):
        scores = frames("<blank>", "K", "K", "<blank>", "AE", "T", "<blank>")
        labels = ["K", "AE", "T"]
        target = scores[:, [CLASS_INDEX[c] for c in labels]]
        self.assertEqual(ctc_force_align(target, scores[:, BLANK], labels), [(1, 2), (4, 4), (5, 5)])

    def test_repeated_label_needs_a_blank(self):
        scores = frames("T", "<blank>", "T")
        target = scores[:, [CLASS_INDEX["T"]] * 2]
        self.assertEqual(ctc_force_align(target, scores[:, BLANK], ["T", "T"]), [(0, 0), (2, 2)])
        self.assertIsNone(ctc_force_align(target[:2], scores[:2, BLANK], ["T", "T"]))

    def test_too_few_frames(self):
        scores = frames("K")
        target = scores[:, [CLASS_INDEX["K"], CLASS_INDEX["AE"]]]
        self.assertIsNone(ctc_force_align(target, scores[:, BLANK], ["K", "AE"]))


class Evidence(unittest.TestCase):
    """Per-phoneme scores separate a correct sound from a replaced one."""

    def test_substituted_vowel_has_negative_margin(self):
        ref = reference("CAT", "K AE1 T")
        good = phone_evidence(frames("<blank>", "K", "AE", "T", "<blank>"), ref)
        bad = phone_evidence(frames("<blank>", "K", "EH", "T", "<blank>"), ref)
        self.assertGreater(good[1].margin, 0)
        self.assertLess(bad[1].margin, 0)
        self.assertGreater(bad[0].margin, 0)  # K is unaffected

    def test_rule_variant_is_not_penalised(self):
        ref = reference("WATER", "W AO1 T ER0")
        evidence = phone_evidence(frames("W", "AO", "DX", "ER"), ref)
        self.assertGreater(evidence[2].margin, 0)


if __name__ == "__main__":
    unittest.main()
