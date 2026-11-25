"""Goodness-of-pronunciation (GOP) features from the phoneme model's frame posteriors.

The greedy comparison (comparison.py) only sees the model's single best symbol per frame. These
features keep how confident the model was. For every reference phoneme they measure how
probable the expected sound is where the phoneme was spoken, and how strong the best competing
sound is. A classifier trained on expert labels turns them into an error decision.

Steps
-----
1. Phoneme classes: the model's 392 output symbols are pooled into the common phonemes (39 CMU
   phonemes + DX, Q, IX), a blank class and an "other" class (symbols of other languages). A
   symbol that stands for several phonemes (e.g. "ɑːɹ" -> AA R) counts for each of them, so the
   class scores are not a normalised distribution.
2. Forced alignment: the reference phonemes are placed on the frames with CTC Viterbi
   alignment. A reference phoneme is matched by its own class and by every class an
   acceptable-variant rule allows at that position (e.g. DX for T), so rule-covered differences
   are not penalised.
3. Features per reference phoneme, taken in its window (its own frames plus the blank frames
   up to the neighbouring phonemes, where CTC models often place the evidence).
"""

from dataclasses import dataclass

import numpy as np

from voxrecte.phonemes import EXTRA_SYMBOLS, PHONEMES, UNKNOWN, ipa_token_to_common
from voxrecte.rules import ReferencePhone, substitution_rule

BLANK_CLASS, OTHER_CLASS = "<blank>", "<other>"
PHONEME_CLASSES = tuple(PHONEMES) + EXTRA_SYMBOLS
CLASSES = (BLANK_CLASS,) + PHONEME_CLASSES + (OTHER_CLASS,)
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}
BLANK, OTHER = CLASS_INDEX[BLANK_CLASS], CLASS_INDEX[OTHER_CLASS]
_FLOOR = 1e-10  # probability floor before taking logs


def class_membership(tokens: list[str], special_ids: dict[str, int]) -> np.ndarray:
    """0/1 matrix (vocab_size, len(CLASSES)): which classes each model symbol counts for.

    The CTC blank goes to the blank class; bos/eos/unk and unmapped symbols to "other"."""
    membership = np.zeros((len(tokens), len(CLASSES)), dtype=np.float32)
    non_phoneme = {special_ids["unk"], special_ids["bos"], special_ids["eos"]}
    for token_id, token in enumerate(tokens):
        if token_id == special_ids["blank"]:
            membership[token_id, BLANK] = 1
            continue
        phonemes = [] if token_id in non_phoneme else ipa_token_to_common(token)
        if not phonemes or UNKNOWN in phonemes:
            membership[token_id, OTHER] = 1
            continue
        for phoneme in set(phonemes):
            membership[token_id, CLASS_INDEX[phoneme]] = 1
    return membership


def class_log_scores(log_posteriors: np.ndarray, membership: np.ndarray) -> np.ndarray:
    """Symbol log posteriors (frames, vocab) -> log class scores (frames, len(CLASSES))."""
    return np.log(np.maximum(np.exp(log_posteriors) @ membership, _FLOOR))


def accepted_classes(ref: ReferencePhone) -> list[int]:
    """Class indices that count as a correct realisation of the reference phoneme."""
    return [CLASS_INDEX[c] for c in PHONEME_CLASSES if c == ref.phoneme or substitution_rule(ref, c)]


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log(sum(exp(x))) along an axis."""
    peak = x.max(axis=axis, keepdims=True)
    return np.squeeze(peak, axis) + np.log(np.exp(x - peak).sum(axis=axis))


def ctc_force_align(target_scores: np.ndarray, blank_scores: np.ndarray,
                    labels: list[str]) -> list[tuple[int, int]] | None:
    """CTC Viterbi alignment of a label sequence to the frames.

    target_scores: (frames, labels) log score of each label position at each frame
    blank_scores:  (frames,) log score of the blank
    labels:        label names; equal neighbours must be separated by a blank (CTC rule)
    Returns the (first, last) frame of every label, or None if the labels do not fit."""
    n_frames, n_labels = target_scores.shape
    n_states = 2 * n_labels + 1  # blank, label 0, blank, label 1, ..., blank
    if n_labels == 0 or n_frames < n_labels:
        return None
    emit = np.empty((n_frames, n_states))
    emit[:, 0::2] = blank_scores[:, None]
    emit[:, 1::2] = target_scores
    can_skip = np.zeros(n_states, dtype=bool)  # label state reachable from the previous label directly
    can_skip[3::2] = [labels[k] != labels[k - 1] for k in range(1, n_labels)]

    score = np.full(n_states, -np.inf)
    score[:2] = emit[0, :2]
    back = np.zeros((n_frames, n_states), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    for t in range(1, n_frames):
        stay, step = score, np.concatenate(([-np.inf], score[:-1]))
        skip = np.where(can_skip, np.concatenate(([-np.inf, -np.inf], score[:-2])), -np.inf)
        options = np.stack([stay, step, skip])
        back[t] = options.argmax(axis=0)
        score = options.max(axis=0) + emit[t]

    state = n_states - 1 if score[-1] >= score[-2] else n_states - 2
    if not np.isfinite(score[state]):
        return None
    spans: list[list[int]] = [[-1, -1] for _ in range(n_labels)]
    for t in range(n_frames - 1, -1, -1):
        if state % 2:
            label = state // 2
            spans[label][0] = t
            if spans[label][1] < 0:
                spans[label][1] = t
        state -= int(back[t, state])
    return [(first, last) for first, last in spans]


@dataclass(frozen=True)
class PhoneEvidence:
    """GOP features of one reference phoneme (log scores, frame counts)."""
    accepted_mean: float  # mean score of the accepted classes over the phoneme's own frames
    accepted_peak: float  # best accepted-class score in the window
    own_peak: float  # best score of the exact reference phoneme in the window
    competitor_peak: float  # best score of any non-accepted, non-blank class in the window
    blank_mean: float  # mean blank score over the phoneme's own frames
    frames: int  # frames assigned to the phoneme
    window: int  # frames in the window

    @property
    def margin(self) -> float:
        """Accepted vs. best competitor; negative when another sound is more likely."""
        return self.accepted_peak - self.competitor_peak


def phone_evidence(class_scores: np.ndarray, reference: list[ReferencePhone]) -> list[PhoneEvidence] | None:
    """Forced-align the reference and measure every phoneme; None if the alignment fails."""
    accepted = [accepted_classes(ref) for ref in reference]
    target = np.stack([_logsumexp(class_scores[:, cls], axis=1) for cls in accepted], axis=1)
    spans = ctc_force_align(target, class_scores[:, BLANK], [ref.phoneme for ref in reference])
    if spans is None:
        return None

    n_frames = class_scores.shape[0]
    evidence = []
    for k, (first, last) in enumerate(spans):
        start = spans[k - 1][1] + 1 if k else 0
        end = spans[k + 1][0] - 1 if k + 1 < len(spans) else n_frames - 1
        window = slice(start, end + 1)
        competitors = np.ones(len(CLASSES), dtype=bool)
        competitors[accepted[k] + [BLANK]] = False
        evidence.append(PhoneEvidence(
            accepted_mean=float(target[first:last + 1, k].mean()),
            accepted_peak=float(target[window, k].max()),
            own_peak=float(class_scores[window, CLASS_INDEX[reference[k].phoneme]].max()),
            competitor_peak=float(class_scores[window][:, competitors].max()),
            blank_mean=float(class_scores[first:last + 1, BLANK].mean()),
            frames=last - first + 1,
            window=end - start + 1,
        ))
    return evidence
