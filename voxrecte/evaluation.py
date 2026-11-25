"""Phoneme-level evaluation of the comparison against the speechocean762 expert labels.

Expert labels
-------------
Five experts scored every reference phoneme (resource/scores-detail.json). Each expert's
annotation of a word is the reference transcription with marks (corpus README):

    no mark   2  pronunciation is correct
    {EH0}     1  right, but with a heavy accent
    (EH0)     0  incorrect or missed
    [L]          an inserted phone (not part of the reference)

scores.json holds the mean of the five votes. The mean mixes "two experts heard an error"
with "all experts heard an accent", so the evaluation keeps the individual votes and counts,
for every reference phoneme, how many experts marked it as incorrect (0) or accented (1).

System output
-------------
For every reference phoneme the comparison gives one outcome: match, accepted (a rule-covered
difference) or a possible error (substitution or deletion). Insertions have no reference
phoneme; they are evaluated against the experts' [ ] marks separately.
"""

import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from voxrecte.alignment import DELETION, INSERTION, SUBSTITUTION, align
from voxrecte.phonemes import VOWELS, ipa_to_common
from voxrecte.rules import build_reference

CORRECT, ACCENTED, WRONG = 2, 1, 0
_EXPERT_TOKEN = re.compile(r"\[[^\]]*\]|\{[^}]*\}|\([^)]*\)|\S+")
FLAGGED_KINDS = (SUBSTITUTION, DELETION)


@dataclass(frozen=True)
class ExpertWord:
    """The five experts' votes for one word."""
    votes: list[list[int]]  # per reference phoneme: one vote (0/1/2) per expert
    insertions: list[int]  # per expert: number of inserted phones marked in the word


@dataclass(frozen=True)
class PhoneOutcome:
    """One reference phoneme: what the experts said and what the system did."""
    utt_id: str
    word_index: int
    word: str
    phoneme: str
    votes: tuple[int, ...]
    system: str  # alignment kind: match, accepted, substitution or deletion
    heard: str | None  # model phoneme aligned to it (None for a deletion)
    rule: str | None  # rule id for ACCEPTED

    @property
    def wrong_votes(self) -> int:
        """Number of experts who scored the phoneme 0 (incorrect or missed)."""
        return self.votes.count(WRONG)

    @property
    def accent_votes(self) -> int:
        """Number of experts who scored the phoneme 1 (accented)."""
        return self.votes.count(ACCENTED)

    @property
    def mean_score(self) -> float:
        """Mean of the five votes; equals the phoneme score in scores.json."""
        return sum(self.votes) / len(self.votes)

    @property
    def flagged(self) -> bool:
        """The system reports a possible error on this phoneme."""
        return self.system in FLAGGED_KINDS

    @property
    def is_vowel(self) -> bool:
        """True for vowels (including IX)."""
        return self.phoneme in VOWELS


def parse_expert_annotation(annotation: str) -> tuple[list[int], list[str]]:
    """'B (EH0) {R} [L]' -> ([2, 0, 1], ['L']): votes for the reference phonemes, inserted phones."""
    votes, inserted = [], []
    for token in _EXPERT_TOKEN.findall(annotation):
        if token.startswith("["):
            inserted.append(token[1:-1])
        elif token.startswith("{"):
            votes.append(ACCENTED)
        elif token.startswith("("):
            votes.append(WRONG)
        else:
            votes.append(CORRECT)
    return votes, inserted


def expert_word(detail_word: dict) -> ExpertWord:
    """Votes per reference phoneme for one word of scores-detail.json."""
    n_phones = len(detail_word["ref-phones"].split())
    per_expert = [parse_expert_annotation(a) for a in detail_word["phones"]]
    for votes, _ in per_expert:
        if len(votes) != n_phones:
            raise ValueError(f"{detail_word['text']}: expert annotation does not match {detail_word['ref-phones']!r}")
    return ExpertWord(
        votes=[[votes[k] for votes, _ in per_expert] for k in range(n_phones)],
        insertions=[len(inserted) for _, inserted in per_expert],
    )


@dataclass(frozen=True)
class UtteranceResult:
    """Evaluation of one recording: phoneme outcomes, unaccepted insertions, expert votes."""
    phones: list[PhoneOutcome]
    insertions: list[tuple[int, str]]  # (word index, inserted model phoneme) for non-accepted insertions
    experts: list[ExpertWord]


def evaluate_utterance(utt_id: str, ipa: str, words: list[dict], detail_words: list[dict],
                       variants: dict | None = None) -> UtteranceResult:
    """Align the model output with the recording's own reference and pair each phoneme with the votes."""
    reference = build_reference(words, variants)
    heard = ipa_to_common(ipa)
    experts = [expert_word(w) for w in detail_words]
    votes = [tuple(v) for e in experts for v in e.votes]
    if len(votes) != len(reference):
        raise ValueError(f"{utt_id}: {len(votes)} expert-scored phonemes, {len(reference)} reference phonemes")

    phones, insertions, current_word = [], [], 0
    for pair in align(reference, heard):
        if pair.ref_index is None:
            if pair.kind == INSERTION:
                insertions.append((current_word, heard[pair.hyp_index]))
            continue
        ref = reference[pair.ref_index]
        current_word = ref.word_index
        phones.append(PhoneOutcome(
            utt_id=utt_id,
            word_index=ref.word_index,
            word=words[ref.word_index]["text"],
            phoneme=ref.phoneme,
            votes=votes[pair.ref_index],
            system=pair.kind,
            heard=heard[pair.hyp_index] if pair.hyp_index is not None else None,
            rule=pair.rule,
        ))
    return UtteranceResult(phones, insertions, experts)



@dataclass(frozen=True)
class DetectionScores:
    """Confusion counts of a detector against an expert-based error definition."""
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def precision(self) -> float:
        """Share of flagged phonemes that are errors."""
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else float("nan")

    @property
    def recall(self) -> float:
        """Share of errors that are flagged."""
        positives = self.true_positives + self.false_negatives
        return self.true_positives / positives if positives else float("nan")

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else float("nan")

    @property
    def false_alarm_rate(self) -> float:
        """Share of correctly pronounced phonemes that the system flags."""
        negatives = self.false_positives + self.true_negatives
        return self.false_positives / negatives if negatives else float("nan")

    def as_dict(self) -> dict:
        """Counts and rounded rates for the JSON output."""
        return {"tp": self.true_positives, "fp": self.false_positives, "fn": self.false_negatives,
                "tn": self.true_negatives, "precision": round(self.precision, 4), "recall": round(self.recall, 4),
                "f1": round(self.f1, 4), "false_alarm_rate": round(self.false_alarm_rate, 4)}


def detection_scores(phones: Sequence[PhoneOutcome], is_error: Callable[[PhoneOutcome], bool | None],
                     flagged: Callable[[PhoneOutcome], bool] = lambda phone: phone.flagged) -> DetectionScores:
    """Flagged phonemes vs. an expert-based error definition `is_error(phone) -> bool | None`.

    Phonemes for which is_error returns None (ambiguous under that definition) are left out.
    `flagged` decides what counts as a detection (the system by default; e.g. one expert's vote)."""
    tp = fp = fn = tn = 0
    for phone in phones:
        truth = is_error(phone)
        if truth is None:
            continue
        if flagged(phone):
            tp, fp = tp + truth, fp + (not truth)
        else:
            fn, tn = fn + truth, tn + (not truth)
    return DetectionScores(tp, fp, fn, tn)


# --- expert-based error definitions and the human reference ----------------------------------------

N_EXPERTS = 5
MAJORITY = 3

# Expert-based definitions of "incorrect phoneme"; None = left out under that definition.
DEFINITIONS: dict[str, tuple[str, Callable[[PhoneOutcome], bool | None]]] = {
    "majority_wrong": (">= 3 of 5 experts scored 0 (main definition)",
                       lambda p: p.wrong_votes >= MAJORITY),
    "majority_wrong_vs_all_correct": (">= 3 of 5 experts scored 0, against phonemes all 5 scored 2 (others left out)",
                                      lambda p: True if p.wrong_votes >= MAJORITY
                                      else False if p.votes.count(CORRECT) == N_EXPERTS else None),
    "any_wrong": (">= 1 of 5 experts scored 0", lambda p: p.wrong_votes >= 1),
    "majority_not_correct": (">= 3 of 5 experts scored 0 or 1",
                             lambda p: N_EXPERTS - p.votes.count(CORRECT) >= MAJORITY),
}


def is_majority_wrong(phone: PhoneOutcome) -> bool:
    """Main error definition: at least 3 of the 5 experts scored the phoneme 0."""
    return phone.wrong_votes >= MAJORITY


def expert_vs_rest(phones: Sequence[PhoneOutcome],
                   flagged: Callable[[PhoneOutcome], bool] = lambda phone: phone.flagged) -> dict:
    """Each expert as a detector against the majority of the other four (>= 3 of 4 votes 0;
    2-2 ties left out), and the system (`flagged`) against the same reference, averaged over the experts."""
    expert_scores, system_scores = [], []
    for expert in range(N_EXPERTS):
        def others_say_wrong(p, expert=expert):
            """Majority of the other four experts: True / False, None on a 2-2 tie."""
            others = [v for i, v in enumerate(p.votes) if i != expert].count(WRONG)
            return True if others >= 3 else False if others <= 1 else None
        expert_scores.append(detection_scores(phones, others_say_wrong, lambda p, e=expert: p.votes[e] == WRONG))
        system_scores.append(detection_scores(phones, others_say_wrong, flagged))
    mean = lambda scores, attr: round(statistics.mean(getattr(s, attr) for s in scores), 4)
    metrics = ("precision", "recall", "f1", "false_alarm_rate")
    return {"single_expert": {m: mean(expert_scores, m) for m in metrics},
            "system": {m: mean(system_scores, m) for m in metrics}}
