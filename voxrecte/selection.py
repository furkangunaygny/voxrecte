"""Greedy selection of reading sentences that cover the English phonemes.

At each step the candidate with the best score is added; scores are compared in order:
  a. number of phonemes not yet covered by any selected sentence
  b. number of phonemes that still occur in fewer than `target` selected sentences
  c. diminishing-returns gain: each phoneme adds weight / (1 + selected sentences containing it),
     with a higher weight for phonemes that are rare in the corpus
  d. fewer words, then the sentence text (so the result is deterministic)
"""

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property

from voxrecte.phonemes import strip_stress
from voxrecte.text import normalize_sentence

WORD_SEPARATOR = " | "


@dataclass
class Sentence:
    """A unique corpus sentence and the recordings that read it."""
    text: str
    utt_ids: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)  # "speaker (gender, age, age group)"
    totals: list[float] = field(default_factory=list)  # expert total score per recording
    transcriptions: list[str] = field(default_factory=list)  # stress-free, words joined by WORD_SEPARATOR

    @property
    def num_words(self) -> int:
        """Number of words in the sentence."""
        return len(self.text.split())

    @property
    def mean_total(self) -> float:
        """Mean expert total score of the recordings of this sentence."""
        return sum(self.totals) / len(self.totals)

    @cached_property
    def variants(self) -> list[str]:
        """Distinct reference transcriptions (recordings of the same sentence can differ, e.g. F AO / F AO R)."""
        return sorted(set(self.transcriptions))

    @cached_property
    def phonemes(self) -> Counter:
        """Phoneme counts; a phoneme counts only as often as it occurs in every transcription."""
        counts = [Counter(t.replace("|", " ").split()) for t in self.transcriptions]
        return +Counter({p: min(c[p] for c in counts) for p in counts[0]})  # unary + drops zero counts


@dataclass(frozen=True)
class Selected:
    """A selected sentence and the phonemes it covered first."""
    sentence: Sentence
    new_phonemes: tuple[str, ...]  # phonemes this sentence covered first


def group_sentences(manifest: Iterable[dict], scores: dict) -> dict[str, Sentence]:
    """Normalised sentence text -> Sentence, built from the manifest rows and scores.json."""
    sentences: dict[str, Sentence] = {}
    for row in manifest:
        text = normalize_sentence(row["text"])
        sentence = sentences.setdefault(text, Sentence(text))
        sentence.utt_ids.append(row["utt_id"])
        sentence.splits.append(row["split"])
        sentence.speakers.append(f"{row['speaker_id']} ({row['gender']}, {row['age']}, {row['age_group']})")
        sentence.totals.append(float(row["total"]))
        words = scores[row["utt_id"]]["words"]
        sentence.transcriptions.append(WORD_SEPARATOR.join(" ".join(strip_stress(w["phones"])) for w in words))
    return sentences


def tally(sentences: Iterable[Sentence]) -> tuple[Counter, Counter]:
    """(phoneme occurrences, number of sentences containing each phoneme)."""
    tokens, containing = Counter(), Counter()
    for sentence in sentences:
        tokens.update(sentence.phonemes)
        containing.update(sentence.phonemes.keys())
    return tokens, containing


def rarity_weights(sentences: Iterable[Sentence]) -> dict[str, float]:
    """Higher weight for phonemes that occur in fewer corpus sentences (only phonemes that occur)."""
    _, containing = tally(sentences)
    return {p: 1 / math.log(1 + n) for p, n in containing.items()}


def select_greedy(candidates: Sequence[Sentence], n: int, target: int, weights: dict[str, float],
                  already_selected: Sequence[Sentence] = ()) -> list[Selected]:
    """Pick n sentences from `candidates`, continuing from `already_selected`."""
    containing = Counter(p for s in already_selected for p in s.phonemes)
    chosen = {s.text for s in already_selected}

    def score(sentence: Sentence) -> tuple:
        """Selection score, compared as a tuple (see the module docstring)."""
        phonemes = sentence.phonemes
        new = sum(containing[p] == 0 for p in phonemes)
        below_target = sum(containing[p] < target for p in phonemes)
        balance = sum(weights[p] / (1 + containing[p]) for p in phonemes)
        return new, below_target, balance, -sentence.num_words, sentence.text

    picked = []
    for _ in range(n):
        best = max((s for s in candidates if s.text not in chosen), key=score)
        picked.append(Selected(best, tuple(sorted(p for p in best.phonemes if containing[p] == 0))))
        chosen.add(best.text)
        containing.update(best.phonemes.keys())
    return picked
