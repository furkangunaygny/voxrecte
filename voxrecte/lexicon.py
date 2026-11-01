"""Pronunciation lexicon built from the speechocean762 reference transcriptions.

resources/lexicon.tsv lists every pronunciation (ARPAbet with stress) a word has in the corpus,
with its count. The most frequent one is the target pronunciation; the other same-length
variants are accepted under rule R8.
"""

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from voxrecte.paths import LEXICON_PATH
from voxrecte.phonemes import strip_stress
from voxrecte.text import words_of

HEADER = "word\tphones\tcount"


class UnknownWordsError(KeyError):
    """Raised when a target sentence contains words that are not in the lexicon."""

    def __init__(self, words: list[str]):
        """Keep the unknown words for the error message."""
        super().__init__(" ".join(words))
        self.words = words


def count_pronunciations(scores: dict) -> Counter:
    """(word, phones) -> number of recordings, over all word entries of scores.json."""
    return Counter((w["text"].upper(), " ".join(w["phones"])) for utt in scores.values() for w in utt["words"])


def write_lexicon(counts: Counter, path: Path = LEXICON_PATH) -> None:
    """Sorted by word, then most frequent pronunciation first."""
    ordered = sorted(counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1]))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER + "\n")
        for (word, phones), count in ordered:
            f.write(f"{word}\t{phones}\t{count}\n")


def load_lexicon(path: Path = LEXICON_PATH) -> dict[str, list[str]]:
    """word -> pronunciations, most frequent first."""
    lexicon: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            word, phones, _count = line.rstrip("\n").split("\t")
            lexicon.setdefault(word, []).append(phones)
    return lexicon


def target_words(text: str, lexicon: dict[str, list[str]]) -> tuple[list[dict], dict[str, set[tuple[str, ...]]]]:
    """Sentence -> (word entries with the target pronunciation, stress-free variants per word for R8)."""
    words = words_of(text)
    unknown = [w for w in words if w not in lexicon]
    if unknown:
        raise UnknownWordsError(unknown)
    entries = [{"text": w, "phones": lexicon[w][0].split()} for w in words]
    return entries, variants_of(words, lexicon)


def variants_of(words: Iterable[str], lexicon: dict[str, list[str]]) -> dict[str, set[tuple[str, ...]]]:
    """Stress-free pronunciations of each word, used for rule R8."""
    return {w: {strip_stress(p.split()) for p in lexicon[w]} for w in words}
