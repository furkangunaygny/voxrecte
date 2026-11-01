"""Sentence normalisation shared by the corpus checks, sentence selection and the lexicon."""

import re

_NON_WORD = re.compile(r"[^A-Z' ]")


def words_of(text: str) -> list[str]:
    """Upper-case words with punctuation removed (apostrophes are kept: "IT'S")."""
    return _NON_WORD.sub(" ", text.upper()).split()


def normalize_sentence(text: str) -> str:
    """Sentence key used to detect repeated sentences: "It's here!" -> "IT'S HERE"."""
    return " ".join(words_of(text))
