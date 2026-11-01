"""Readers for the prepared speechocean762 corpus.

The manifest (outputs/speechocean762_manifest.csv) is written by scripts/prepare_speechocean762.py;
resource/scores.json holds the experts' sentence, word and phoneme scores and the reference
transcription (ARPAbet) of every word.
"""

import csv
import json
from pathlib import Path

from voxrecte.paths import DATA_DIR, MANIFEST_PATH, corpus_dir, predictions_path

SPLITS = ("train", "test")
SENTENCE_SCORES = ("accuracy", "completeness", "fluency", "prosodic", "total")
CORRECT_PHONEME_SCORE = 2.0  # phoneme scores: 0 wrong or missing, 1 heavy accent, 2 correct


def read_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, str]]:
    """One dict per recording, in manifest order (train first, then test)."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_scores(data_dir: Path = DATA_DIR) -> dict:
    """utt_id -> expert scores and word-level reference transcriptions (resource/scores.json)."""
    return json.loads((corpus_dir(data_dir) / "resource" / "scores.json").read_text(encoding="utf-8"))


def load_predictions(data_dir: Path = DATA_DIR, split: str | None = None) -> dict[str, str]:
    """utt_id -> model IPA output, optionally only for one split."""
    with open(predictions_path(data_dir), encoding="utf-8") as f:
        return {r["utt_id"]: r["ipa"] for r in csv.DictReader(f) if split is None or r["split"] == split}


def all_phonemes_correct(words: list[dict]) -> bool:
    """True if the experts scored every reference phoneme of the recording as correct."""
    return all(score == CORRECT_PHONEME_SCORE for w in words for score in w["phones-accuracy"])
