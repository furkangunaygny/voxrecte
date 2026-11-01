"""Step 3: select reading sentences from speechocean762 that cover the English phonemes.

1. Sentences are deduplicated after upper-casing and removing punctuation.
2. Phonemes come from the corpus' own reference transcriptions (ARPAbet, stress digits dropped
   for counting, leaving the 39 CMU phonemes).
3. --core sentences are picked greedily (voxrecte/selection.py) with the target that every
   phoneme occurs in --core-k of them; --backup more sentences raise the target by one.
4. Sentences listed in resources/excluded_sentences.csv (broken grammar or sensitive content,
   reviewed by hand because study participants will read them) are not candidates.
5. A pilot subset of --pilot core sentences is picked with the same rule.

Input:   outputs/speechocean762_manifest.csv, data/speechocean762/resource/scores.json,
         resources/excluded_sentences.csv
Outputs (written to outputs/):
  sentence_selection.csv   selected sentences (core/backup, pilot rank), phonemes and source recording IDs
  phoneme_coverage.csv     per-phoneme counts in the corpus and in the selection
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from voxrecte.files import write_csv, write_dict_csv
from voxrecte.paths import DATA_DIR, EXCLUDED_SENTENCES_PATH, OUTPUTS_DIR
from voxrecte.phonemes import PHONEMES, VOWELS
from voxrecte.selection import Selected, Sentence, group_sentences, rarity_weights, select_greedy, tally
from voxrecte.speechocean import load_scores, read_manifest
from voxrecte.text import normalize_sentence


@dataclass
class SelectionResult:
    """Everything the outputs are written from."""
    sentences: dict[str, Sentence]
    candidates: list[Sentence]
    core: list[Selected]
    backup: list[Selected]
    pilot: list[Selected]
    coverage: list[dict]

    @property
    def selected(self) -> list[Selected]:
        """Core sentences followed by the backups."""
        return self.core + self.backup

    @property
    def pilot_rank(self) -> dict[str, int]:
        """Sentence text -> position in the pilot subset."""
        return {s.sentence.text: rank for rank, s in enumerate(self.pilot, 1)}

    def role(self, rank: int) -> str:
        """'core' or 'backup' for a 1-based selection rank."""
        return "core" if rank <= len(self.core) else "backup"


def parse_args():
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--core", type=int, default=20)
    parser.add_argument("--backup", type=int, default=5)
    parser.add_argument("--pilot", type=int, default=5)
    parser.add_argument("--core-k", type=int, default=3, help="target number of core sentences per phoneme")
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=10)
    return parser.parse_args()


def load_excluded(sentences: dict[str, Sentence]) -> dict[str, str]:
    """Normalised sentence -> reason; every excluded sentence must exist in the corpus."""
    with open(EXCLUDED_SENTENCES_PATH, encoding="utf-8") as f:
        excluded = {normalize_sentence(r["text"]): r["reason"] for r in csv.DictReader(f)}
    missing = set(excluded) - set(sentences)
    if missing:
        raise ValueError(f"excluded sentences not found in the corpus: {sorted(missing)}")
    return excluded


def run_selection(args) -> SelectionResult:
    """Group the corpus sentences, filter the candidates, pick core, backup and pilot sentences."""
    sentences = group_sentences(read_manifest(), load_scores(args.data_dir))
    unknown = {p for s in sentences.values() for p in s.phonemes} - set(PHONEMES)
    if unknown:
        raise ValueError(f"phonemes outside the 39-phoneme set: {sorted(unknown)}")
    excluded = load_excluded(sentences)

    candidates = sorted(
        (s for s in sentences.values()
         if args.min_words <= s.num_words <= args.max_words and s.text not in excluded),
        key=lambda s: s.text,
    )
    weights = rarity_weights(sentences.values())
    core = select_greedy(candidates, args.core, args.core_k, weights)
    backup = select_greedy(candidates, args.backup, args.core_k + 1, weights,
                           already_selected=[s.sentence for s in core])
    pilot = select_greedy([s.sentence for s in core], args.pilot, 1, weights)
    result = SelectionResult(sentences, candidates, core, backup, pilot, [])
    result.coverage = coverage_rows(result)
    return result


def coverage_rows(result: SelectionResult) -> list[dict]:
    """Per-phoneme counts in the corpus (unique sentences), the core set and core + backup."""
    corpus_tokens, corpus_sentences = tally(result.sentences.values())
    core_tokens, core_sentences = tally(s.sentence for s in result.core)
    all_tokens, all_sentences = tally(s.sentence for s in result.selected)
    total_corpus, total_core = sum(corpus_tokens.values()), sum(core_tokens.values())
    return [
        {
            "phoneme": p,
            "type": "vowel" if p in VOWELS else "consonant",
            "corpus_count": corpus_tokens[p],
            "corpus_pct": round(100 * corpus_tokens[p] / total_corpus, 2),
            "corpus_sentences": corpus_sentences[p],
            "core_count": core_tokens[p],
            "core_pct": round(100 * core_tokens[p] / total_core, 2),
            "core_sentences": core_sentences[p],
            "core_backup_count": all_tokens[p],
            "core_backup_sentences": all_sentences[p],
            "sentences": " ".join(str(rank) for rank, s in enumerate(result.selected, 1) if p in s.sentence.phonemes),
        }
        for p in sorted(PHONEMES, key=lambda p: (-corpus_tokens[p], p))
    ]


def write_selection_csv(result: SelectionResult, path: Path) -> None:
    """One row per selected sentence, in selection order."""
    headers = ["rank", "role", "pilot_rank", "text", "num_words", "num_phonemes", "distinct_phonemes",
               "new_phonemes", "transcription", "utt_ids", "splits", "mean_total_score"]
    rows = [
        [rank, result.role(rank), result.pilot_rank.get(s.text, ""), s.text, s.num_words,
         sum(s.phonemes.values()), len(s.phonemes), " ".join(sel.new_phonemes), " || ".join(s.variants),
         " ".join(s.utt_ids), " ".join(s.splits), round(s.mean_total, 1)]
        for rank, sel in enumerate(result.selected, 1)
        for s in [sel.sentence]
    ]
    write_csv(path, headers, rows)


def main():
    """Run the selection and write the sentence and coverage tables."""
    args = parse_args()
    result = run_selection(args)
    write_selection_csv(result, OUTPUTS_DIR / "sentence_selection.csv")
    write_dict_csv(OUTPUTS_DIR / "phoneme_coverage.csv", result.coverage)

    min_core = min(c["core_sentences"] for c in result.coverage)
    print(f"{len(result.sentences)} unique sentences, {len(result.candidates)} candidates; "
          f"min core sentences per phoneme {min_core}")
    for rank, sel in enumerate(result.selected, 1):
        pilot = result.pilot_rank.get(sel.sentence.text, "-")
        print(f"{rank:2d} {result.role(rank):<6} p{pilot} {sel.sentence.text}")


if __name__ == "__main__":
    main()
