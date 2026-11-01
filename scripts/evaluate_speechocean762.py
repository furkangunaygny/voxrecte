"""Step 6: evaluate the pronunciation comparison on the speechocean762 test recordings.

The system (wav2vec 2.0 phoneme model + alignment with acceptable-variant rules) is run on all
2500 test recordings and compared with the five experts' phoneme labels (voxrecte/evaluation.py).
Acceptable-variant rules use reference variants from the training split only.

Metrics
  1. flag rate by the number of experts who marked a phoneme as incorrect (no binarisation);
  2. precision / recall / F1 with "incorrect" defined by expert majority (>= 3 of 5 votes 0),
     plus alternative definitions as a sensitivity check;
  3. the same detection scores for each single expert against the other four (human reference);
  4. correlation with the experts' mean phoneme score, word accuracy and sentence accuracy;
  5. breakdowns by age group and by phoneme; inserted phones against the experts' [ ] marks;
  6. the most frequent false alarms, misses and correct detections, with example words.

Input:   data/predictions/speechocean762_ipa.csv (transcribe_speechocean762.py),
         data/speechocean762/resource/scores.json and scores-detail.json
Outputs (written to outputs/):
  evaluation_metrics.json      all metrics and error patterns
  evaluation_utterances.csv    per test recording: expert scores, system counts, model output
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

from voxrecte.evaluation import (
    CORRECT, WRONG, DetectionScores, PhoneOutcome, UtteranceResult, detection_scores, evaluate_utterance,
)
from voxrecte.files import write_csv, write_json
from voxrecte.paths import DATA_DIR, OUTPUTS_DIR, corpus_dir
from voxrecte.phonemes import PHONEMES
from voxrecte.rules import reference_variants
from voxrecte.speechocean import load_predictions, load_scores, read_manifest

N_EXPERTS = 5
MAJORITY = 3
TOP_PATTERNS = 12
EXAMPLES_PER_PATTERN = 3

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


@dataclass
class TestSet:
    """Evaluation results for all test recordings."""
    rows: list[dict]
    results: dict[str, UtteranceResult]
    scores: dict
    predictions: dict[str, str]
    phones: list[PhoneOutcome] = field(default_factory=list)

    def __post_init__(self):
        """Collect the phoneme outcomes of all recordings in one list."""
        self.phones = [p for r in self.results.values() for p in r.phones]

    def phones_of(self, utt_ids: set[str]) -> list[PhoneOutcome]:
        """Phoneme outcomes of the given recordings."""
        return [p for p in self.phones if p.utt_id in utt_ids]


def parse_args():
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return parser.parse_args()


def run_test_set(data_dir: Path) -> TestSet:
    """Evaluate every test recording; variants for rule R8 come from the training split only."""
    manifest = read_manifest()
    scores = load_scores(data_dir)
    detail = json.loads((corpus_dir(data_dir) / "resource" / "scores-detail.json").read_text(encoding="utf-8"))
    train_words = (w for r in manifest if r["split"] == "train" for w in scores[r["utt_id"]]["words"])
    variants = reference_variants(train_words)
    predictions = load_predictions(data_dir, split="test")
    rows = [r for r in manifest if r["split"] == "test"]
    results = {r["utt_id"]: evaluate_utterance(r["utt_id"], predictions[r["utt_id"]], scores[r["utt_id"]]["words"],
                                               detail[r["utt_id"]]["words"], variants) for r in rows}
    return TestSet(rows, results, scores, predictions)


# --- metrics --------------------------------------------------------------------------------------

def flag_rate_by_votes(phones: Sequence[PhoneOutcome]) -> dict:
    """Share of flagged phonemes for each number of 'incorrect' votes, and for 'accented' votes
    among phonemes no expert marked as incorrect."""
    by_wrong, by_accent = defaultdict(list), defaultdict(list)
    for p in phones:
        by_wrong[p.wrong_votes].append(p.flagged)
        if p.wrong_votes == 0:
            by_accent[p.accent_votes].append(p.flagged)
    summarise = lambda groups: {k: {"phonemes": len(v), "flagged_pct": round(100 * sum(v) / len(v), 1)}
                                for k, v in sorted(groups.items())}
    return {"by_wrong_votes": summarise(by_wrong), "by_accent_votes_without_wrong": summarise(by_accent)}


def expert_vs_rest(phones: Sequence[PhoneOutcome]) -> dict:
    """Each expert as a detector against the majority of the other four (>= 3 of 4 votes 0;
    2-2 ties left out), and the system against the same reference, averaged over the experts."""
    expert_scores, system_scores = [], []
    for expert in range(N_EXPERTS):
        def others_say_wrong(p, expert=expert):
            """Majority of the other four experts: True / False, None on a 2-2 tie."""
            others = [v for i, v in enumerate(p.votes) if i != expert].count(WRONG)
            return True if others >= 3 else False if others <= 1 else None
        expert_scores.append(detection_scores(phones, others_say_wrong, lambda p, e=expert: p.votes[e] == WRONG))
        system_scores.append(detection_scores(phones, others_say_wrong))
    mean = lambda scores, attr: round(statistics.mean(getattr(s, attr) for s in scores), 4)
    metrics = ("precision", "recall", "f1", "false_alarm_rate")
    return {"single_expert": {m: mean(expert_scores, m) for m in metrics},
            "system": {m: mean(system_scores, m) for m in metrics}}


def correlations(test: TestSet) -> dict:
    """Agreement with the experts' continuous scores at phoneme, word and sentence level."""
    phone_x = [0.0 if p.flagged else 1.0 for p in test.phones]
    phone_y = [p.mean_score for p in test.phones]
    word_x, word_y, utt_x, utt_y = [], [], [], []
    for row in test.rows:
        result = test.results[row["utt_id"]]
        by_word = defaultdict(list)
        for p in result.phones:
            by_word[p.word_index].append(not p.flagged)
        for w_i, word in enumerate(test.scores[row["utt_id"]]["words"]):
            word_x.append(statistics.mean(by_word[w_i]))
            word_y.append(word["accuracy"])
        utt_x.append(statistics.mean(not p.flagged for p in result.phones))
        utt_y.append(float(row["accuracy"]))
    return {
        "phoneme_vs_mean_score": _correlation(phone_x, phone_y),
        "word_vs_word_accuracy": _correlation(word_x, word_y),
        "sentence_vs_sentence_accuracy": _correlation(utt_x, utt_y),
    }


def _correlation(x, y) -> dict:
    """Pearson and Spearman correlation of two equally long sequences."""
    return {"n": len(x), "pearson": round(float(pearsonr(x, y)[0]), 4), "spearman": round(float(spearmanr(x, y)[0]), 4)}


def by_age_group(test: TestSet) -> dict:
    """Detection scores (main definition) for adults and children separately."""
    is_error = DEFINITIONS["majority_wrong"][1]
    groups = {}
    for group in ("adult", "child"):
        utt_ids = {r["utt_id"] for r in test.rows if r["age_group"] == group}
        phones = test.phones_of(utt_ids)
        groups[group] = {"utterances": len(utt_ids), "phonemes": len(phones),
                         **detection_scores(phones, is_error).as_dict()}
    return groups


def by_phoneme(phones: Sequence[PhoneOutcome]) -> list[dict]:
    """Detection scores (main definition) for each of the 39 phonemes."""
    is_error = DEFINITIONS["majority_wrong"][1]
    rows = []
    for phoneme in PHONEMES:
        subset = [p for p in phones if p.phoneme == phoneme]
        scores = detection_scores(subset, is_error)
        rows.append({"phoneme": phoneme, "phonemes": len(subset),
                     "expert_errors": scores.true_positives + scores.false_negatives,
                     "recall": scores.recall, "false_alarm_rate": scores.false_alarm_rate,
                     "precision": scores.precision})
    return rows


def by_phoneme_class(phones: Sequence[PhoneOutcome]) -> dict:
    """Detection scores (main definition) for vowels and consonants."""
    is_error = DEFINITIONS["majority_wrong"][1]
    return {name: detection_scores([p for p in phones if p.is_vowel == vowel], is_error).as_dict()
            for name, vowel in (("vowel", True), ("consonant", False))}


def insertion_scores(test: TestSet) -> dict:
    """Words where the system inserted a phoneme vs. words where >= 3 experts marked an insertion."""
    tp = fp = fn = tn = 0
    for result in test.results.values():
        system_words = {w for w, _ in result.insertions}
        for w_i, expert in enumerate(result.experts):
            truth = sum(n > 0 for n in expert.insertions) >= MAJORITY
            flagged = w_i in system_words
            tp, fp, fn, tn = tp + (flagged and truth), fp + (flagged and not truth), \
                fn + (truth and not flagged), tn + (not flagged and not truth)
    return DetectionScores(tp, fp, fn, tn).as_dict()


def error_patterns(phones: Sequence[PhoneOutcome], select: Callable[[PhoneOutcome], bool]) -> list[dict]:
    """Most frequent system outcomes ('EH -> IY', 'T -> (deleted)', 'R (not flagged)') among selected phonemes."""
    counts, examples = Counter(), defaultdict(list)
    for p in phones:
        if not select(p):
            continue
        key = _pattern(p)
        counts[key] += 1
        if len(examples[key]) < EXAMPLES_PER_PATTERN and p.word not in examples[key]:
            examples[key].append(p.word)
    return [{"pattern": k, "count": n, "examples": examples[k]} for k, n in counts.most_common(TOP_PATTERNS)]


def _pattern(p: PhoneOutcome) -> str:
    """Short label of what the system did with a phoneme."""
    if p.system == "deletion":
        return f"{p.phoneme} -> (deleted)"
    if p.system in ("match", "accepted"):
        return f"{p.phoneme} (not flagged)"
    return f"{p.phoneme} -> {p.heard}"


def build_metrics(test: TestSet) -> dict:
    """All evaluation results as one JSON-serialisable dict."""
    phones = test.phones
    return {
        "split": "test",
        "utterances": len(test.rows),
        "reference_phonemes": len(phones),
        "expert_votes": {"all_correct_pct": round(100 * sum(p.votes.count(CORRECT) == N_EXPERTS for p in phones)
                                                  / len(phones), 2),
                         "majority_wrong_pct": round(100 * sum(p.wrong_votes >= MAJORITY for p in phones)
                                                     / len(phones), 2)},
        "system_outcomes": dict(Counter(p.system for p in phones)),
        "flag_rate": flag_rate_by_votes(phones),
        "detection": {key: {"definition": label, **detection_scores(phones, rule).as_dict()}
                      for key, (label, rule) in DEFINITIONS.items()},
        "expert_vs_rest": expert_vs_rest(phones),
        "correlations": correlations(test),
        "age_groups": by_age_group(test),
        "insertions": insertion_scores(test),
        "phoneme_classes": by_phoneme_class(phones),
        "phonemes": by_phoneme(phones),
        "false_alarms": error_patterns(phones, lambda p: p.flagged and p.votes.count(CORRECT) == N_EXPERTS),
        "misses": error_patterns(phones, lambda p: not p.flagged and p.wrong_votes >= MAJORITY),
        "detections": error_patterns(phones, lambda p: p.flagged and p.wrong_votes >= MAJORITY),
    }


# --- outputs --------------------------------------------------------------------------------------

def write_utterances(test: TestSet, path: Path) -> None:
    """One row per test recording: expert scores, system counts and the model output."""
    headers = ["utt_id", "speaker_id", "age_group", "text", "expert_accuracy", "expert_total", "target_phonemes",
               "possible_errors", "accepted", "insertions", "match_rate", "model_ipa"]
    rows = []
    for row in test.rows:
        result = test.results[row["utt_id"]]
        flagged = sum(p.flagged for p in result.phones)
        rows.append([row["utt_id"], row["speaker_id"], row["age_group"], row["text"], row["accuracy"], row["total"],
                     len(result.phones), flagged, sum(p.system == "accepted" for p in result.phones),
                     len(result.insertions), round(1 - flagged / len(result.phones), 3),
                     test.predictions[row["utt_id"]]])
    write_csv(path, headers, rows)


def main():
    """Evaluate the test split and write the metrics and the per-recording table."""
    args = parse_args()
    test = run_test_set(args.data_dir)
    metrics = build_metrics(test)
    write_json(OUTPUTS_DIR / "evaluation_metrics.json", metrics)
    write_utterances(test, OUTPUTS_DIR / "evaluation_utterances.csv")
    for key, d in metrics["detection"].items():
        print(f"{key}: P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} FA={d['false_alarm_rate']:.3f}")
    print("expert vs rest:", metrics["expert_vs_rest"])
    print("correlations:", metrics["correlations"])
    print("age:", {g: (v["f1"], v["false_alarm_rate"]) for g, v in metrics["age_groups"].items()})
    print("insertions:", metrics["insertions"])


if __name__ == "__main__":
    main()
