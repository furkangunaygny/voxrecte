"""Step 4: check the ARPAbet <-> eSpeak IPA mapping on real recordings and write the mapping table.

Only the training split is used, so that the rules are not tuned on the test recordings kept for
step 6. Checks:
  1. every model token seen on train: mapped from the table, after reduction, or unknown;
  2. phoneme error rate with and without the acceptable-variant rules, on all train recordings
     and on "clean" ones, where the experts scored every reference phoneme as correct; on clean
     recordings the remaining differences are model errors or variants without a rule;
  3. word examples for each rule, and the most frequent remaining differences.

The symbol mapping table itself is voxrecte/phonemes.py (IPA_TO_COMMON); the rules are voxrecte/rules.py.

Input:   data/predictions/speechocean762_ipa.csv (transcribe_speechocean762.py),
         data/speechocean762/resource/scores.json
Outputs (written to outputs/):
  phoneme_mapping.csv         symbol mapping table (model IPA token -> common phonemes, uses on train)
  phoneme_mapping_check.json  token mapping, error rates, rule use, remaining differences, word examples
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from voxrecte.alignment import DELETION, INSERTION, SUBSTITUTION, align, count_errors
from voxrecte.comparison import DIFFERENCE_LABELS, STATUS_ACCEPTED, compare
from voxrecte.files import write_csv, write_json
from voxrecte.paths import DATA_DIR, OUTPUTS_DIR
from voxrecte.phonemes import IPA_TO_COMMON, NOT_MAPPED, ipa_to_common, mapping_source
from voxrecte.rules import RULES, build_reference, reference_variants
from voxrecte.speechocean import all_phonemes_correct, load_predictions, load_scores

EXAMPLES_PER_RULE = 4
KIND_OF_LABEL = {label: kind for kind, label in DIFFERENCE_LABELS.items()}


@dataclass
class ErrorRate:
    """Phoneme error rate of a group of recordings, without and with the rules."""
    utterances: int = 0
    reference_phonemes: int = 0
    strict_errors: int = 0
    rule_errors: int = 0
    accepted: int = 0

    def add(self, reference_phonemes: int, strict_errors: int, summary: dict) -> None:
        """Add one recording: reference length, errors without rules, rule-based summary."""
        self.utterances += 1
        self.reference_phonemes += reference_phonemes
        self.strict_errors += strict_errors
        self.rule_errors += summary["possible_errors"]
        self.accepted += summary["accepted_differences"]

    def as_dict(self) -> dict:
        """Counts and error rates in percent."""
        pct = lambda errors: round(100 * errors / self.reference_phonemes, 2)
        return {"utterances": self.utterances, "reference_phonemes": self.reference_phonemes,
                "per_strict_pct": pct(self.strict_errors), "per_with_rules_pct": pct(self.rule_errors),
                "accepted_differences": self.accepted}


@dataclass
class CleanAnalysis:
    """Rule use, remaining differences and word examples on clean recordings."""
    rule_use: Counter = field(default_factory=Counter)
    residual: dict = field(default_factory=lambda: {k: Counter() for k in (SUBSTITUTION, DELETION, INSERTION)})
    examples: dict = field(default_factory=lambda: defaultdict(list))

    def add(self, utt_id: str, words: list[dict], result: dict) -> None:
        """Add the per-word comparison result of one clean recording."""
        for source, word in zip(words, result["words"]):
            for difference in word["accepted"]:
                self.rule_use[difference["rule"]] += 1
            for difference in word["possible_errors"]:
                self._count_residual(difference)
            if word["status"] == STATUS_ACCEPTED:  # only accepted differences: a clean example
                self._add_example(utt_id, source, word)

    def _count_residual(self, difference: dict) -> None:
        """Count a remaining difference by kind ('EH → IY', deleted 'T', inserted 'AH')."""
        kind = KIND_OF_LABEL[difference["type"]]
        key = {SUBSTITUTION: f"{difference['expected']} → {difference['heard']}",
               DELETION: difference["expected"], INSERTION: difference["heard"]}[kind]
        self.residual[kind][key] += 1

    def _add_example(self, utt_id: str, source: dict, word: dict) -> None:
        """Keep the word as an example of each rule it uses (a few distinct words per rule)."""
        for rule in {d["rule"] for d in word["accepted"]}:
            examples = self.examples[rule]
            if len(examples) < EXAMPLES_PER_RULE and word["word"] not in {e["word"] for e in examples}:
                examples.append({"word": word["word"], "utt_id": utt_id,
                                 "ref": " ".join(source["phones"]), "hyp": " ".join(word["heard"])})


def parse_args():
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return parser.parse_args()


def analyse(predictions: dict[str, str], scores: dict) -> tuple[dict[str, ErrorRate], CleanAnalysis]:
    """Error rates on all and on clean train recordings, plus the clean-recording analysis."""
    rates = {"all": ErrorRate(), "clean": ErrorRate()}
    clean = CleanAnalysis()
    variants = reference_variants(w for utt_id in predictions for w in scores[utt_id]["words"])  # train only
    for utt_id, ipa in sorted(predictions.items()):
        words = scores[utt_id]["words"]
        reference = build_reference(words, variants)
        strict_errors = count_errors(align(reference, ipa_to_common(ipa), use_rules=False))
        result = compare(ipa, words, variants)
        groups = ("all", "clean") if all_phonemes_correct(words) else ("all",)
        for group in groups:
            rates[group].add(len(reference), strict_errors, result["summary"])
        if "clean" in groups:
            clean.add(utt_id, words, result)
    return rates, clean


def build_check(tokens: Counter, rates: dict[str, ErrorRate], clean: CleanAnalysis) -> dict:
    """All check results as one JSON-serialisable dict."""
    by_source = Counter()
    for token, count in tokens.items():
        by_source[mapping_source(token)] += count
    return {
        "split": "train",
        "model_tokens": {"distinct": len(tokens), "total": sum(tokens.values()), "by_mapping": dict(by_source)},
        "unknown_tokens": {t: c for t, c in tokens.most_common() if mapping_source(t) == NOT_MAPPED},
        "alignment": {group: rate.as_dict() for group, rate in rates.items()},
        "rule_use_clean": dict(clean.rule_use.most_common()),
        "residual_clean": {kind: dict(counts.most_common(15)) for kind, counts in clean.residual.items()},
        "rule_examples_clean": {rule: clean.examples[rule] for rule in RULES if clean.examples[rule]},
    }


def main():
    """Run the checks on the training split and write the mapping table and the results."""
    args = parse_args()
    predictions = load_predictions(args.data_dir, split="train")
    tokens = Counter(token for ipa in predictions.values() for token in ipa.split())
    rates, clean = analyse(predictions, load_scores(args.data_dir))
    check = build_check(tokens, rates, clean)

    write_json(OUTPUTS_DIR / "phoneme_mapping_check.json", check)
    write_csv(OUTPUTS_DIR / "phoneme_mapping.csv", ["ipa_token", "common", "seen_on_train"],
              [[token, " ".join(common), tokens[token]] for token, common in IPA_TO_COMMON.items()])
    for group, rate in check["alignment"].items():
        print(f"{group}: {rate}")
    print("rule use (clean):", check["rule_use_clean"])


if __name__ == "__main__":
    main()
