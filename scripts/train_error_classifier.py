"""Train a phoneme error classifier on GOP features and evaluate it like step 6.

The step 5 comparison flags a phoneme whenever the model's single best symbol differs from the
reference; that gives many false alarms (step 6). Here a classifier learns from the experts'
labels when a difference is a real error. Its inputs per reference phoneme:

  - GOP features from the model's frame posteriors (voxrecte/gop.py): how probable the expected
    sound is, how strong the best competitor is, durations;
  - context: the phoneme, stress, position in the word, the scores of the neighbouring phonemes
    and the recording's overall level (speakers and recording conditions differ);
  - what the step 5 comparison did with the phoneme (match, accepted, substitution, deletion).

Target: the step 6 main definition, >= 3 of 5 experts scored the phoneme 0.

Protocol (the test split is used once, at the end)
  1. Model choice and threshold: 5-fold cross-validation on the training split, folds split by
     speaker; candidates are logistic regression and gradient-boosted trees, chosen by average
     precision. Two thresholds: the one with the best F1, and the one that keeps the recall of
     the step 5 comparison (to compare precision and false alarms at equal recall).
  2. The chosen model is trained on the whole training split and evaluated on the test split
     with the step 6 metrics, next to the step 5 comparison on the same phonemes.

Input:   data/posteriors/speechocean762_class_scores.npz (extract_class_scores.py),
         data/predictions/speechocean762_ipa.csv (transcribe_speechocean762.py),
         data/speechocean762/resource/scores.json and scores-detail.json
Outputs: outputs/error_classifier_metrics.json   model choice, thresholds, test metrics, feature importance
         data/models/error_classifier.joblib     trained model (not tracked by git)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from voxrecte.evaluation import (
    DEFINITIONS, MAJORITY, PhoneOutcome, detection_scores, evaluate_utterance, expert_vs_rest, is_majority_wrong,
)
from voxrecte.files import write_json
from voxrecte.gop import PhoneEvidence, phone_evidence
from voxrecte.paths import DATA_DIR, OUTPUTS_DIR, class_scores_path, corpus_dir, models_dir
from voxrecte.phonemes import PHONEMES
from voxrecte.rules import build_reference, reference_variants
from voxrecte.speechocean import load_predictions, load_scores, read_manifest

SEED = 0
N_FOLDS = 5
SYSTEM_KINDS = ("match", "accepted", "substitution", "deletion")
WORD_POSITIONS = ("single", "first", "middle", "last")

NUMERIC_FEATURES = [
    "accepted_mean", "accepted_peak", "own_peak", "competitor_peak", "margin", "blank_mean", "frames", "window",
    "prev_margin", "next_margin", "utt_margin_median", "utt_flag_rate",
]
CATEGORICAL_FEATURES = ["phoneme", "system", "stress", "word_position"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --- data ------------------------------------------------------------------------------------------

class PhoneTable:
    """Feature matrix, labels and phoneme outcomes of one split, row-aligned."""

    def __init__(self):
        """Start empty; rows are added per recording."""
        self.rows: list[list[float]] = []
        self.phones: list[PhoneOutcome] = []
        self.speakers: list[str] = []
        self.age_groups: list[str] = []

    def add_recording(self, phones: list[PhoneOutcome], evidence: list[PhoneEvidence], reference,
                      word_lengths: list[int], speaker: str, age_group: str) -> None:
        """Append one row per reference phoneme of a recording."""
        margins = [e.margin for e in evidence]
        utt_margin = float(np.median(margins))
        utt_flag_rate = float(np.mean([p.flagged for p in phones]))
        position_in_word = _positions_in_word(reference, word_lengths)
        for k, (phone, ev, ref) in enumerate(zip(phones, evidence, reference)):
            self.rows.append([
                ev.accepted_mean, ev.accepted_peak, ev.own_peak, ev.competitor_peak, ev.margin, ev.blank_mean,
                ev.frames, ev.window,
                margins[k - 1] if k else utt_margin, margins[k + 1] if k + 1 < len(margins) else utt_margin,
                utt_margin, utt_flag_rate,
                PHONEMES.index(ref.phoneme), SYSTEM_KINDS.index(phone.system),
                -1 if ref.stress is None else ref.stress, WORD_POSITIONS.index(position_in_word[k]),
            ])
            self.phones.append(phone)
            self.speakers.append(speaker)
            self.age_groups.append(age_group)

    @property
    def x(self) -> np.ndarray:
        """Feature matrix (rows, FEATURES)."""
        return np.asarray(self.rows, dtype=np.float64)

    @property
    def y(self) -> np.ndarray:
        """Main-definition labels: >= 3 of 5 experts scored the phoneme 0."""
        return np.array([is_majority_wrong(p) for p in self.phones])

    @property
    def baseline(self) -> np.ndarray:
        """Step 5 comparison decisions (substitution or deletion)."""
        return np.array([p.flagged for p in self.phones])


def _positions_in_word(reference, word_lengths: list[int]) -> list[str]:
    """'single' / 'first' / 'middle' / 'last' for every reference phoneme."""
    positions, seen = [], defaultdict(int)
    for ref in reference:
        index, length = seen[ref.word_index], word_lengths[ref.word_index]
        seen[ref.word_index] += 1
        positions.append("single" if length == 1 else "first" if index == 0
                         else "last" if index == length - 1 else "middle")
    return positions


def build_tables(data_dir: Path) -> tuple[dict[str, PhoneTable], int]:
    """Phone tables for train and test; returns them and the number of recordings skipped."""
    manifest = read_manifest()
    scores = load_scores(data_dir)
    detail = json.loads((corpus_dir(data_dir) / "resource" / "scores-detail.json").read_text(encoding="utf-8"))
    variants = reference_variants(w for r in manifest if r["split"] == "train" for w in scores[r["utt_id"]]["words"])
    predictions = load_predictions(data_dir)
    class_scores = np.load(class_scores_path(data_dir))

    tables, skipped = {"train": PhoneTable(), "test": PhoneTable()}, 0
    for row in manifest:
        utt_id, words = row["utt_id"], scores[row["utt_id"]]["words"]
        reference = build_reference(words, variants)
        evidence = phone_evidence(class_scores[utt_id].astype(np.float32), reference)
        if evidence is None:
            skipped += 1
            continue
        result = evaluate_utterance(utt_id, predictions[utt_id], words, detail[utt_id]["words"], variants)
        tables[row["split"]].add_recording(result.phones, evidence, reference, [len(w["phones"]) for w in words],
                                           row["speaker_id"], row["age_group"])
    return tables, skipped


# --- models ----------------------------------------------------------------------------------------

def candidate_models() -> dict:
    """Classifiers compared by cross-validation."""
    numeric = list(range(len(NUMERIC_FEATURES)))
    categorical = list(range(len(NUMERIC_FEATURES), len(FEATURES)))
    logistic = make_pipeline(
        ColumnTransformer([("num", StandardScaler(), numeric),
                           ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)]),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    boosting = HistGradientBoostingClassifier(
        categorical_features=categorical, learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=SEED,
    )
    return {"logistic_regression": logistic, "gradient_boosting": boosting}


def out_of_fold(model, table: PhoneTable) -> np.ndarray:
    """Cross-validated error probabilities on the training split, folds split by speaker."""
    x, y, probabilities = table.x, table.y, np.zeros(len(table.rows))
    for fit_idx, val_idx in GroupKFold(n_splits=N_FOLDS).split(x, y, groups=table.speakers):
        model.fit(x[fit_idx], y[fit_idx])
        probabilities[val_idx] = model.predict_proba(x[val_idx])[:, 1]
    return probabilities


def thresholds(y: np.ndarray, probabilities: np.ndarray, target_recall: float) -> dict:
    """Threshold with the best F1, and the highest threshold that keeps `target_recall`."""
    precision, recall, cut = precision_recall_curve(y, probabilities)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    keeps_recall = np.nonzero(recall >= target_recall)[0]
    return {"best_f1": float(cut[f1.argmax()]), "baseline_recall": float(cut[keeps_recall.max()])}


# --- evaluation ------------------------------------------------------------------------------------

def detection_report(table: PhoneTable, flags: np.ndarray) -> dict:
    """Step 6 metrics for a set of decisions on a phone table."""
    flag_of = {id(p): bool(f) for p, f in zip(table.phones, flags)}
    flagged = lambda p: flag_of[id(p)]
    phones = table.phones
    by_votes = defaultdict(list)
    for p, f in zip(phones, flags):
        by_votes[p.wrong_votes].append(bool(f))
    return {
        "detection": {key: {"definition": label, **detection_scores(phones, rule, flagged).as_dict()}
                      for key, (label, rule) in DEFINITIONS.items()},
        "flag_rate_by_wrong_votes": {k: round(100 * float(np.mean(v)), 1) for k, v in sorted(by_votes.items())},
        "expert_vs_rest": expert_vs_rest(phones, flagged)["system"],
        "age_groups": {group: detection_scores([p for p, g in zip(phones, table.age_groups) if g == group],
                                               is_majority_wrong, flagged).as_dict()
                       for group in ("adult", "child")},
    }


def feature_importance(model, table: PhoneTable) -> list[dict]:
    """Drop in average precision when one feature is shuffled (test split)."""
    result = permutation_importance(model, table.x, table.y, scoring="average_precision", n_repeats=5,
                                    random_state=SEED)
    order = np.argsort(-result.importances_mean)
    return [{"feature": FEATURES[i], "ap_drop": round(float(result.importances_mean[i]), 4)} for i in order]


def main():
    """Choose a model by cross-validation on train, then evaluate it once on test."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    tables, skipped = build_tables(args.data_dir)
    train, test = tables["train"], tables["test"]
    print(f"phonemes: train {len(train.rows)}, test {len(test.rows)}; recordings not aligned: {skipped}")

    baseline_train = detection_scores(train.phones, is_majority_wrong)
    cv = {}
    for name, model in candidate_models().items():
        probabilities = out_of_fold(model, train)
        cv[name] = {"average_precision": round(float(average_precision_score(train.y, probabilities)), 4),
                    "roc_auc": round(float(roc_auc_score(train.y, probabilities)), 4),
                    "thresholds": thresholds(train.y, probabilities, baseline_train.recall)}
        print(f"cv {name}: AP={cv[name]['average_precision']} AUC={cv[name]['roc_auc']}")
    chosen = max(cv, key=lambda name: cv[name]["average_precision"])

    model = candidate_models()[chosen].fit(train.x, train.y)
    probabilities = model.predict_proba(test.x)[:, 1]
    test_results = {"step5_comparison": detection_report(test, test.baseline)}
    for label, cut in cv[chosen]["thresholds"].items():
        test_results[f"classifier_{label}_threshold"] = {"threshold": round(cut, 4),
                                                        **detection_report(test, probabilities >= cut)}

    metrics = {
        "target": f">= {MAJORITY} of 5 experts scored the phoneme 0",
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "data": {"train_phonemes": len(train.rows), "test_phonemes": len(test.rows),
                 "train_errors": int(train.y.sum()), "test_errors": int(test.y.sum()),
                 "recordings_not_aligned": skipped},
        "cross_validation": {"folds": N_FOLDS, "split_by": "speaker", "models": cv, "chosen": chosen,
                             "step5_train_recall": round(baseline_train.recall, 4)},
        "test": {"average_precision": round(float(average_precision_score(test.y, probabilities)), 4),
                 "roc_auc": round(float(roc_auc_score(test.y, probabilities)), 4),
                 "step5_comparison_average_precision": round(float(average_precision_score(test.y, test.baseline)), 4),
                 **test_results},
        "feature_importance": feature_importance(model, test),
    }
    write_json(OUTPUTS_DIR / "error_classifier_metrics.json", metrics)
    models_dir(args.data_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "thresholds": cv[chosen]["thresholds"]},
                models_dir(args.data_dir) / "error_classifier.joblib")

    for name, result in test_results.items():
        d = result["detection"]["majority_wrong"]
        print(f"test {name}: P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} "
              f"FA={d['false_alarm_rate']:.3f}")


if __name__ == "__main__":
    main()
