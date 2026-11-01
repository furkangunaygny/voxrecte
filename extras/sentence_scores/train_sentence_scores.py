"""Train a sentence-level pronunciation score predictor on frozen wav2vec 2.0 features.

Additional experiment, not a step of the project plan (kept in extras/).

A ridge regressor maps the mean-pooled hidden state of one model layer to the expert sentence
scores (accuracy, fluency, prosodic, total). The layer and the regularisation strength are chosen
on a speaker-disjoint validation split taken from the training set; the chosen setting is refitted
on the whole training set and scored once on test.

Completeness is not modelled: 4975 of 5000 utterances have the maximum score of 10.

Input:   outputs/speechocean762_manifest.csv, data/features/wav2vec2_meanpool.npy (extract_features.py)
Outputs (written to extras/sentence_scores/outputs/):
  sentence_scores_split.json            validation speakers
  sentence_scores_baseline.json         chosen settings, metrics, per-layer validation curve
  sentence_scores_test_predictions.csv  per-utterance test predictions
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from voxrecte.files import write_csv, write_json
from voxrecte.paths import DATA_DIR, features_dir
from voxrecte.speechocean import read_manifest

TARGETS = ("accuracy", "fluency", "prosodic", "total")
ALPHAS = (1, 10, 100, 1000, 10000, 100000)
SCORE_RANGE = (0, 10)
SEED = 0
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"  # kept apart from the plan deliverables


@dataclass
class Dataset:
    """Features, expert scores and the fit / validation / test split."""
    rows: list[dict]
    features: np.ndarray  # (utterances, layers, hidden_size)
    scores: np.ndarray  # (utterances, len(TARGETS))
    feature_meta: dict
    val_speakers: list[str]
    fit_idx: np.ndarray  # train minus validation speakers
    val_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def train_idx(self) -> np.ndarray:
        """All training recordings: fit + validation."""
        return np.concatenate([self.fit_idx, self.val_idx])


def pcc(y, p) -> float:
    """Pearson correlation coefficient."""
    return float(np.corrcoef(y, p)[0, 1])


def mse(y, p) -> float:
    """Mean squared error."""
    return float(np.mean((y - p) ** 2))


def fit(x, y, alpha):
    """Standardise the features and fit a ridge regressor."""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x, y)


def split_speakers(rows: list[dict], n_val: int) -> list[str]:
    """Pick n_val training speakers for validation, keeping the child/adult ratio."""
    groups: dict[str, set[str]] = {}
    for row in rows:
        if row["split"] == "train":
            groups.setdefault(row["age_group"], set()).add(row["speaker_id"])
    total = sum(len(speakers) for speakers in groups.values())
    rng = np.random.default_rng(SEED)
    val = []
    for group in sorted(groups):
        speakers = sorted(groups[group])
        k = round(n_val * len(speakers) / total)
        val += sorted(rng.choice(speakers, size=k, replace=False).tolist())
    return val


def load_dataset(data_dir: Path, n_val_speakers: int) -> Dataset:
    """Load manifest and features, check that they match and split the speakers."""
    rows = read_manifest()
    features = np.load(features_dir(data_dir) / "wav2vec2_meanpool.npy").astype(np.float32)
    meta = json.loads((features_dir(data_dir) / "wav2vec2_meanpool.json").read_text(encoding="utf-8"))
    if meta["utt_ids"] != [row["utt_id"] for row in rows]:
        raise ValueError("feature file does not match the manifest; rerun extract_features.py")

    val_speakers = split_speakers(rows, n_val_speakers)
    part = np.array(["test" if r["split"] == "test" else "val" if r["speaker_id"] in val_speakers else "fit"
                     for r in rows])
    scores = np.array([[float(row[t]) for t in TARGETS] for row in rows])
    fit_idx, val_idx, test_idx = (np.where(part == p)[0] for p in ("fit", "val", "test"))
    return Dataset(rows, features, scores, meta, val_speakers, fit_idx, val_idx, test_idx)


def validation_curve(data: Dataset) -> np.ndarray:
    """Validation PCC for every (layer, alpha, target); one multi-output fit covers all targets."""
    curve = np.zeros((data.features.shape[1], len(ALPHAS), len(TARGETS)))
    for layer in range(data.features.shape[1]):
        for a, alpha in enumerate(ALPHAS):
            model = fit(data.features[data.fit_idx, layer], data.scores[data.fit_idx], alpha)
            pred = model.predict(data.features[data.val_idx, layer])
            curve[layer, a] = [pcc(data.scores[data.val_idx, t], pred[:, t]) for t in range(len(TARGETS))]
        print(f"layer {layer:2d}: best val PCC " + " ".join(
            f"{t}={curve[layer, :, i].max():.3f}" for i, t in enumerate(TARGETS)))
    return curve


def evaluate(data: Dataset, curve: np.ndarray) -> tuple[dict, np.ndarray]:
    """Per target: best (layer, alpha) on validation, refit on all training data, score on test."""
    results, test_pred = {}, np.zeros((len(data.test_idx), len(TARGETS)))
    for i, target in enumerate(TARGETS):
        layer, a = np.unravel_index(np.argmax(curve[:, :, i]), curve[:, :, i].shape)
        layer, alpha = int(layer), ALPHAS[a]
        x, y = data.features[:, layer], data.scores[:, i]
        val_pred = np.clip(fit(x[data.fit_idx], y[data.fit_idx], alpha).predict(x[data.val_idx]), *SCORE_RANGE)
        test_pred[:, i] = np.clip(fit(x[data.train_idx], y[data.train_idx], alpha).predict(x[data.test_idx]),
                                  *SCORE_RANGE)
        y_val, y_test = y[data.val_idx], y[data.test_idx]
        results[target] = {
            "layer": layer,
            "alpha": alpha,
            "val": {"pcc": pcc(y_val, val_pred), "mse": mse(y_val, val_pred)},
            "test": {"pcc": pcc(y_test, test_pred[:, i]), "mse": mse(y_test, test_pred[:, i])},
            "test_mse_predict_mean": mse(y_test, np.full(len(y_test), y[data.train_idx].mean())),
        }
    return results, test_pred


def write_predictions(data: Dataset, test_pred: np.ndarray, path: Path) -> None:
    """One row per test recording with the expert and the predicted scores."""
    headers = ["utt_id", "speaker_id", "age_group"] + [c for t in TARGETS for c in (t, f"{t}_pred")]
    rows = [
        [data.rows[idx]["utt_id"], data.rows[idx]["speaker_id"], data.rows[idx]["age_group"]]
        + [v for i in range(len(TARGETS)) for v in (data.scores[idx, i], round(test_pred[j, i], 3))]
        for j, idx in enumerate(data.test_idx)
    ]
    write_csv(path, headers, rows)


def main():
    """Validation curve, final evaluation on test, outputs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--val-speakers", type=int, default=15)
    args = parser.parse_args()

    data = load_dataset(args.data_dir, args.val_speakers)
    curve = validation_curve(data)
    results, test_pred = evaluate(data, curve)
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": {k: data.feature_meta[k] for k in ("model", "pooling", "shape")},
        "regressor": "StandardScaler + Ridge",
        "alphas": ALPHAS,
        "sizes": {"fit": len(data.fit_idx), "val": len(data.val_idx), "train": len(data.train_idx),
                  "test": len(data.test_idx)},
        "results": results,
        "val_pcc_by_layer": {t: curve[:, :, i].max(axis=1).round(4).tolist() for i, t in enumerate(TARGETS)},
    }
    write_json(OUTPUT_DIR / "sentence_scores_split.json", {"seed": SEED, "val_speakers": data.val_speakers})
    write_json(OUTPUT_DIR / "sentence_scores_baseline.json", summary)
    write_predictions(data, test_pred, OUTPUT_DIR / "sentence_scores_test_predictions.csv")
    print(f"Wrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
