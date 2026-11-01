"""Train a sentence-level pronunciation score predictor on frozen wav2vec 2.0 features.

A ridge regressor maps the mean-pooled hidden state of one model layer to the expert
sentence scores (accuracy, fluency, prosodic, total). The layer and the regularisation
strength are chosen on a speaker-disjoint validation split taken from the training set;
the chosen setting is then refitted on the whole training set and scored once on test.

Completeness is not modelled: 4975 of 5000 utterances have the maximum score of 10.

Input:   outputs/speechocean762_manifest.csv, data/features/wav2vec2_meanpool.npy (extract_features.py)
Outputs (written to outputs/):
  sentence_scores_split.json            validation speakers
  sentence_scores_baseline.json         chosen settings, metrics, per-layer validation curve
  sentence_scores_baseline.md           report
  sentence_scores_test_predictions.csv  per-utterance test predictions
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ("accuracy", "fluency", "prosodic", "total")
ALPHAS = (1, 10, 100, 1000, 10000, 100000)
SEED = 0
ROOT = Path(__file__).resolve().parent.parent


def pcc(y, p):
    return float(np.corrcoef(y, p)[0, 1])


def mse(y, p):
    return float(np.mean((y - p) ** 2))


def split_speakers(rows, n_val):
    """Pick n_val training speakers for validation, keeping the child/adult ratio."""
    groups = {}
    for r in rows:
        if r["split"] == "train":
            groups.setdefault(r["age_group"], set()).add(r["speaker_id"])
    total = sum(len(s) for s in groups.values())
    rng = np.random.default_rng(SEED)
    val = []
    for group in sorted(groups):
        speakers = sorted(groups[group])
        k = round(n_val * len(speakers) / total)
        val += sorted(rng.choice(speakers, size=k, replace=False).tolist())
    return val


def fit(x, y, alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--val-speakers", type=int, default=15)
    args = parser.parse_args()

    out_dir = ROOT / "outputs"
    with open(out_dir / "speechocean762_manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    feats = np.load(args.data_dir / "features" / "wav2vec2_meanpool.npy").astype(np.float32)
    meta = json.loads((args.data_dir / "features" / "wav2vec2_meanpool.json").read_text(encoding="utf-8"))
    if meta["utt_ids"] != [r["utt_id"] for r in rows]:
        raise ValueError("feature file does not match the manifest; rerun extract_features.py")

    val_speakers = split_speakers(rows, args.val_speakers)
    part = np.array([
        "test" if r["split"] == "test" else "val" if r["speaker_id"] in val_speakers else "fit"
        for r in rows
    ])
    y = np.array([[float(r[t]) for t in TARGETS] for r in rows])
    fit_idx, val_idx, test_idx = (np.where(part == p)[0] for p in ("fit", "val", "test"))
    train_idx = np.concatenate([fit_idx, val_idx])

    # Validation PCC for every (layer, alpha); one multi-output fit covers all targets.
    num_layers = feats.shape[1]
    curve = np.zeros((num_layers, len(ALPHAS), len(TARGETS)))
    for layer in range(num_layers):
        for a, alpha in enumerate(ALPHAS):
            pred = fit(feats[fit_idx, layer], y[fit_idx], alpha).predict(feats[val_idx, layer])
            curve[layer, a] = [pcc(y[val_idx, t], pred[:, t]) for t in range(len(TARGETS))]
        print(f"layer {layer:2d}: best val PCC " + " ".join(
            f"{t}={curve[layer, :, i].max():.3f}" for i, t in enumerate(TARGETS)))

    results = {}
    test_pred = np.zeros((len(test_idx), len(TARGETS)))
    for i, target in enumerate(TARGETS):
        layer, a = np.unravel_index(np.argmax(curve[:, :, i]), curve[:, :, i].shape)
        layer, alpha = int(layer), ALPHAS[a]
        val_pred = np.clip(fit(feats[fit_idx, layer], y[fit_idx, i], alpha).predict(feats[val_idx, layer]), 0, 10)
        test_pred[:, i] = np.clip(fit(feats[train_idx, layer], y[train_idx, i], alpha).predict(feats[test_idx, layer]), 0, 10)
        train_mean = y[train_idx, i].mean()
        results[target] = {
            "layer": layer,
            "alpha": alpha,
            "val": {"pcc": pcc(y[val_idx, i], val_pred), "mse": mse(y[val_idx, i], val_pred)},
            "test": {"pcc": pcc(y[test_idx, i], test_pred[:, i]), "mse": mse(y[test_idx, i], test_pred[:, i])},
            "test_mse_predict_mean": mse(y[test_idx, i], np.full(len(test_idx), train_mean)),
        }

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": {"model": meta["model"], "pooling": meta["pooling"], "shape": meta["shape"]},
        "regressor": "StandardScaler + Ridge",
        "alphas": ALPHAS,
        "sizes": {"fit": len(fit_idx), "val": len(val_idx), "train": len(train_idx), "test": len(test_idx)},
        "results": results,
        "val_pcc_by_layer": {t: curve[:, :, i].max(axis=1).round(4).tolist() for i, t in enumerate(TARGETS)},
    }
    (out_dir / "sentence_scores_split.json").write_text(
        json.dumps({"seed": SEED, "val_speakers": val_speakers}, indent=2), encoding="utf-8")
    (out_dir / "sentence_scores_baseline.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(out_dir / "sentence_scores_test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "speaker_id", "age_group"] + [c for t in TARGETS for c in (t, f"{t}_pred")])
        for j, idx in enumerate(test_idx):
            r = rows[idx]
            w.writerow([r["utt_id"], r["speaker_id"], r["age_group"]]
                       + [v for i in range(len(TARGETS)) for v in (y[idx, i], round(test_pred[j, i], 3))])

    lines = [
        "# Cümle düzeyi puan tahmini — temel model\n",
        f"Özellikler: `{meta['model']['id']}` (değiştirilmeden), katman başına zaman ortalaması · "
        f"Tahminci: StandardScaler + Ridge · Çalıştırma: {summary['timestamp_utc']}\n",
        f"Veri: eğitim {len(fit_idx)} + doğrulama {len(val_idx)} kayıt "
        f"({len(val_speakers)} konuşmacı, çocuk/yetişkin oranı korunarak), test {len(test_idx)} kayıt. "
        "Katman ve alpha yalnızca doğrulama kümesinde seçildi; seçilen ayar tüm eğitim verisiyle yeniden eğitilip "
        "test kümesinde bir kez ölçüldü.\n",
        "| Puan | Katman | Alpha | Doğrulama PCC | Doğrulama MSE | Test PCC | Test MSE | Test MSE (ortalama tahmini) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t, r in results.items():
        lines.append(f"| {t} | {r['layer']} | {r['alpha']} | {r['val']['pcc']:.3f} | {r['val']['mse']:.3f} | "
                     f"{r['test']['pcc']:.3f} | {r['test']['mse']:.3f} | {r['test_mse_predict_mean']:.3f} |")
    lines += [
        "\nPCC: tahmin ile uzman puanı arasındaki Pearson korelasyonu (1 = tam uyum). "
        "MSE: ortalama kare hata (0–10 ölçeğinde). Son sütun, her kayda eğitim ortalamasını veren modelin hatasıdır.\n",
        "Completeness modellenmedi: 5000 kaydın 4975'inde puan 10.\n",
        "## Katmana göre doğrulama PCC (en iyi alpha)\n",
        "| Katman | " + " | ".join(TARGETS) + " |",
        "| --- |" + " --- |" * len(TARGETS),
    ]
    for layer in range(num_layers):
        lines.append(f"| {layer} | " + " | ".join(f"{curve[layer, :, i].max():.3f}" for i in range(len(TARGETS))) + " |")
    (out_dir / "sentence_scores_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
