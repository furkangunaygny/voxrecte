"""Extract utterance-level wav2vec 2.0 features for every speechocean762 recording.

Additional experiment, not a step of the project plan (kept in extras/).

Each recording is passed once through the pretrained model (weights are not changed). The hidden
states of every layer are averaged over time, giving one (num_layers, hidden_size) array per
utterance. Used by train_sentence_scores.py.

Input:   outputs/speechocean762_manifest.csv and data/speechocean762/ (prepare_speechocean762.py)
Outputs (written to data/features/, not tracked by git):
  wav2vec2_meanpool.npy   float16 array, shape (utterances, layers, hidden_size)
  wav2vec2_meanpool.json  utterance order, model revision, shapes, run info
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from voxrecte.files import write_json
from voxrecte.paths import DATA_DIR, corpus_dir, features_dir
from voxrecte.recognizer import MODEL_ID, PhonemeRecognizer, load_audio, model_revision
from voxrecte.speechocean import read_manifest

FEATURES_NAME = "wav2vec2_meanpool"
PROGRESS_EVERY = 500


def main():
    """Extract and save the features of every recording, in manifest order."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()

    rows = read_manifest()
    out_dir = features_dir(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recognizer = PhonemeRecognizer(cpu=args.cpu)
    features = np.zeros((len(rows), recognizer.num_layers, recognizer.hidden_size), dtype=np.float16)

    start = time.perf_counter()
    for i, row in enumerate(rows):
        features[i] = recognizer.layer_means(load_audio(corpus_dir(args.data_dir) / row["wav_path"]))
        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == len(rows):
            print(f"{i + 1}/{len(rows)} utterances, {time.perf_counter() - start:.0f}s")

    np.save(out_dir / f"{FEATURES_NAME}.npy", features)
    write_json(out_dir / f"{FEATURES_NAME}.json", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"id": MODEL_ID, "revision": model_revision()},
        "device": str(recognizer.device),
        "device_name": torch.cuda.get_device_name(0) if recognizer.device.type == "cuda" else None,
        "pooling": "mean over time of each hidden state",
        "shape": list(features.shape),
        "elapsed_s": round(time.perf_counter() - start, 1),
        "utt_ids": [row["utt_id"] for row in rows],
    })
    print(f"Wrote {features.shape} features to {out_dir}")


if __name__ == "__main__":
    main()
