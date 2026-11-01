"""Extract utterance-level wav2vec 2.0 features for every speechocean762 recording.

Each recording is passed once through the pretrained model (weights are not changed).
The hidden states of every layer are mean-pooled over time, giving one
(num_layers, hidden_size) array per utterance.

Input:   outputs/speechocean762_manifest.csv and data/speechocean762/ (prepare_speechocean762.py)
Outputs (written to data/features/, not tracked by git):
  wav2vec2_meanpool.npy   float16 array, shape (utterances, layers, hidden_size)
  wav2vec2_meanpool.json  utterance order, model revision, shapes, run info
"""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import HfApi
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SAMPLE_RATE = 16_000
ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    corpus_dir = args.data_dir / "speechocean762"
    out_dir = args.data_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "outputs" / "speechocean762_manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device).eval()
    num_layers = model.config.num_hidden_layers + 1  # CNN/projection output + each transformer layer
    features = np.zeros((len(rows), num_layers, model.config.hidden_size), dtype=np.float16)

    start = time.perf_counter()
    for i, row in enumerate(rows):
        audio, sr = sf.read(corpus_dir / row["wav_path"], dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{row['utt_id']}: expected {SAMPLE_RATE} Hz, got {sr}")
        inputs = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(inputs.input_values.to(device), output_hidden_states=True).hidden_states
        features[i] = torch.stack(hidden)[:, 0].mean(dim=1).float().cpu().numpy()
        if (i + 1) % 500 == 0 or i + 1 == len(rows):
            print(f"{i + 1}/{len(rows)} utterances, {time.perf_counter() - start:.0f}s")

    np.save(out_dir / "wav2vec2_meanpool.npy", features)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"id": MODEL_ID, "revision": HfApi().model_info(MODEL_ID).sha},
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "pooling": "mean over time of each hidden state",
        "shape": list(features.shape),
        "elapsed_s": round(time.perf_counter() - start, 1),
        "utt_ids": [r["utt_id"] for r in rows],
    }
    (out_dir / "wav2vec2_meanpool.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {features.shape} features to {out_dir}")


if __name__ == "__main__":
    main()
