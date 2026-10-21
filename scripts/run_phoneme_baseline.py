"""Run a pretrained wav2vec 2.0 phoneme recognizer on sample recordings.

Model:   facebook/wav2vec2-lv-60-espeak-cv-ft (CTC, eSpeak IPA phonemes)
Samples: hf-internal-testing/librispeech_asr_dummy (LibriSpeech clean/validation)

Outputs (written to outputs/):
  phoneme_baseline.json  per-sample predictions + timing
  phoneme_baseline.md    human-readable table
  environment.json       versions of Python, libraries, CUDA, GPU and model revision
"""

import argparse
import io
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"
DATASET_FILE = "clean/validation-00000-of-00001.parquet"
SAMPLE_RATE = 16_000
ROOT = Path(__file__).resolve().parent.parent


def load_samples(n):
    path = hf_hub_download(DATASET_ID, DATASET_FILE, repo_type="dataset")
    table = pq.read_table(path).slice(0, n).to_pylist()
    samples = []
    for row in table:
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{row['id']}: expected {SAMPLE_RATE} Hz, got {sr}")
        samples.append({"id": row["id"], "text": row["text"], "audio": audio})
    return samples


def collect_environment(device):
    api = HfApi()
    packages = [
        "torch", "torchaudio", "transformers", "accelerate", "datasets", "huggingface-hub",
        "tokenizers", "safetensors", "librosa", "soundfile", "numpy", "scipy", "pyarrow",
    ]
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: version(p) for p in packages},
        "cuda": {
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "run_device": str(device),
        "model": {"id": MODEL_ID, "revision": api.model_info(MODEL_ID).sha},
        "dataset": {
            "id": DATASET_ID,
            "file": DATASET_FILE,
            "revision": api.dataset_info(DATASET_ID).sha,
        },
    }
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--num-samples", type=int, default=5)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    # do_phonemize=False: we only decode model output, so the phonemizer/eSpeak backend is not needed.
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(MODEL_ID, do_phonemize=False)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device).eval()

    samples = load_samples(args.num_samples)
    results = []
    for s in samples:
        inputs = feature_extractor(s["audio"], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device)).logits
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        ids = torch.argmax(logits, dim=-1)[0].cpu()
        phonemes = tokenizer.decode(ids)
        duration = len(s["audio"]) / SAMPLE_RATE
        results.append({
            "id": s["id"],
            "duration_s": round(duration, 2),
            "inference_s": round(elapsed, 3),
            "reference_text": s["text"],
            "predicted_phonemes": phonemes,
        })
        print(f"[{s['id']}] {duration:.1f}s audio, {elapsed:.2f}s inference\n  text: {s['text']}\n  ipa:  {phonemes}\n")

    env = collect_environment(device)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "phoneme_baseline.json").write_text(
        json.dumps({"model": MODEL_ID, "device": str(device), "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        f"# wav2vec 2.0 phoneme baseline\n",
        f"Model: `{MODEL_ID}` · Device: `{device}` · Run: {env['timestamp_utc']}\n",
        "| ID | Duration (s) | Inference (s) | Reference text | Predicted phonemes (IPA) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['id']} | {r['duration_s']} | {r['inference_s']} | {r['reference_text']} | {r['predicted_phonemes']} |")
    (out_dir / "phoneme_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote results and environment to {out_dir}")


if __name__ == "__main__":
    main()
